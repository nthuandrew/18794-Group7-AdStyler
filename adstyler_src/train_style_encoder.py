import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# --- 1. Model Architecture (Copied and adapted from infer.py) ---
# Must keep the architecture consistent so infer.py can load the weights.

class StyleClip(nn.Module):
    """backbone + projection head"""
    def __init__(self, feat_dim=512, clip_model_name="ViT-B/32"):
        super(StyleClip, self).__init__()
        # Load CLIP model directly, do not rely on external path variables
        self.clip, self.transform = clip.load(clip_model_name, device="cpu") # Load onto CPU first
        
        # Based on ViT-B/32 output dimension, usually 512
        dim_in = self.clip.visual.output_dim 
        
        model_list = []
        for i in range(4):
            model_list += [nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True)]
        
        model_list.append(nn.Linear(dim_in, feat_dim))
        self.head = nn.Sequential(*model_list)

        self.freeze_encoder()

    def get_transform(self):
        return self.transform

    def freeze_encoder(self, freeze=True):
        for param in self.clip.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        feat = self.clip.encode_image(x)
        feat = feat.to(torch.float32) # Ensure float32 for the head
        feat = self.head(feat)
        return feat

class ClipMlp(nn.Module):
    """The full Style Encoder Wrapper"""
    def __init__(self, feat_dim=768, out_dim=1024, clip_model_name="ViT-B/32"):
        super(ClipMlp, self).__init__()
        # Note: feat_dim here refers to the output of StyleClip's head (default 768 in load_style_encoder)
        # out_dim is the dimension of the final projection head (1024).
        
        self.encoder = StyleClip(feat_dim=feat_dim, clip_model_name=clip_model_name)
        self.head = nn.Linear(feat_dim, out_dim)
    
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        feat = F.normalize(feat, dim=1) # Normalize embeddings for contrastive learning
        return feat

# --- 2. Loss Function: Supervised Contrastive Loss ---

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                  has the same class as sample i. Can be asymmetric.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            # *** POSITIVE/NEGATIVE PAIR LOGIC IS HERE ***
            # Create a mask where mask[i][j] is 1 if label[i] == label[j] (Positive Pair)
            # and 0 otherwise (Negative Pair).
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (do not compare an image with itself)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # Add epsilon to prevent division by zero
        epsilon = 1e-8
        # Only divide where mask sum is > 0 to avoid NaNs, otherwise the term contributes 0 loss
        denominator = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (denominator + epsilon)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# --- 3. Dataset ---

class StyleDataset(Dataset):
    def __init__(self, root_dir, preprocess, separator="____"):
        self.root_dir = root_dir
        self.separator = separator
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Collect all Style IDs to build a mapping (str -> int)
        unique_styles = set()
        
        print(f"Scanning {root_dir}...")
        for filename in os.listdir(root_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                if self.separator in filename:
                    # Parse Style ID: "s0283" form "s0283____..."
                    style_id = filename.split(self.separator)[0]
                    
                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.labels.append(style_id)
                    unique_styles.add(style_id)
        
        # Create mapping from Style ID string to Integer index
        self.style2idx = {style: idx for idx, style in enumerate(sorted(list(unique_styles)))}
        self.idx2style = {idx: style for style, idx in self.style2idx.items()}
        
        print(f"Found {len(self.image_paths)} images with {len(unique_styles)} unique styles.")
        # print(f"Styles found: {self.style2idx.keys()}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        style_id = self.labels[idx]
        label_idx = self.style2idx[style_id]
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return random noise to prevent crash (or handle appropriately)
            image = torch.zeros((3, 224, 224))
        
        return image, label_idx

# --- 4. Main Training Loop ---

def train():
    # --- Config ---
    IMAGE_DIR = "style30k/images_top10" # Change this to your image directory
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "style_encoder.pth"
    CLIP_MODEL = "ViT-B/32" 
    
    # Check directory
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory '{IMAGE_DIR}' not found.")
        return

    # 1. Initialize Model
    # The parameters feat_dim=768 (StyleClip output) and out_dim=1024 (Final embedding dim)
    # must match what infer.py expects.
    model = ClipMlp(feat_dim=768, out_dim=1024, clip_model_name=CLIP_MODEL)
    model = model.to(DEVICE)
    
    # Get preprocess transform (located inside model.encoder.transform)
    preprocess = model.encoder.get_transform()

    # 2. Prepare Data
    dataset = StyleDataset(IMAGE_DIR, preprocess=preprocess)
    if len(dataset) == 0:
        print("No images found. Exiting.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # 3. Optimizer & Loss
    # Only train the MLP head. The CLIP encoder is frozen in StyleClip.__init__
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = SupConLoss(temperature=0.07)

    # 4. Training Loop
    model.train()
    print(f"Start training on {DEVICE} for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in progress_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward Pass
            # SupConLoss expects input shape: [bsz, n_views, feat_dim]
            # Since we are not doing multi-view augmentation (SimCLR style), n_views=1
            # features shape transformation: [bsz, dim] -> [bsz, 1, dim]
            features = model(images)
            features = features.unsqueeze(1) 
            
            # Calculate Loss (Contrastive learning happens here based on labels)
            loss = criterion(features, labels)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

    # 5. Save Model
    # The format expected by infer.py is a state_dict containing a "model" key.
    # Also, keys with "module." prefix are handled automatically by load_style_encoder.
    print(f"Saving model to {SAVE_PATH}...")
    torch.save({"model": model.state_dict()}, SAVE_PATH)
    print("Done!")

if __name__ == "__main__":
    train()