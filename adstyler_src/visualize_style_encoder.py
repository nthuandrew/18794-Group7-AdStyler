import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm

# --- 1. Model Architecture (Same as training script) ---
# We verify the model structure here to load the checkpoint correctly.

class StyleClip(nn.Module):
    """backbone + projection head"""
    def __init__(self, feat_dim=512, clip_model_name="ViT-B/32"):
        super(StyleClip, self).__init__()
        self.clip, self.transform = clip.load(clip_model_name, device="cpu") 
        dim_in = self.clip.visual.output_dim 
        
        model_list = []
        for i in range(4):
            model_list += [nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True)]
        
        model_list.append(nn.Linear(dim_in, feat_dim))
        self.head = nn.Sequential(*model_list)

    def get_transform(self):
        return self.transform

    def forward(self, x):
        feat = self.clip.encode_image(x)
        feat = feat.to(torch.float32)
        feat = self.head(feat)
        return feat

class ClipMlp(nn.Module):
    """The full Style Encoder Wrapper"""
    def __init__(self, feat_dim=768, out_dim=1024, clip_model_name="ViT-B/32"):
        super(ClipMlp, self).__init__()
        self.encoder = StyleClip(feat_dim=feat_dim, clip_model_name=clip_model_name)
        self.head = nn.Linear(feat_dim, out_dim)
    
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        feat = F.normalize(feat, dim=1) 
        return feat

# --- 2. Dataset (Same as training script) ---

class StyleDataset(Dataset):
    def __init__(self, root_dir, preprocess, separator="____"):
        self.root_dir = root_dir
        self.separator = separator
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        unique_styles = set()
        
        print(f"Scanning {root_dir}...")
        for filename in os.listdir(root_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                if self.separator in filename:
                    style_id = filename.split(self.separator)[0]
                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.labels.append(style_id)
                    unique_styles.add(style_id)
        
        # Sort keys to ensure consistent coloring
        self.style2idx = {style: idx for idx, style in enumerate(sorted(list(unique_styles)))}
        self.idx2style = {idx: style for style, idx in self.style2idx.items()}
        
        print(f"Found {len(self.image_paths)} images with {len(unique_styles)} unique styles.")

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
            image = torch.zeros((3, 224, 224))
        
        return image, label_idx

# --- 3. Inference and Visualization Logic ---

def extract_features(model, dataloader, device):
    """Run inference to get embeddings and labels for all images."""
    model.eval()
    all_features = []
    all_labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            
            # Forward pass
            features = model(images)
            
            # Move to CPU and numpy
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    # Concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_features, all_labels

def plot_tsne(features, labels, idx2style, save_path="tsne_visualization.png"):
    """Reduce dimensions using t-SNE and plot the result."""
    print("Running t-SNE (this might take a while)...")
    
    # Initialize t-SNE
    # n_components=2 for 2D plot
    # init='pca' usually helps with stability
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(features)
    
    # Plotting
    print("Plotting...")
    plt.figure(figsize=(16, 12))
    
    unique_labels = np.unique(labels)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label_idx in enumerate(unique_labels):
        # Select points belonging to this style
        indices = labels == label_idx
        style_name = idx2style[label_idx]
        
        plt.scatter(
            projections[indices, 0], 
            projections[indices, 1], 
            c=[colors[i]], 
            label=style_name, 
            alpha=0.7, 
            s=30 # point size
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Styles")
    plt.title("t-SNE Visualization of Style Embeddings")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()

def main():
    # --- Configuration ---
    IMAGE_DIR = "style30k/images_top10" # Your image directory
    MODEL_PATH = "style_encoder.pth"    # Your trained checkpoint
    CLIP_MODEL = "ViT-B/32"
    BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Checkpoint '{MODEL_PATH}' not found.")
        return

    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = ClipMlp(feat_dim=768, out_dim=1024, clip_model_name=CLIP_MODEL)
    
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    
    preprocess = model.encoder.get_transform()

    # 2. Prepare Data
    dataset = StyleDataset(IMAGE_DIR, preprocess=preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Extract Features
    features, labels = extract_features(model, dataloader, DEVICE)
    
    print(f"Extracted feature shape: {features.shape}")
    
    # 4. Visualization
    plot_tsne(features, labels, dataset.idx2style)

if __name__ == "__main__":
    main()