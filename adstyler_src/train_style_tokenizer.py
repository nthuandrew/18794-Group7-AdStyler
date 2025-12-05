import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import clip
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

# --- 1. Model Definitions (Must match training/infer structure) ---

class StyleClip(nn.Module):
    def __init__(self, feat_dim=512, clip_model_name="ViT-B/32"):
        super(StyleClip, self).__init__()
        self.clip, self.transform = clip.load(clip_model_name, device="cpu") 
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
        feat = feat.to(torch.float32)
        feat = self.head(feat)
        return feat

class ClipMlp(nn.Module):
    def __init__(self, feat_dim=768, out_dim=1024, clip_model_name="ViT-B/32"):
        super(ClipMlp, self).__init__()
        self.encoder = StyleClip(feat_dim=feat_dim, clip_model_name=clip_model_name)
        self.head = nn.Linear(feat_dim, out_dim)
    
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        feat = F.normalize(feat, dim=1) 
        return feat

class StyleTokenizer(nn.Module):
    def __init__(self, input_size=1024, intermediate_size=512, out_size=768, n_tokens=8):
        super(StyleTokenizer, self).__init__()
        # Simple MLP Projector
        self.proj = nn.Sequential(
            nn.Linear(input_size, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, out_size * n_tokens),
        )
        self.n_tokens = n_tokens
        self.out_size = out_size

    def forward(self, x):
        x = self.proj(x.to(torch.float32))
        # Reshape to (batch, n_tokens, out_size)
        x = x.reshape(x.shape[0], self.n_tokens, self.out_size)
        return x

# --- 2. Dataset ---

class StyleFinetuneDataset(Dataset):
    """
    Returns:
    1. pixel_values: Image for VAE (Target)
    2. style_image: Image for Style Encoder (Reference)
    3. input_ids: Tokenized generic text prompt (Content)
    """
    def __init__(self, root_dir, tokenizer, style_preprocess, size=512):
        self.root_dir = root_dir
        self.image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Collect images
        for filename in os.listdir(root_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                if "____" in filename: # Simple check for your file format
                    self.image_paths.append(os.path.join(root_dir, filename))
        
        print(f"Found {len(self.image_paths)} images for tokenizer training.")

        self.tokenizer = tokenizer
        self.style_preprocess = style_preprocess
        
        # Transform for VAE (Target Image)
        self.vae_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
        ])
        
        # Generic content prompt
        self.generic_prompt = "a photo" 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 1. Target Image (for VAE)
            pixel_values = self.vae_transform(image)
            
            # 2. Reference Image (for Style Encoder)
            # In self-reconstruction, target and reference are the same image
            style_image = self.style_preprocess(image)
            
            # 3. Text Input (Generic content description)
            text_inputs = self.tokenizer(
                self.generic_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids[0]
            
            return pixel_values, style_image, input_ids
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data to prevent crash
            return torch.zeros(3, 512, 512), torch.zeros(3, 224, 224), torch.zeros(77, dtype=torch.long)

# --- 3. Training Loop ---

def train_style_tokenizer():
    # --- Configuration ---
    IMAGE_DIR = "style30k/images_top10"
    STYLE_ENCODER_PATH = "style_encoder.pth"
    SAVE_PATH = "style_tokenizer.pth" # Usually mapped to 'linear.ckpt' in infer.py
    
    SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    BATCH_SIZE = 4 # SD training takes VRAM, adjust if OOM
    EPOCHS = 5
    LR = 1e-4 # Learning rate for the MLP
    
    print(f"Using device: {DEVICE}")

    # 1. Load Frozen Models (SD & Style Encoder)
    print("Loading Stable Diffusion (Frozen)...")
    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD_MODEL_ID, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(SD_MODEL_ID, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(SD_MODEL_ID, subfolder="unet").to(DEVICE)
    noise_scheduler = DDPMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")
    
    # Freeze SD
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    print("Loading Style Encoder (Frozen)...")
    style_encoder = ClipMlp(feat_dim=768, out_dim=1024, clip_model_name="ViT-B/32")
    # Load weights
    ckpt = torch.load(STYLE_ENCODER_PATH, map_location="cpu")
    style_encoder.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    style_encoder.to(DEVICE)
    style_encoder.requires_grad_(False)
    
    style_preprocess = style_encoder.encoder.get_transform()

    # 2. Initialize Trainable Model (Style Tokenizer)
    print("Initializing Style Tokenizer (Trainable)...")
    # input_size=1024 (from StyleEncoder), out_size=768 (for SD v1.5)
    style_tokenizer = StyleTokenizer(input_size=1024, out_size=768, n_tokens=8)
    style_tokenizer.to(DEVICE)
    style_tokenizer.train()

    # 3. Setup Data & Optimizer
    dataset = StyleFinetuneDataset(IMAGE_DIR, tokenizer, style_preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(style_tokenizer.parameters(), lr=LR)
    
    # 4. Training Loop
    print(f"Start training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for pixel_values, style_images, input_ids in progress_bar:
            pixel_values = pixel_values.to(DEVICE) # Target images
            style_images = style_images.to(DEVICE) # Reference images
            input_ids = input_ids.to(DEVICE)       # Text tokens
            
            # --- A. Prepare Latents (Target) ---
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215 # Scaling factor for SD v1.5
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()
            
            # Add noise to latents (Forward Diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # --- B. Prepare Conditions (Prompt) ---
            # 1. Get Style Embeddings
            with torch.no_grad():
                style_emb = style_encoder(style_images) # [bsz, 1024]
            
            # 2. Project to Style Tokens (Trainable Step)
            style_tokens = style_tokenizer(style_emb) # [bsz, 8, 768]
            
            # 3. Get Text Tokens
            with torch.no_grad():
                text_emb = text_encoder(input_ids)[0] # [bsz, 77, 768]
            
            # 4. Concatenate: [Style, Text]
            # This is how we inject style into the prompt
            # Result shape: [bsz, 8 + 77, 768]
            encoder_hidden_states = torch.cat([style_tokens, text_emb], dim=1)
            
            # --- C. Prediction & Loss ---
            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Calculate Loss (MSE between actual noise and predicted noise)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # --- D. Backprop ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(dataloader):.4f}")
        
    # 5. Save
    print(f"Saving Style Tokenizer to {SAVE_PATH}...")
    torch.save(style_tokenizer.state_dict(), SAVE_PATH)
    print("Training Complete!")

if __name__ == "__main__":
    train_style_tokenizer()