import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Hugging Face & Diffusers
from transformers import CLIPTokenizer
from clip_tokenizer import CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# --- 1. Metadata Projector Definition ---
class MetadataProjector(nn.Module):
    """
    Projects normalized coordinates [x, y, w, h] into the embedding space.
    Output shape: [batch, 1, hidden_dim]
    """
    def __init__(self, input_dim=4, hidden_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # x: [batch, 4]
        feat = self.net(x)
        # Reshape to be a single token: [batch, 1, 768]
        return feat.unsqueeze(1)

# --- 2. Dataset Class ---

class AdImageNetDataset(Dataset):
    def __init__(self, dataset_path, json_path, tokenizer, size=512):
        self.size = size
        self.tokenizer = tokenizer
        
        print(f"Loading image dataset from {dataset_path}...")
        self.ds = load_from_disk(dataset_path)
        
        print(f"Loading layout data from {json_path}...")
        with open(json_path, 'r') as f:
            self.layout_data = json.load(f)
            
        if len(self.ds) != len(self.layout_data):
            print(f"Warning: Dataset size ({len(self.ds)}) and JSON size ({len(self.layout_data)}) mismatch!")
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            # 1. Get Image
            item = self.ds[idx]
            image = item['image'].convert("RGB")
            pixel_values = self.image_transforms(image)
            
            # 2. Get Ad Copy & Layout
            meta_entry = self.layout_data[idx]
            ad_copy = meta_entry.get('ad_copy', '').replace('\n', ' ')
            
            layout = meta_entry.get('text_layout', {})
            x = float(layout.get('x', 0))
            y = float(layout.get('y', 0))
            w = float(layout.get('width', 0))
            h = float(layout.get('height', 0))
            metadata_vec = torch.tensor([x, y, w, h], dtype=torch.float32)
            
            # 3. Tokenize Ad Copy (Using CLIP Tokenizer)
            # Max length 77 is standard for SD v1.5
            text_inputs = self.tokenizer(
                ad_copy,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids[0]
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "metadata_vec": metadata_vec
            }
            
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            # Dummy fallback
            return {
                "pixel_values": torch.zeros(3, self.size, self.size),
                "input_ids": torch.zeros(77, dtype=torch.long),
                "metadata_vec": torch.zeros(4, dtype=torch.float32)
            }

# --- 3. Main Training Function ---

def main():
    # --- Config ---
    DATASET_PATH = "./AdImageNet/train"
    JSON_PATH = "train_layout.json"
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    OUTPUT_DIR = "ad_lora_clip_output" # Changed output dir name
    
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4 # Standard LR for LoRA
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load SD Components
    print("Loading Stable Diffusion Components...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")

    # Freeze everything initially
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False) # Keeping CLIP frozen for robustness
    unet.requires_grad_(False)
    
    # Move frozen models to device
    vae.to(DEVICE)
    text_encoder.to(DEVICE)
    # UNet will be moved after adding LoRA

    # 2. Initialize Metadata Projector (Trainable)
    print("Initializing Metadata Projector...")
    meta_projector = MetadataProjector().to(DEVICE)
    meta_projector.train()

    # 3. Setup LoRA for U-Net
    print("Setting up LoRA for U-Net...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config).to(DEVICE)
    unet.train()
    unet.print_trainable_parameters()

    # 4. Dataloader
    dataset = AdImageNetDataset(DATASET_PATH, JSON_PATH, tokenizer)
    # use 10% of dataset for quick testing
    dataset = torch.utils.data.Subset(dataset, list(range(int(0.01 * len(dataset)))))
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 5. Optimizer
    # Optimize LoRA weights AND Metadata Projector
    params_to_optimize = [
        *unet.parameters(),
        *meta_projector.parameters()
    ]
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=LEARNING_RATE,
        weight_decay=1e-2
    )
    
    # 6. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            metadata_vec = batch["metadata_vec"].to(DEVICE)

            # --- Forward Pass ---
            
            # A. Encode Ad Copy (CLIP) -> [B, 77, 768]
            with torch.no_grad():
                text_embeds = text_encoder(input_ids)[0]

            # B. Encode Metadata (Projector) -> [B, 1, 768]
            meta_embeds = meta_projector(metadata_vec)

            # C. Concatenate: [Ad Copy, Metadata] -> [B, 78, 768]
            # This combined embedding is what U-Net sees
            encoder_hidden_states = torch.cat([text_embeds, meta_embeds], dim=1)

            # D. Encode Image to Latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

            # E. Add Noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # F. Predict Noise (U-Net with LoRA)
            model_pred = unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states=encoder_hidden_states
            ).sample

            # G. Loss
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save LoRA
        unet.save_pretrained(os.path.join(save_path, "unet_lora"))
        # Save Projector
        torch.save(meta_projector.state_dict(), os.path.join(save_path, "meta_projector.pth"))
        
        print(f"Saved checkpoint to {save_path}")

    print("Training finished!")

if __name__ == "__main__":
    main()