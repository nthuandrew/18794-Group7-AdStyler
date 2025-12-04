import torch
import os
import torch.nn as nn
from PIL import Image
import numpy as np

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from peft import PeftModel

# --- 1. Define Metadata Projector (Must match training) ---
class MetadataProjector(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, x):
        feat = self.net(x)
        return feat.unsqueeze(1) # [Batch, 1, 768]

def run_test():
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    
    # Paths to your trained Ad-LoRA results
    # 請修改為您訓練出的 checkpoint 路徑
    AD_CHECKPOINT_DIR = "ad_lora_clip_output/checkpoint-epoch-5" 
    UNET_LORA_PATH = os.path.join(AD_CHECKPOINT_DIR, "unet_lora")
    META_PROJECTOR_PATH = os.path.join(AD_CHECKPOINT_DIR, "meta_projector.pth")

    # Inputs
    AD_COPY = "A smartphone is on sale now!"
    # Layout: x, y, width, height (0-1)
    LAYOUT = [0.1, 0.1, 0.8, 0.2]
    
    OUTPUT_IMAGE = "output_ad_only.png"

    # --- Step 1: Load Basic SD Models ---
    print("Loading Stable Diffusion...")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae").to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet").to(DEVICE)
    scheduler = PNDMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")

    # --- Step 2: Load Trained Components ---
    print("Loading Ad-LoRA & Projector...")
    
    # Load LoRA
    unet = PeftModel.from_pretrained(unet, UNET_LORA_PATH)
    unet = unet.merge_and_unload() # Merge for inference
    
    # Load Metadata Projector
    meta_projector = MetadataProjector().to(DEVICE)
    if os.path.exists(META_PROJECTOR_PATH):
        meta_projector.load_state_dict(torch.load(META_PROJECTOR_PATH, map_location=DEVICE))
    else:
        print(f"Error: Meta projector not found at {META_PROJECTOR_PATH}")
        return

    # --- Step 3: Prepare Prompt Embeddings ---
    print("Constructing Embeddings...")
    
    # A. Conditional Embedding (The Ad)
    # 1. Text -> [1, 77, 768]
    text_input = tokenizer(
        AD_COPY, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeds = text_encoder(text_input.input_ids.to(DEVICE))[0]

    # 2. Layout -> [1, 1, 768]
    layout_tensor = torch.tensor(LAYOUT, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        meta_embeds = meta_projector(layout_tensor)

    # 3. Concat -> [1, 78, 768]
    prompt_embeds = torch.cat([text_embeds, meta_embeds], dim=1)

    # B. Negative Embedding (Unconditional)
    # Empty Text + Zero Metadata
    uncond_input = tokenizer(
        "", padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_text_embeds = text_encoder(uncond_input.input_ids.to(DEVICE))[0]
        
        # Zero Layout
        zero_layout = torch.zeros_like(layout_tensor)
        uncond_meta_embeds = meta_projector(zero_layout) # Or just torch.zeros_like(meta_embeds)
    
    negative_prompt_embeds = torch.cat([uncond_text_embeds, uncond_meta_embeds], dim=1)

    # --- Step 4: Generate ---
    print("Generating Image...")
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
        safety_checker=None, feature_extractor=None
    ).to(DEVICE)

    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]

    image.save(OUTPUT_IMAGE)
    print(f"Saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    run_test()