import torch
import os
import torch.nn as nn
from PIL import Image
import numpy as np

# SD & Hugging Face Components
from transformers import CLIPTokenizer
from clip_tokenizer import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from peft import PeftModel

# Custom Modules (From your files)
from infer import StableDiffusionPipelineWithStyle, StyleTokenizer, load_style_encoder
import random
import json

# --- 1. Define Helper Classes ---

class MetadataProjector(nn.Module):
    """
    Projects normalized coordinates [x, y, w, h] into the embedding space.
    Must match the architecture defined in train_ad_lora.py (CLIP version).
    """
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
class AdStylerCLIPWrapper(nn.Module):
    """
    Wraps CLIPTextModel + MetadataProjector to work with StableDiffusionPipelineWithStyle.
    Constructs the final prompt embedding: [Style(8) + Text(77) + Meta(1)] = 86 Tokens.
    """
    def __init__(self, clip_model, meta_projector, device):
        super().__init__()
        self.clip_model = clip_model
        self.meta_projector = meta_projector
        self.target_device = device
        self.dtype = clip_model.dtype 
        self.config = clip_model.config
        
        self.style_token_len = 8
        self.clip_max_len = 77 # CLIP standard max length

    def forward(self, input_ids=None, style_embeds=None, style_id=None, attention_mask=None, **kwargs):
        context = getattr(self, 'current_context', None)
        is_uncond = (context is None)
        
        # --- 1. 安全截斷 (Truncate) ---
        # infer.py 可能傳入長度 520 的 inputs
        # 我們必須同時截斷 input_ids 和 attention_mask
        if input_ids.shape[1] > self.clip_max_len:
            input_ids_for_clip = input_ids[:, :self.clip_max_len]
            if attention_mask is not None:
                attention_mask_for_clip = attention_mask[:, :self.clip_max_len]
            else:
                attention_mask_for_clip = None
        else:
            input_ids_for_clip = input_ids
            attention_mask_for_clip = attention_mask

        # --- 2. 數值鉗制 (Clamp) - 解決 CUDA Error 的關鍵 ---
        # infer.py 加入的 <style_token> ID 會超過 CLIP 原本的 vocab size
        # 這會導致 Embedding Lookup 越界崩潰。
        # 我們把所有未知的 ID 都變成 0 (或者是 End-of-Text token，這裡用 0 較安全)
        vocab_size = self.config.vocab_size
        if (input_ids_for_clip >= vocab_size).any():
            input_ids_for_clip = torch.where(
                input_ids_for_clip < vocab_size, 
                input_ids_for_clip, 
                torch.tensor(0, device=self.target_device)
            )

        # --- 3. Run CLIP ---
        # 現在輸入既短又安全了
        text_outputs = self.clip_model(
            input_ids=input_ids_for_clip, 
            attention_mask=attention_mask_for_clip, # 傳入截斷後的 mask
            output_hidden_states=True
        )
        text_embeds = text_outputs.last_hidden_state # [Batch, 77, 768]
        
        batch_size = text_embeds.shape[0]

        # --- 4. Encode Metadata ---
        if is_uncond:
            # Unconditional: Zero Metadata
            dummy_meta = torch.zeros(batch_size, 1, 768).to(self.target_device)
            meta_embeds = dummy_meta
        else:
            layout = context['layout']
            layout_tensor = torch.tensor(layout, dtype=torch.float32).unsqueeze(0).to(self.target_device)
            layout_tensor = layout_tensor.repeat(batch_size, 1)
            meta_embeds = self.meta_projector(layout_tensor)

        # --- 5. Encode Style ---
        if style_embeds is not None:
            style_part = style_embeds
        else:
            # Pad with Zeros for Uncond
            style_part = torch.zeros(batch_size, self.style_token_len, 768).to(self.target_device)

        # --- 6. Concatenate ---
        # Final Length: 8 + 77 + 1 = 86
        final_output = torch.cat([style_part, text_embeds, meta_embeds], dim=1)
        
        return [final_output]
def get_ref_image_path_smart(style_name):
    STYLE_REPRESENTATIVES_PATH = "style_representatives.json"

    # Load the map once
    if os.path.exists(STYLE_REPRESENTATIVES_PATH):
        with open(STYLE_REPRESENTATIVES_PATH, 'r', encoding='utf-8') as f:
            style_lookup = json.load(f)
    else:
        print(f"Warning: {STYLE_REPRESENTATIVES_PATH} not found. Style lookup will fail.")
        style_lookup = {}
    
    if style_name in style_lookup:
        return style_lookup[style_name]["image_path"]
    

    for existing_style in style_lookup.keys():
        if style_name.lower() in existing_style.lower() or existing_style.lower() in style_name.lower():
            print(f"Style '{style_name}' not found, but matched with '{existing_style}'.")
            return style_lookup[existing_style]["image_path"]

    available_styles = list(style_lookup.keys())
    random_style = random.choice(available_styles)
    print(f"Warning: Style '{style_name}' not found. Using random style '{random_style}'.")
    return style_lookup[random_style]["image_path"]

def run_adstyler_inference(ad_copy="A smartphone is on sale now!", layout=[0.1, 0.1, 0.8, 0.2], style="Impressionism", output_image_path="output_adstyler.png"):
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    # Inputs
    AD_COPY = ad_copy
    LAYOUT = layout 
    STYLE_IMAGE_PATH = get_ref_image_path_smart(style)
    OUTPUT_IMAGE = output_image_path

    # Base Models
    BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    
    # 1. Style Components Paths
    STYLE_ENCODER_PATH = "style_encoder.pth"
    STYLE_TOKENIZER_PATH = "style_tokenizer.pth"
    
    # 2. Ad Components Paths (CLIP Version)
    # *** Update this to your CLIP training output folder ***
    AD_CHECKPOINT_DIR = "ad_lora_clip_output/checkpoint-epoch-10" 
    UNET_LORA_PATH = os.path.join(AD_CHECKPOINT_DIR, "unet_lora")
    META_PROJECTOR_PATH = os.path.join(AD_CHECKPOINT_DIR, "meta_projector.pth")


    # --- Step 1: Initialize Basic SD Components ---
    print("Loading Basic Components...")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
    # Load standard CLIP Text Model
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder").to(DEVICE)
    
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet")
    scheduler = PNDMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")
    
    # Load Ad-LoRA (U-Net)
    print(f"Loading Ad-LoRA from {UNET_LORA_PATH}...")
    unet = PeftModel.from_pretrained(unet, UNET_LORA_PATH)
    unet = unet.merge_and_unload()

    # --- Step 2: Load Metadata Projector ---
    print("Loading Metadata Projector...")
    meta_projector = MetadataProjector(input_dim=4, hidden_dim=768).to(DEVICE)
    if os.path.exists(META_PROJECTOR_PATH):
        meta_projector.load_state_dict(torch.load(META_PROJECTOR_PATH, map_location=DEVICE))
        meta_projector.eval()
    else:
        print(f"Error: Meta projector not found at {META_PROJECTOR_PATH}")
        return

    # --- Step 3: Initialize Pipeline ---
    print("Initializing StableDiffusionPipelineWithStyle...")
    
    pipe = StableDiffusionPipelineWithStyle(
        vae=vae,
        text_encoder=text_encoder, # Pass real CLIP first
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    pipe = pipe.to(DEVICE)

    # *** KEY STEP: Inject Custom Wrapper ***
    # Replace the pipeline's text encoder with our Wrapper (CLIP + Meta Projector)
    ad_wrapper = AdStylerCLIPWrapper(text_encoder, meta_projector, DEVICE)
    pipe.text_encoder = ad_wrapper

    # --- Step 4: Load Style Components ---
    print("Loading Style Components...")
    
    # Style Encoder (with_head=True for 1024 dim)
    style_encoder, style_preprocess = load_style_encoder(
        device=DEVICE, 
        style_ckpt=STYLE_ENCODER_PATH,
        with_head=True 
    )
    pipe.style_encoder = style_encoder
    pipe.style_preprocess = style_preprocess

    # Style Tokenizer (Input: 1024, Output: 768)
    style_tokenizer = StyleTokenizer(
        input_size=1024, 
        out_size=768, 
        n_tokens=8, 
        with_placeholder=False, 
        prefix_model="mlp"
    )
    if os.path.exists(STYLE_TOKENIZER_PATH):
        style_tokenizer.load_state_dict(torch.load(STYLE_TOKENIZER_PATH, map_location=DEVICE))
    else:
        print("Warning: Style Tokenizer weights not found, using random init.")
        
    style_tokenizer = style_tokenizer.to(DEVICE)
    pipe.style_tokenizer = style_tokenizer
    
    # Set fc_linear to Identity (CLIP 768 -> 768)
    pipe.fc_linear = torch.nn.Linear(768, 768).to(DEVICE)
    with torch.no_grad():
        pipe.fc_linear.weight.copy_(torch.eye(768))
        pipe.fc_linear.bias.zero_()

    # --- Step 5: Run Inference ---
    print(f"Generating Ad Image...")
    if not os.path.exists(STYLE_IMAGE_PATH):
        print("Error: Style image not found.")
        return
    style_image = Image.open(STYLE_IMAGE_PATH).convert('RGB')

    # Inject Context for Wrapper
    pipe.text_encoder.current_context = {
        'ad_copy': AD_COPY, # Unused in logic but good for debug
        'layout': LAYOUT
    }

    output = pipe(
        prompt=AD_COPY,           
        style_image=style_image,
        guidance_scale=7.5,
        style_guidance_scale=5, 
        num_inference_steps=30
    )
    
    pipe.text_encoder.current_context = None

    image = output.images[0]
    image.save(OUTPUT_IMAGE)
    print(f"Result saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    Style_list = ["Traditional culture 1", "Impressionism", "hand drawn style", "Game scene picture 2", "graphic portrait style", "Op style", "Traditional Chinese ink painting style 2", "National characteristic art 1", "Architectural sketch 1", "Pulp noir style"]
    AD_COPY = "An headphone is 50 percent off today!"
    METADATA = [0.1, 0.1, 0.8, 0.2] # x, y, width, height
    STYLE = "Architectural sketch 1" 

    OUTPUT_IMAGE = "output_adstyler.png"

    run_adstyler_inference(
        ad_copy=AD_COPY, 
        layout=METADATA, 
        style=STYLE, 
        output_image_path=OUTPUT_IMAGE
    )