import torch
import os
from PIL import Image
from transformers import CLIPTokenizer
from clip_tokenizer import CLIPTextModel  
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor

# Import classes and functions directly from your 'infer.py'
# Ensure 'infer.py', 'clip_tokenizer.py', etc., are in the same directory
from infer import StableDiffusionPipelineWithStyle, StyleTokenizer, load_style_encoder

def run_inference():
    # --- Configuration ---
    # 1. Base Model (Using SD v1.5 as per StyleTokenizer paper)
    BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5" 
    
    # 2. Path to your trained Style Encoder checkpoint
    STYLE_ENCODER_PATH = "style_encoder.pth" 
    
    # 3. Path to a reference style image for testing
    # Ensure this file exists
    STYLE_IMAGE_PATH = "style30k/images_top10/s0546____0907_01_query_2_img_000019_1682592134454_0365780909858325.jpeg.jpg" 
    
    # 4. Text Prompt (Paper suggests content-only description without style words)
    PROMPT = "A dog sitting on the grass"
    
    # 5. Output filename
    OUTPUT_IMAGE = "output_style_test.png"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Step 1: Load Basic Components from Pretrained SD ---
    print("Loading Stable Diffusion components...")
    
    # Load VAE, Tokenizer, Text Encoder, U-Net, Scheduler, etc.
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet")
    scheduler = PNDMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(BASE_MODEL_ID, subfolder="feature_extractor")
    
    # Safety checker is optional, set to None if not needed to save memory/time
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(BASE_MODEL_ID, subfolder="safety_checker")

    # --- Step 2: Initialize the Custom Pipeline ---
    print("Initializing Style Pipeline...")
    # We construct the pipeline manually instead of using load_pretrained() 
    # to avoid hardcoded paths inside infer.py
    pipe = StableDiffusionPipelineWithStyle(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor
    )
    pipe = pipe.to(DEVICE)

    # --- Step 3: Load Your Trained Style Encoder ---
    print(f"Loading Style Encoder from {STYLE_ENCODER_PATH}...")
    if not os.path.exists(STYLE_ENCODER_PATH):
        print(f"Error: Style encoder checkpoint not found at {STYLE_ENCODER_PATH}")
        return

    # Use the helper function from infer.py to load the style encoder
    # Note: Ensure your training script saved the state_dict with key "model"
    style_encoder, style_preprocess = load_style_encoder(
        device=DEVICE, 
        style_ckpt=STYLE_ENCODER_PATH,
        model_path=None, # Can be None if ClipMlp doesn't require it for initialization
        with_head=True 
    )
    
    # Attach the loaded encoder and preprocessor to the pipeline
    pipe.style_encoder = style_encoder
    pipe.style_preprocess = style_preprocess

    # --- Step 4: Initialize/Load Style Tokenizer (Projection Layer) ---
    print("Initializing Style Tokenizer...")
    
    # Parameters must match the definitions in infer.py
    # Style Encoder output (ClipMlp head) is 1024 dim (based on your training script)
    style_token_input_size = 1024
    style_tokenizer_out_dim = 768 # SD v1.5 text token dimension
    n_tokens = 8 # Number of style tokens to generate
    
    # Initialize the StyleTokenizer (MLP projector)
    # Ideally, you should load trained weights here (e.g., 'linear.ckpt').
    # For now, we initialize it randomly to verify the pipeline structure.
    style_tokenizer = StyleTokenizer(
        input_size=style_token_input_size, 
        out_size=style_tokenizer_out_dim, 
        n_tokens=n_tokens, 
        with_placeholder=False, 
        prefix_model="mlp"
    )
    
    # *** IMPORTANT: Uncomment the following line if you have trained StyleTokenizer weights ***
    if os.path.exists("style_tokenizer.pth"):
        print("Loading trained Style Tokenizer weights...")
        style_tokenizer.load_state_dict(torch.load("style_tokenizer.pth"))
    else:
        print("WARNING: Using random weights! Result will be noise.")

    style_tokenizer = style_tokenizer.to(DEVICE)
    pipe.style_tokenizer = style_tokenizer
    
    # Initialize the fc_linear layer required by infer.py's __call__ method.
    # In infer.py, there is a line: prompt_embeds = self.fc_linear(prompt_embeds)
    # We initialize it as a Linear layer. Ideally, this should also be loaded from a checkpoint.
    # Assuming input/output dims are 768 for SD v1.5.
    
    # Initialize the fc_linear layer
    pipe.fc_linear = torch.nn.Linear(768, 768).to(DEVICE)
    with torch.no_grad():
        pipe.fc_linear.weight.copy_(torch.eye(768)) 
        pipe.fc_linear.bias.zero_()                

    # --- Step 5: Run Inference ---
    print(f"Generating image with prompt: '{PROMPT}' and style ref: {STYLE_IMAGE_PATH}")
    
    if not os.path.exists(STYLE_IMAGE_PATH):
        print(f"Error: Style image not found at {STYLE_IMAGE_PATH}")
        return

    # Load and convert reference image
    style_image = Image.open(STYLE_IMAGE_PATH).convert('RGB')

    # Generate image
    # Note: Since StyleTokenizer is randomly initialized (unless loaded),
    # the output style will likely be noise or random, but this confirms the pipeline works.
    output = pipe(
        prompt=PROMPT,
        style_image=style_image,
        guidance_scale=7.5,
        style_guidance_scale=2.5, # Strength of style guidance
        num_inference_steps=30
    )
    
    # Save the result
    image = output.images[0]
    image.save(OUTPUT_IMAGE)
    print(f"Image saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    run_inference()