import torch
import torch.nn.functional as F
from PIL import Image
import os
import json
from tqdm import tqdm
from collections import defaultdict

# Reuse code from infer.py to ensure model consistency
from infer import load_style_encoder

def get_embeddings_batch(image_paths, encoder, preprocess, device, batch_size=32):
    """
    Calculate embeddings for a list of image paths.
    """
    embeddings = []
    # Process in batches to avoid OOM
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:min(i + batch_size, len(image_paths))]
        batch_tensors = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Add a dummy tensor to keep indices aligned, or handle skip
                # Here we just append zero tensor (will have 0 similarity)
                batch_tensors.append(torch.zeros(3, 224, 224))

        if not batch_tensors:
            continue

        # Stack and move to GPU
        batch_input = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            # encoder output shape: [batch, 1024] (if with_head=True)
            feats = encoder(batch_input)
            embeddings.append(feats.cpu())

    if embeddings:
        return torch.cat(embeddings, dim=0)
    else:
        return None

def main():
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Paths
    STYLE_ENCODER_PATH = "style_encoder.pth"
    IMAGES_DIR = "style30k/images_top10"  # Directory containing your 10 style images
    STYLE_MAPPER_PATH = "style30k/style_mapper.json" # Path to the name-to-id map
    OUTPUT_JSON = "style_representatives.json" # Output file

    # 1. Load Style Mapper
    print(f"Loading style mapper from {STYLE_MAPPER_PATH}...")
    with open(STYLE_MAPPER_PATH, 'r', encoding='utf-8') as f:
        mapper_data = json.load(f)
    
    # Create reverse mapping: ID -> Name (for easier lookup later)
    # style_name2id = mapper_data.get("style_name2id", {})
    style_id2name = mapper_data.get("style_id2name", {})
    
    # 2. Group Images by Style ID
    print(f"Scanning images in {IMAGES_DIR}...")
    style_groups = defaultdict(list)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for filename in os.listdir(IMAGES_DIR):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            if "____" in filename:
                # Filename format: s0283____filename.jpg
                style_id = filename.split("____")[0]
                full_path = os.path.join(IMAGES_DIR, filename)
                style_groups[style_id].append(full_path)
    
    print(f"Found {len(style_groups)} unique styles in the directory.")

    # 3. Load Style Encoder
    print("Loading Style Encoder...")
    # Important: Use with_head=True to use the full embedding space (1024 dim) for calculation
    style_encoder, style_preprocess = load_style_encoder(
        device=DEVICE, 
        style_ckpt=STYLE_ENCODER_PATH,
        with_head=True 
    )
    style_encoder.eval()

    # 4. Find Representative for Each Style
    representatives = {} # format: { style_name: { id: "sXXX", path: "..." } }
    
    print("Calculating centroids and finding representatives...")
    for style_id, paths in tqdm(style_groups.items()):
        
        # A. Get Embeddings for all images in this style
        embeddings = get_embeddings_batch(paths, style_encoder, style_preprocess, DEVICE)
        
        if embeddings is None or embeddings.shape[0] == 0:
            continue

        # B. Calculate Centroid (Mean Vector)
        # Shape: [1, 1024]
        centroid = torch.mean(embeddings, dim=0, keepdim=True)
        
        # Normalize centroid for cosine similarity (embeddings are already normalized by encoder usually, but let's be safe)
        centroid = F.normalize(centroid, dim=1)
        
        # C. Calculate Similarity to Centroid
        # embeddings: [N, 1024], centroid: [1, 1024]
        # Cosine similarity is effectively dot product for normalized vectors
        sims = torch.mm(embeddings, centroid.T).squeeze(1) # [N]
        
        # D. Find Index of Maximum Similarity
        best_idx = torch.argmax(sims).item()
        best_path = paths[best_idx]
        
        # E. Store Result
        # Get the human-readable name from the mapper
        style_name = style_id2name.get(style_id, f"Unknown_Style_{style_id}")
        
        representatives[style_name] = {
            "style_id": style_id,
            "image_path": best_path,
            "similarity_to_centroid": float(sims[best_idx])
        }

    # 5. Save to JSON
    print(f"Saving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(representatives, f, indent=4, ensure_ascii=False)
    
    print("Done! Example content:")
    # Print first item as example
    if representatives:
        first_key = list(representatives.keys())[0]
        print(f"Style: {first_key}, Data: {representatives[first_key]}")

if __name__ == "__main__":
    main()