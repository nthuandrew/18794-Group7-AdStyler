"""
Diffusion Model Service Module - Image Generation Interface
"""
from PIL import Image
import io
import random
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from django.conf import settings

# Add project root to path for adstyler_src imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ADSTYLER_SRC_DIR = BASE_DIR / 'adstyler_src'
if ADSTYLER_SRC_DIR.exists():
    sys.path.insert(0, str(BASE_DIR))

try:
    from adstyler_src.test_adstyler import run_adstyler_inference
except ImportError as e:
    print(f"Warning: Could not import adstyler_src.test_adstyler: {e}")
    print(f"Make sure adstyler_src directory is accessible from {BASE_DIR}")
    run_adstyler_inference = None


class DiffusionService:
    """Diffusion model service - Currently returns randomly generated images as placeholders"""
    
    def __init__(self):
        self.width = 512
        self.height = 512
    
    def generate_image(self, prompt: str, width: int = 512, height: int = 512) -> Image.Image:
        """
        Generate image (currently placeholder implementation)
        
        Args:
            prompt: Image generation prompt
            width: Image width
            height: Image height
            
        Returns:
            PIL Image object
        """
        self.width = width
        self.height = height
        
        # TODO: Implement actual diffusion model call here
        # Currently returns a randomly generated colored image as placeholder
        
        # Create a random colored image
        image = Image.new('RGB', (width, height))
        pixels = image.load()
        
        # Generate random gradient background
        base_color = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200)
        )
        
        for x in range(width):
            for y in range(height):
                # Create gradient effect
                r = min(255, base_color[0] + random.randint(-30, 30))
                g = min(255, base_color[1] + random.randint(-30, 30))
                b = min(255, base_color[2] + random.randint(-30, 30))
                pixels[x, y] = (r, g, b)
        
        # Add some random shapes as placeholders
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # Draw some random circles
        for _ in range(random.randint(3, 8)):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            radius = random.randint(20, 100)
            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
            draw.ellipse(
                [x1 - radius, y1 - radius, x1 + radius, y1 + radius],
                fill=color,
                outline=None
            )
        
        return image
    
    def save_image(self, image: Image.Image, filename: str) -> str:
        """
        Save image to media directory
        
        Args:
            image: PIL Image object
            filename: Filename
            
        Returns:
            Saved file path (relative to MEDIA_URL)
        """
        # Ensure media directory exists
        media_dir = settings.MEDIA_ROOT
        os.makedirs(media_dir, exist_ok=True)
        
        # Save image
        file_path = os.path.join(media_dir, filename)
        image.save(file_path, 'PNG')
        
        # Return path relative to MEDIA_URL
        return os.path.join(settings.MEDIA_URL, filename)
    
    def generate_image_from_prompt(
        self,
        style: str,
        ad_copy: Optional[str] = None,
        text_layout: Optional[Dict] = None
    ) -> str:
        """
        Generate image using AdStyler model
        
        Args:
            style: Style name for the ad
            ad_copy: Advertising copy text
            text_layout: Text layout specification (x, y, width, height, alignment, color)
            
        Returns:
            Image URL path
        """
        if run_adstyler_inference is None:
            raise ImportError("AdStyler inference function is not available. Please check adstyler_src imports.")
        
        if not text_layout:
            raise ValueError("text_layout is required for AdStyler image generation")
        
        # Extract layout metadata
        metadata = [
            text_layout.get("x", 0.1),
            text_layout.get("y", 0.1),
            text_layout.get("width", 0.5),
            text_layout.get("height", 0.5)
        ]

        media_dir = settings.MEDIA_ROOT
        os.makedirs(media_dir, exist_ok=True)

        # Generate unique filename
        import hashlib
        import time
        filename_hash = hashlib.md5(f"{style}{ad_copy}{time.time()}".encode()).hexdigest()[:8]
        filename = f"generated_adstyler_{filename_hash}.png"
        output_path = os.path.join(media_dir, filename)

        try:
            print(f"Generating image with AdStyler: style={style}, ad_copy={ad_copy[:50]}...")
            run_adstyler_inference(
                ad_copy=ad_copy or "",
                layout=metadata,
                style=style,
                output_image_path=output_path,
            )
            print(f"✓ Image generated successfully: {output_path}")
        except Exception as e:
            print(f"✗ Error generating image with AdStyler: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Return URL for frontend
        image_url = os.path.join(settings.MEDIA_URL, filename)
        return image_url
