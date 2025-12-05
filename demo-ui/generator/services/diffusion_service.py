"""
Diffusion Model Service Module - Image Generation Interface
"""
from PIL import Image
import io
import random
import os
from typing import Dict, Optional
from django.conf import settings
from adstyler_src.test_adstyler import run_adstyler_inference


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
        prompt: str,
        ad_copy: Optional[str] = None,
        text_layout: Optional[Dict] = None
    ) -> str:
        """
        Generate image from prompt, optionally add text overlay
        
        Args:
            prompt: Image generation prompt
            ad_copy: Advertising copy text to overlay on image
            text_layout: Text layout specification (x, y, width, height, alignment, color)
            
        Returns:
            Tuple of (image object, saved path)
        """
        # Generate base image
        # image = self.generate_image(prompt)
        
        # # Add text overlay if provided
        # if ad_copy and text_layout:
        #     from .text_overlay_service import TextOverlayService
        #     text_service = TextOverlayService()
        #     try:
        #         image = text_service.add_text_overlay(image, ad_copy, text_layout)
        #         print(f"✓ Text overlay added: '{ad_copy[:50]}...' at position ({text_layout.get('x', 0):.2f}, {text_layout.get('y', 0):.2f})")
        #     except Exception as e:
        #         print(f"⚠ Warning: Failed to add text overlay: {e}")
        #         # Continue with image without text overlay
        
        # # Generate filename
        # import hashlib
        # import time
        # filename_hash = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8]
        # filename = f"generated_{filename_hash}.png"
        
        # # Save image
        # saved_path = self.save_image(image, filename)
        
        # return image, saved_path
        AD_COPY = "An headphone is 50 percent off today!"
        METADATA = [0.1, 0.1, 0.8, 0.2] # x, y, width, height
        STYLE = "Architectural sketch 1" 

        # filename = "output_adstyler.png"
        # media_dir = settings.MEDIA_ROOT
        # os.makedirs(media_dir, exist_ok=True)
        
        # # Save image
        # file_path = os.path.join(media_dir, filename)

        # run_adstyler_inference(
        #     ad_copy=AD_COPY, 
        #     layout=METADATA, 
        #     style=STYLE, 
        #     output_image_path=file_path
        # )
        # return os.path.join(settings.MEDIA_URL, filename)
        media_dir = settings.MEDIA_ROOT
        os.makedirs(media_dir, exist_ok=True)

        filename = "output_adstyler.png"
        output_path = os.path.join(media_dir, filename)

        run_adstyler_inference(
            ad_copy=AD_COPY,
            layout=METADATA,
            style=STYLE,
            output_image_path=output_path,
        )

        # 回傳給前端用的 URL
        image_url = os.path.join(settings.MEDIA_URL, filename)
        return image_url

        

