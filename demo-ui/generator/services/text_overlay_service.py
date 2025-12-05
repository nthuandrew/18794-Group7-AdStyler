"""
Text Overlay Service Module - Draw text on images according to layout specifications
"""
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Dict, Optional, Tuple


class TextOverlayService:
    """Service for overlaying text on images based on layout specifications"""
    
    def __init__(self):
        self._font_cache = {}
    
    def _get_font(self, size: int = 40):
        """Get font object, with fallback to default font"""
        cache_key = size
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]
        
        try:
            # Try to load a nice font (works on macOS, Linux, Windows)
            if os.name == 'nt':  # Windows
                font_path = "C:/Windows/Fonts/arial.ttf"
            elif os.name == 'posix':  # macOS/Linux
                # Try common font paths
                font_paths = [
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                ]
                font_path = None
                for path in font_paths:
                    if os.path.exists(path):
                        font_path = path
                        break
            else:
                font_path = None
            
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, size)
            else:
                # Fallback to default font
                font = ImageFont.load_default()
            
            self._font_cache[cache_key] = font
            return font
        except Exception as e:
            print(f"Warning: Could not load font, using default: {e}")
            font = ImageFont.load_default()
            self._font_cache[cache_key] = font
            return font
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse color string to RGB tuple"""
        color_map = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
        }
        color_str_lower = color_str.lower().strip()
        if color_str_lower in color_map:
            return color_map[color_str_lower]
        # Try to parse as hex color
        if color_str.startswith('#'):
            try:
                hex_color = color_str[1:]
                if len(hex_color) == 6:
                    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            except:
                pass
        # Default to white if parsing fails
        return (255, 255, 255)
    
    def _calculate_text_dimensions(
        self,
        text: str,
        max_width: int,
        font_size: int = 40
    ) -> Tuple[list, ImageFont.FreeTypeFont, int, int]:
        """
        Calculate text dimensions by rendering on a temporary canvas
        
        Args:
            text: Text to measure
            max_width: Maximum width constraint
            font_size: Font size to use
            
        Returns:
            Tuple of (lines, font, text_width, text_height)
        """
        # Split text by newlines first (respect user's line breaks)
        paragraphs = text.split('\n')
        all_lines = []
        
        font = self._get_font(font_size)
        
        # Create a temporary image to measure text
        temp_img = Image.new('RGB', (max_width * 2, max_width * 2))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Process each paragraph
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                all_lines.append("")
                continue
            
            # Split paragraph into words
            words = paragraph.split()
            current_line = []
            current_width = 0
            
            # Calculate space width
            space_bbox = temp_draw.textbbox((0, 0), " ", font=font)
            space_width = space_bbox[2] - space_bbox[0]
            
            for word in words:
                word_bbox = temp_draw.textbbox((0, 0), word, font=font)
                word_width = word_bbox[2] - word_bbox[0]
                
                if current_width + word_width <= max_width or not current_line:
                    current_line.append(word)
                    current_width += word_width + space_width
                else:
                    all_lines.append(" ".join(current_line))
                    current_line = [word]
                    current_width = word_width
            
            if current_line:
                all_lines.append(" ".join(current_line))
        
        # Calculate total dimensions
        line_height = font_size + 4
        text_height = len(all_lines) * line_height
        text_width = 0
        
        for line in all_lines:
            if line:
                bbox = temp_draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                text_width = max(text_width, line_width)
        
        return all_lines, font, text_width, text_height
    
    def _find_optimal_font_size(
        self,
        text: str,
        target_width: int,
        target_height: int,
        min_font_size: int = 8,
        max_font_size: int = 200
    ) -> Tuple[list, ImageFont.FreeTypeFont, int, int]:
        """
        Find optimal font size that fits text within target dimensions
        
        Args:
            text: Text to render
            target_width: Target width in pixels
            target_height: Target height in pixels
            min_font_size: Minimum font size
            max_font_size: Maximum font size
            
        Returns:
            Tuple of (lines, font, text_width, text_height)
        """
        # Binary search for optimal font size
        low, high = min_font_size, max_font_size
        best_result = None
        best_font_size = min_font_size
        
        while low <= high:
            mid_font_size = (low + high) // 2
            lines, font, text_width, text_height = self._calculate_text_dimensions(
                text, target_width, mid_font_size
            )
            
            # Check if text fits
            if text_width <= target_width and text_height <= target_height:
                best_result = (lines, font, text_width, text_height)
                best_font_size = mid_font_size
                low = mid_font_size + 1  # Try larger font
            else:
                high = mid_font_size - 1  # Try smaller font
        
        # If no fit found, use the best attempt
        if best_result is None:
            # Fallback: use minimum font size
            lines, font, text_width, text_height = self._calculate_text_dimensions(
                text, target_width, min_font_size
            )
            return lines, font, text_width, text_height
        
        return best_result
    
    def draw_text_on_image(
        self,
        image: Image.Image,
        text: str,
        text_layout: Dict,
        add_outline: bool = True,
        outline_color: Tuple[int, int, int] = (0, 0, 0),
        outline_width: int = 2
    ) -> Image.Image:
        """
        Draw text on image according to text_layout specifications
        
        Args:
            image: PIL Image object (base image)
            text: Text to draw (can contain \n for line breaks)
            text_layout: Dictionary with keys:
                - x: X position of top-left corner (0.0 to 1.0, relative to image width)
                - y: Y position of top-left corner (0.0 to 1.0, relative to image height)
                - width: Text box width (0.0 to 1.0, relative to image width)
                - height: Text box height (0.0 to 1.0, relative to image height)
                - alignment: Text alignment ('left', 'center', 'right')
                - color: Text color (color name or hex code)
            add_outline: Whether to add outline to text for better visibility
            outline_color: Color of the outline
            outline_width: Width of the outline
            
        Returns:
            PIL Image with text drawn on it
        """
        # Create a copy to avoid modifying original
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Get image dimensions
        img_width, img_height = result_image.size
        
        # Calculate text box position and size in pixels
        x = int(text_layout.get('x', 0.1) * img_width)
        y = int(text_layout.get('y', 0.1) * img_height)
        box_width = int(text_layout.get('width', 0.5) * img_width)
        box_height = int(text_layout.get('height', 0.5) * img_height)
        
        # Ensure minimum dimensions
        box_width = max(box_width, 50)
        box_height = max(box_height, 20)
        
        # Get text color
        text_color = self._parse_color(text_layout.get('color', 'white'))
        
        # Get alignment
        alignment = text_layout.get('alignment', 'left').lower()
        
        # Find optimal font size and calculate text layout
        lines, font, text_width, text_height = self._find_optimal_font_size(
            text, box_width, box_height
        )
        
        # Calculate line height from actual font
        line_height = font.size + 4
        
        # Draw each line
        for i, line in enumerate(lines):
            if i * line_height >= box_height:
                break  # Stop if we exceed box height
            
            line_y = y + i * line_height
            
            # Calculate x position based on alignment
            if line:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
            else:
                line_width = 0
            
            if alignment == 'center':
                line_x = x + (box_width - line_width) // 2
            elif alignment == 'right':
                line_x = x + box_width - line_width
            else:  # left
                line_x = x
            
            # Draw outline first (if enabled) for better visibility
            if line and add_outline:
                for adj in range(-outline_width, outline_width + 1):
                    for adj2 in range(-outline_width, outline_width + 1):
                        if adj != 0 or adj2 != 0:
                            draw.text(
                                (line_x + adj, line_y + adj2),
                                line,
                                font=font,
                                fill=outline_color
                            )
            
            # Draw text
            if line:
                draw.text((line_x, line_y), line, font=font, fill=text_color)
        
        return result_image
    
    def add_text_overlay(
        self,
        image: Image.Image,
        ad_copy: str,
        text_layout: Dict
    ) -> Image.Image:
        """
        Convenience method to add text overlay to image
        
        Args:
            image: PIL Image object
            ad_copy: Text to overlay
            text_layout: Layout specification dictionary
            
        Returns:
            PIL Image with text overlay
        """
        return self.draw_text_on_image(image, ad_copy, text_layout)
