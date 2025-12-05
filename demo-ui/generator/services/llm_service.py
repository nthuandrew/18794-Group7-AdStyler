"""
LLM Service Module - Using GPT4All for prompt engineering
"""
from gpt4all import GPT4All
import json
import re
import os


class LLMService:
    """Process user input using GPT4All to generate ad information JSON"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize LLM service
        
        Args:
            model_name: Model name or path. If None, will try to get from GPT4ALL_MODEL env var,
                       or use default DeepSeek-R1-Distill-Qwen-7B
        """
        # Initialize GPT4All model
        self.model = None
        self.model_name = model_name or os.getenv('GPT4ALL_MODEL', 'DeepSeek-R1-Distill-Qwen-7B')
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize GPT4All model"""
        try:
            print(f"Attempting to load model: {self.model_name}")
            
            # Try to load from path if it exists
            if os.path.exists(self.model_name):
                print(f"Loading model from path: {self.model_name}")
                try:
                    self.model = GPT4All(self.model_name)
                    print(f"✓ Successfully loaded model: {self.model_name}")
                    return
                except Exception as e:
                    print(f"✗ Failed to load from path: {e}")
            
            # If path doesn't exist or failed, load default model
            print("Model path not found, loading default model...")
            try:
                self.model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
                print("✓ Loaded default model: orca-mini-3b-gguf2-q4_0.gguf")
            except Exception as e:
                print(f"✗ Failed to load default model: {e}")
                raise
                
        except Exception as e:
            print(f"✗ Model initialization failed: {e}")
            print("⚠ Will use fallback (no model, will return default JSON)")
            # If initialization fails, use fallback
            self.model = None
    
    def process_prompt(self, user_text: str, target_style: str = None) -> dict:
        """
        Process user input using LLM to generate ad information JSON
        
        Args:
            user_text: User input text
            target_style: Target style (optional, if None then LLM will decide automatically)
            
        Returns:
            Dictionary containing ad information
        """
        # Default JSON structure
        style_text = target_style if target_style else "professional"
        default_json = {
            "ad_description": f"{user_text}, {style_text} style, high quality, detailed",
            "style": style_text,
            "ad_copy": f"Discover {user_text} - Experience the difference",
            "text_layout": {
                "x": 0.1,
                "y": 0.1,
                "width": 0.5,
                "height": 0.5,
                "alignment": "left"
            },
            "text_style": {
                "font_size": "large",
                "color": "white",
                "font_weight": "bold"
            }
        }
        
        if not self.model:
            # If model not initialized, return default JSON
            return default_json
        
        # Build prompt engineering instructions
        style_instruction = ""
        if target_style:
            style_instruction = f"\nTarget style: {target_style}"
        else:
            style_instruction = "\nPlease automatically select an appropriate ad style based on the user input."
        
        system_prompt = """You are a professional ad design assistant. Generate a JSON object directly without any thinking process or explanation.

CRITICAL: Output ONLY the JSON object. Do NOT include any reasoning, thinking, or explanation before or after the JSON.

Example:
User input: KFC chicken advertisement
Output:
{
    "ad_description": "A delicious KFC chicken advertisement, modern style, professional photography, high quality, detailed",
    "style": "modern",
    "ad_copy": "KFC BEYOND Fried Chicken, IT'S A KENTUCKY FRIED MIRACLE.",
    "text_layout": {
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.5,
        "alignment": "left"
    },
    "text_style": {
        "font_size": "large",
        "color": "white",
        "font_weight": "bold"
    }
}

Format requirements:
1. ad_description: Use English, detailed description of ad image, including scene, style, quality, etc.
2. style: Style name, such as modern, vintage, minimalist, luxury, cartoon, realistic, etc.
3. ad_copy: Generate creative and compelling advertising copy that fits the ad theme. Do NOT copy the user input directly.
4. text_layout: 
   - x, y, width, height: Values between 0-1
   - alignment: left, center, or right
5. text_style:
   - font_size: small, medium, or large
   - color: Color name, such as white, black, red, blue, etc.
   - font_weight: normal or bold

Output ONLY the JSON object starting with { and ending with }. No other text."""

        user_prompt = f"User input: {user_text}{style_instruction}\n\nPlease generate JSON format ad information according to the above format requirements."
        
        try:
            # Generate response using GPT4All
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate(
                full_prompt,
                max_tokens=800,  # Reduced to limit thinking chain length
                temp=0.5,  # Lower temperature for more direct output
            )
            
            # Clean response text, try to extract JSON
            response = response.strip()
            
            # Print full response for debugging
            print(f"LLM full response: {response}")
            
            # Try to extract JSON from response - use more precise regex
            # First try to match complete JSON (from first { to last })
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # If JSON looks incomplete (doesn't end with }), try to fix
                if not json_str.rstrip().endswith('}'):
                    # Try to find last complete structure
                    # Calculate brace balance
                    brace_count = 0
                    last_valid_pos = -1
                    for i, char in enumerate(json_str):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                last_valid_pos = i
                    
                    if last_valid_pos > 0:
                        json_str = json_str[:last_valid_pos + 1]
                    else:
                        # If can't fix, try to manually complete
                        json_str = json_str.rstrip().rstrip(',') + '}'
                
                try:
                    ad_info = json.loads(json_str)
                    # Validate required fields
                    required_keys = ['ad_description', 'style', 'ad_copy', 'text_layout', 'text_style']
                    if all(key in ad_info for key in required_keys):
                        # Detect and fix common errors: if ad_copy is instruction text or user input, generate a default
                        ad_copy_value = str(ad_info.get('ad_copy', ''))
                        if 'Please automatically' in ad_copy_value or 'based on the user input' in ad_copy_value or ad_copy_value == user_text:
                            ad_info['ad_copy'] = default_json['ad_copy']
                        
                        # Detect and fix: if style is instruction text, use default
                        style_value = str(ad_info.get('style', ''))
                        if 'such as' in style_value or 'optional' in style_value or len(style_value) > 50:
                            ad_info['style'] = 'modern'
                        
                        # Validate text_layout structure
                        if isinstance(ad_info.get('text_layout'), dict):
                            layout = ad_info['text_layout']
                            layout_keys = ['x', 'y', 'width', 'height', 'alignment']
                            if all(key in layout for key in layout_keys):
                                # Validate value ranges
                                if all(0 <= layout.get(k, -1) <= 1 for k in ['x', 'y', 'width', 'height']):
                                    return ad_info
                        else:
                            # If text_layout missing or incomplete, use default
                            print("text_layout missing or incomplete, using default")
                            ad_info['text_layout'] = default_json['text_layout']
                            if all(key in ad_info for key in required_keys):
                                return ad_info
                    else:
                        # If fields missing, try to complete with defaults
                        print(f"Missing required fields, trying to complete. Existing fields: {list(ad_info.keys())}")
                        for key in required_keys:
                            if key not in ad_info:
                                if key == 'ad_copy':
                                    ad_info[key] = default_json['ad_copy']
                                elif key in default_json:
                                    ad_info[key] = default_json[key]
                        
                        # Validate again
                        if all(key in ad_info for key in required_keys):
                            if isinstance(ad_info.get('text_layout'), dict):
                                layout = ad_info['text_layout']
                                layout_keys = ['x', 'y', 'width', 'height', 'alignment']
                                if all(key in layout for key in layout_keys):
                                    if all(0 <= layout.get(k, -1) <= 1 for k in ['x', 'y', 'width', 'height']):
                                        return ad_info
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    print(f"Attempted JSON string (first 300 chars): {json_str[:300]}")
            
            # If parsing fails, return default JSON
            print(f"LLM response parsing failed, using default. Response content (first 500 chars): {response[:500]}")
            return default_json
            
        except Exception as e:
            print(f"LLM processing error: {e}")
            # Return default JSON on error
            return default_json

