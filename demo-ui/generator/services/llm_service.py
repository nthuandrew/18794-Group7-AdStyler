"""
LLM Service Module - Using GPT4All for prompt engineering
"""
from gpt4all import GPT4All
import json
import re
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional

# Add layout-llm-finetuning to path for imports
# __file__ is: demo-ui/generator/services/llm_service.py
# We need to go up 4 levels to reach project root: 18794-Group7-AdStyler
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LAYOUT_LLM_DIR = BASE_DIR / 'layout-llm-finetuning'
FINETUNE_DIR = LAYOUT_LLM_DIR / 'finetune_layout_llm'
TRAIN_DIST_DIR = LAYOUT_LLM_DIR / 'train_layout_distribution'

# Setup paths for layout-llm-finetuning imports

if LAYOUT_LLM_DIR.exists():
    # Add directories to path in correct order
    if FINETUNE_DIR.exists():
        sys.path.insert(0, str(FINETUNE_DIR))
    if TRAIN_DIST_DIR.exists():
        sys.path.insert(0, str(TRAIN_DIST_DIR))
    sys.path.insert(0, str(LAYOUT_LLM_DIR))
    
    # Paths added to sys.path for imports
else:
    print(f"Warning: layout-llm-finetuning directory not found at: {LAYOUT_LLM_DIR}")

try:
    from django.conf import settings as django_settings
    HAS_DJANGO = True
except ImportError:
    # If not in Django context, use os.getenv
    django_settings = None
    HAS_DJANGO = False


class LLMService:
    """Process user input using GPT4All to generate ad information JSON"""
    
    # Allowed styles list (defined as class constant to avoid duplication)
    STYLE_LIST = [
        "Traditional culture 1",
        "Impressionism",
        "hand drawn style",
        "Game scene picture 2",
        "graphic portrait style",
        "Op style",
        "Traditional Chinese ink painting style 2",
        "National characteristic art 1",
        "Architectural sketch 1",
        "Pulp noir style"
    ]
    
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
        
        self.layout_method = getattr(django_settings, 'LAYOUT_GENERATION_METHOD', 'sample')
        print(f"Layout generation method configured: {self.layout_method}")

        self.layout_llm_model = None
        self.layout_llm_tokenizer = None
        self.layout_model = None
        self.layout_thresholds = None
        self.train_layouts = None
        
        if self.layout_method == 'llm':
            print("Initializing layout LLM...")
            self._initialize_layout_llm()
            # Check if initialization was successful
            if self.layout_llm_model is None:
                print("⚠ Layout LLM initialization failed, but continuing with sample method")
        elif self.layout_method == 'sample':
            print("Initializing layout sampler...")
            self._initialize_layout_sampler()
        else:
            print(f"Warning: Unknown layout generation method: {self.layout_method}, using 'sample'")
            self.layout_method = 'sample'
            self._initialize_layout_sampler()
        
        print(f"Final layout generation method: {self.layout_method}")
    
    def _initialize_layout_llm(self):
        """Initialize finetuned layout LLM for text_layout generation"""
        try:
            if HAS_DJANGO:
                checkpoint_path = getattr(django_settings, 'LAYOUT_LLM_CHECKPOINT_PATH', None)
                base_model = getattr(django_settings, 'LAYOUT_LLM_BASE_MODEL', 'Qwen/Qwen2.5-1.5B')
                prob_model_path = getattr(django_settings, 'LAYOUT_PROB_MODEL_PATH', None)
                thresholds_path = getattr(django_settings, 'LAYOUT_THRESHOLDS_PATH', None)
            else:
                checkpoint_path = os.getenv('LAYOUT_LLM_CHECKPOINT_PATH', None)
                base_model = os.getenv('LAYOUT_LLM_BASE_MODEL', 'Qwen/Qwen2.5-1.5B')
                prob_model_path = os.getenv('LAYOUT_PROB_MODEL_PATH', None)
                thresholds_path = os.getenv('LAYOUT_THRESHOLDS_PATH', None)
            
            print(f"Checkpoint path: {checkpoint_path}")
            print(f"Base model: {base_model}")
            print(f"Prob model path: {prob_model_path}")
            print(f"Thresholds path: {thresholds_path}")
            
            if not checkpoint_path:
                print("Error: LAYOUT_LLM_CHECKPOINT_PATH is not set")
                print("Please set LAYOUT_LLM_CHECKPOINT_PATH in settings.py or environment variable")
                raise ValueError("LAYOUT_LLM_CHECKPOINT_PATH not configured")
            
            if not os.path.exists(checkpoint_path):
                print(f"Error: Layout LLM checkpoint not found at: {checkpoint_path}")
                print("Please check the path in settings.py:")
                print(f"  LAYOUT_LLM_CHECKPOINT_PATH = '{checkpoint_path}'")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            print("Loading finetuned layout LLM...")
            try:
                # Use importlib to load modules from file paths
                import importlib.util
                
                # Import infer_layout_llm
                infer_module_path = FINETUNE_DIR / 'infer_layout_llm.py'
                if not infer_module_path.exists():
                    raise FileNotFoundError(f"infer_layout_llm.py not found at {infer_module_path}")
                
                print(f"Loading infer_layout_llm from {infer_module_path}")
                spec = importlib.util.spec_from_file_location("infer_layout_llm", infer_module_path)
                infer_module = importlib.util.module_from_spec(spec)
                # Add parent directory to sys.path for infer_layout_llm's imports
                original_path = sys.path.copy()
                sys.path.insert(0, str(FINETUNE_DIR.parent))
                try:
                    spec.loader.exec_module(infer_module)
                finally:
                    sys.path[:] = original_path
                
                load_model_for_inference = infer_module.load_model_for_inference
                print("✓ Loaded infer_layout_llm module")
                
                # Import layout_inference
                layout_inf_path = TRAIN_DIST_DIR / 'layout_inference.py'
                if not layout_inf_path.exists():
                    raise FileNotFoundError(f"layout_inference.py not found at {layout_inf_path}")
                
                print(f"Loading layout_inference from {layout_inf_path}")
                spec = importlib.util.spec_from_file_location("layout_inference", layout_inf_path)
                layout_inf_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(layout_inf_module)
                load_model_and_thresholds = layout_inf_module.load_model_and_thresholds
                print("✓ Loaded layout_inference module")
                
            except ModuleNotFoundError as import_error:
                missing_module = str(import_error).split("'")[1] if "'" in str(import_error) else "unknown"
                print(f"✗ Missing required dependency: {missing_module}")
                print(f"\nTo use layout LLM generation, please install the required dependencies:")
                print(f"  pip install torch transformers peft accelerate scikit-learn joblib numpy")
                print(f"\nOr install from requirements.txt:")
                print(f"  pip install -r requirements.txt")
                print(f"\nError details: {import_error}")
                raise ImportError(
                    f"Missing dependency '{missing_module}'. "
                    f"Please install layout LLM dependencies: "
                    f"pip install torch transformers peft accelerate scikit-learn joblib numpy"
                ) from import_error
            except Exception as import_error:
                print(f"Error importing layout LLM modules: {import_error}")
                import traceback
                traceback.print_exc()
                print(f"\nMake sure layout-llm-finetuning directory is accessible")
                print(f"Expected paths:")
                print(f"  - {FINETUNE_DIR / 'infer_layout_llm.py'} (exists: {infer_module_path.exists() if 'infer_module_path' in locals() else 'unknown'})")
                print(f"  - {TRAIN_DIST_DIR / 'layout_inference.py'} (exists: {layout_inf_path.exists() if 'layout_inf_path' in locals() else 'unknown'})")
                raise
            
            print(f"Loading model from checkpoint: {checkpoint_path}")
            self.layout_llm_model, self.layout_llm_tokenizer = load_model_for_inference(
                checkpoint_path, base_model
            )
            
            if prob_model_path and thresholds_path and os.path.exists(prob_model_path) and os.path.exists(thresholds_path):
                print("Loading layout probability model and thresholds...")
                ctx = load_model_and_thresholds(prob_model_path, thresholds_path)
                self.layout_model = ctx["model"]
                self.layout_thresholds = ctx["thresholds"]
                print("✓ Layout probability model loaded")
            else:
                print("Warning: Layout probability model or thresholds not found, will skip distribution checking")
                if prob_model_path:
                    print(f"  Prob model path: {prob_model_path} (exists: {os.path.exists(prob_model_path) if prob_model_path else False})")
                if thresholds_path:
                    print(f"  Thresholds path: {thresholds_path} (exists: {os.path.exists(thresholds_path) if thresholds_path else False})")
            
            print("✓ Layout LLM loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load layout LLM: {e}")
            import traceback
            traceback.print_exc()
            print("\nFalling back to layout sampling")
            self.layout_method = 'sample'
            self._initialize_layout_sampler()
    
    def _initialize_layout_sampler(self):
        """Initialize layout sampler from training data"""
        try:
            if HAS_DJANGO:
                train_layout_path = getattr(django_settings, 'TRAIN_LAYOUT_JSON_PATH', None)
            else:
                train_layout_path = os.getenv('TRAIN_LAYOUT_JSON_PATH', None)
            if not train_layout_path or not os.path.exists(train_layout_path):
                print(f"Warning: Train layout JSON not found: {train_layout_path}")
                print("Will use default layout")
                self.train_layouts = []
                return
            
            print(f"Loading training layouts from {train_layout_path}...")
            with open(train_layout_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract all text_layouts
            self.train_layouts = []
            for item in data:
                text_layout = item.get("text_layout")
                if text_layout and isinstance(text_layout, dict):
                    # Validate layout (color is optional, will be added if missing)
                    if all(k in text_layout for k in ['x', 'y', 'width', 'height', 'alignment']):
                        # Ensure color exists
                        if "color" not in text_layout:
                            text_layout["color"] = "white"
                        self.train_layouts.append(text_layout)
            
            print(f"✓ Loaded {len(self.train_layouts)} training layouts for sampling")
        except Exception as e:
            print(f"Failed to load training layouts: {e}")
            self.train_layouts = []
    
    def _sample_layout_from_training_data(self) -> Dict:
        """Sample a random text_layout from training data"""
        if self.train_layouts and len(self.train_layouts) > 0:
            layout = random.choice(self.train_layouts).copy()
            # Ensure color field exists
            if "color" not in layout:
                layout["color"] = "white"
            return layout
        else:
            # Fallback to default layout
            return {
                "x": 0.1,
                "y": 0.1,
                "width": 0.5,
                "height": 0.5,
                "alignment": "center",
                "color": "white"
            }
    
    def _generate_layout_with_llm(self, ad_copy: str) -> Optional[Dict]:
        """Generate text_layout using finetuned layout LLM"""
        try:
            from infer_layout_llm import generate_layout, extract_json_from_text, validate_and_fix_layout
            
            # Generate layout
            response = generate_layout(
                self.layout_llm_model,
                self.layout_llm_tokenizer,
                ad_copy,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9
            )
            
            # Extract JSON
            result = extract_json_from_text(response)
            if not result:
                return None
            
            text_layout = result.get("text_layout")
            if not text_layout:
                return None
            
            # Validate and fix
            text_layout = validate_and_fix_layout(text_layout)
            return text_layout
        except Exception as e:
            print(f"Layout LLM generation failed: {e}")
            return None
    
    def _validate_and_fix_style(self, style_value: str) -> str:
        """Validate style and return correct style from allowed list, or original if not found"""
        style_value = str(style_value).strip()
        for allowed_style in self.STYLE_LIST:
            if style_value.lower() == allowed_style.lower():
                return allowed_style
        # Return original style value if not found in allowed list
        print(f"Warning: Style '{style_value}' not in allowed list, using original value")
        return style_value
    
    def _generate_text_layout(self, ad_copy: str = None) -> Dict:
        """Generate text_layout using configured method"""
        if self.layout_method == 'llm':
            if self.layout_llm_model is None:
                print("Warning: Layout LLM model not loaded, falling back to sampling")
                return self._sample_layout_from_training_data()
            else:
                layout = self._generate_layout_with_llm(ad_copy or "")
                if layout:
                    return layout
                return self._sample_layout_from_training_data()
        elif self.layout_method == 'sample':
            return self._sample_layout_from_training_data()
        else:
            # Fallback to default
            return {
                "x": 0.1,
                "y": 0.1,
                "width": 0.5,
                "height": 0.5,
                "alignment": "left",
                "color": "white"
            }
    
    def _try_load_model(self, model_name, allow_download=False):
        """
        尝试加载模型，优先使用 CUDA GPU，失败则回退到 CPU
        
        Returns:
            GPT4All model instance or None if failed
        """
        # 检查可用的 GPU 设备（用于调试）
        try:
            available_gpus = GPT4All.list_gpus()
            if available_gpus:
                print(f"检测到可用 GPU 设备: {available_gpus}")
            else:
                print("未检测到可用 GPU 设备")
        except Exception as e:
            print(f"无法检测 GPU 设备: {e}")
        
        # 优先尝试使用 CUDA GPU
        print("尝试使用 CUDA GPU 加载模型...")
        try:
            model = GPT4All(
                model_name,
                allow_download=allow_download,
                device="cuda"  # 使用 CUDA GPU
            )
            print("✓ 成功使用 CUDA GPU 加载模型")
            return model
        except Exception as gpu_error:
            error_msg = str(gpu_error).lower()
            # 检查是否是 GPU 相关的错误（内存不足、CUDA 不可用等）
            if "cuda" in error_msg or "gpu" in error_msg or "memory" in error_msg or "device" in error_msg:
                print(f"⚠ CUDA GPU 加载失败: {gpu_error}")
                print("回退到 CPU 模式...")
            else:
                # 其他错误也回退到 CPU 试试
                print(f"⚠ 加载失败: {gpu_error}")
                print("尝试回退到 CPU 模式...")
        
        # 回退到 CPU
        try:
            model = GPT4All(
                model_name,
                allow_download=allow_download,
                device="cpu"  # 明确指定使用 CPU
            )
            print("✓ 成功使用 CPU 加载模型")
            return model
        except Exception as cpu_error:
            print(f"✗ CPU 加载也失败: {cpu_error}")
            return None
    
    def _initialize_model(self):
        """Initialize GPT4All model with CUDA GPU priority, fallback to CPU if unavailable"""
        try:
            print(f"Attempting to load model: {self.model_name}")
            
            # Try to load from path if it exists
            if os.path.exists(self.model_name):
                print(f"Loading model from path: {self.model_name}")
                self.model = self._try_load_model(self.model_name, allow_download=False)
                if self.model:
                    return
                else:
                    raise Exception("Failed to load model from path")
            
            # If path doesn't exist, try to load by name (will auto-download if not found)
            print(f"Model path not found, attempting to load by name: {self.model_name}")
            print("If model is not found locally, it will be downloaded automatically...")
            self.model = self._try_load_model(self.model_name, allow_download=True)
            if self.model:
                return
            else:
                raise Exception("Failed to load/download model by name")
                
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
        # If target_style is provided, use it (or original value if not in list); otherwise use first style as default
        if target_style:
            style_text = self._validate_and_fix_style(target_style)
        else:
            style_text = self.STYLE_LIST[0]  # Default to first style when no target_style provided
        
        default_json = {
            "ad_description": f"{user_text}, {style_text}, high quality, detailed",
            "style": style_text,
            "ad_copy": f"Discover {user_text} - Experience the difference",
            "text_layout": {
                "x": 0.1,
                "y": 0.1,
                "width": 0.5,
                "height": 0.5,
                "alignment": "left",
                "color": "white"
            }
        }

        # user_input = "KFC chicken in hand drawn style"
        # example_json = {
        #         "ad_description": "A vibrant and modern depiction of a delicious KFC chicken meal in an adventurous setting, \
        #             showcasing high-quality photography that captures the essence of taste and comfort.",
        #         "style": "hand drawn style",
        #         "ad_copy": "KFC BEYOND Fried Chicken, IT'S A KENTUCKY FRIED MIRACLE.",
        #         "text_layout": {
        #             "x": 0.1,
        #             "y": 0.2,
        #             "width": 0.3,
        #             "height": 0.5,
        #             "alignment": "left",
        #             "color": "yellow"
        #         }
        #     }
        
        if not self.model:
            # If model not initialized, return default JSON
            return default_json
        
        # Build prompt engineering instructions
        style_instruction = ""
        if target_style:
            style_instruction = f"\nTarget style: {target_style}"
        else:
            style_instruction = "\nPlease automatically select an appropriate ad style based on the user input."
        
        style_list_str = "\n".join([f"  - {style}" for style in self.STYLE_LIST])
        
        system_prompt = f"""You are a professional ad design assistant. Generate a JSON object directly without any thinking process or explanation.

CRITICAL: Output ONLY the JSON object. Do NOT include any reasoning, thinking, or explanation before or after the JSON.

Example:
User input: KFC chicken advertisement
Output:
{{
    "ad_description": "A delicious KFC chicken advertisement, Pulp noir style, professional photography, high quality, detailed",
    "style": "Pulp noir style",
    "ad_copy": "KFC BEYOND Fried Chicken, IT'S A KENTUCKY FRIED MIRACLE."
}}

Format requirements:
1. ad_description: Use English, detailed description of ad image, including scene, style, quality, etc. The style mentioned in ad_description must match the style field.
2. style: MUST be one of the following styles (choose the most appropriate one for the user's request):
{style_list_str}
3. ad_copy: Generate creative and compelling advertising copy that fits the ad theme. Do NOT copy the user input directly.

IMPORTANT: The "style" field MUST be exactly one of the styles listed above. Do NOT use any other style names.

Output ONLY the JSON object starting with {{ and ending with }}. No other text."""

        user_prompt = f"User input: {user_text}{style_instruction}\n\nPlease generate JSON format ad information according to the above format requirements."
        
        try:
            # Generate response using GPT4All
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Print prompt for debugging
            print("=" * 80)
            print("DeepSeek Prompt:")
            print("=" * 80)
            print(full_prompt)
            print("=" * 80)
            print("\nGenerating response with DeepSeek (this may take a while)...\n")
            
            response = self.model.generate(
                full_prompt,
                max_tokens=2500,  # Increased to allow longer reasoning chain and ensure completion
                temp=0.5,  # Lower temperature for more direct output
                top_k=40,  # Increase diversity while maintaining quality
                top_p=0.9,  # Nucleus sampling
                repeat_penalty=1.1,  # Prevent repetition
            )
            
            # Clean response text, try to extract JSON
            response = response.strip()
            
            # Print full response including thinking process
            print("=" * 80)
            print("DeepSeek Full Response (including thinking process):")
            print("=" * 80)
            print(response)
            print("=" * 80)
            print(f"\nResponse length: {len(response)} characters\n")
            
            # Remove thinking process (DeepSeek-R1 format: <think>...</think>)
            # Extract only the final output, ignoring reasoning
            final_output = response
            
            # Try to remove <think>...</think> tags and their content
            think_pattern = r'<think>.*?</think>'
            final_output = re.sub(think_pattern, '', final_output, flags=re.DOTALL | re.IGNORECASE)
            
            # Clean up extra whitespace
            final_output = final_output.strip()
            
            # Print the cleaned output (without thinking process)
            if final_output != response:
                print("=" * 80)
                print("Final Output (thinking process removed):")
                print("=" * 80)
                print(final_output)
                print("=" * 80)
                print(f"\nFinal output length: {len(final_output)} characters\n")
            
            # Try to extract JSON from final output
            # Strategy: Find the LAST complete JSON object (DeepSeek often outputs multiple)
            json_str = None
            
            # Method 1: Find all JSON objects and take the last one
            # This handles cases where DeepSeek outputs JSON in thinking process and final output
            json_objects = []
            brace_count = 0
            start_pos = -1
            
            for i, char in enumerate(final_output):
                if char == '{':
                    if brace_count == 0:
                        start_pos = i  # Start of a new JSON object
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_pos != -1:
                        # Found a complete JSON object
                        json_candidate = final_output[start_pos:i+1]
                        json_objects.append(json_candidate)
                        start_pos = -1
            
            # Use the last JSON object found (most likely the final output)
            if json_objects:
                json_str = json_objects[-1]
                print(f"✓ Found {len(json_objects)} JSON object(s), using the last one")
                if len(json_objects) > 1:
                    print(f"  First JSON (length): {len(json_objects[0])} chars")
                    print(f"  Last JSON (length): {len(json_str)} chars")
            
            # Method 2: Fallback to regex if method 1 didn't work
            if not json_str:
                print("⚠ Method 1 failed, trying regex fallback...")
                json_match = re.search(r'\{.*\}', final_output, re.DOTALL)
                
                # If not found in cleaned output, try original response
                if not json_match:
                    print("⚠ No JSON found in cleaned output, trying original response...")
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    
                    # Try to extract the last complete JSON from the match
                    # (in case it contains multiple JSONs)
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
            
            if json_str:
                # Clean up the JSON string
                json_str = json_str.strip()
                
                # Remove any trailing text after the closing brace
                # Find the last } and truncate there
                last_brace = json_str.rfind('}')
                if last_brace != -1:
                    json_str = json_str[:last_brace + 1]
                
                # Print the extracted JSON for debugging
                print("=" * 80)
                print("Extracted JSON string:")
                print("=" * 80)
                print(json_str)
                print("=" * 80)
                
                try:
                    ad_info = json.loads(json_str)
                    # Validate required fields (only ad_description, style, ad_copy)
                    required_keys = ['ad_description', 'style', 'ad_copy']
                    if all(key in ad_info for key in required_keys):
                        # Detect and fix common errors: if ad_copy is instruction text or user input, generate a default
                        ad_copy_value = str(ad_info.get('ad_copy', ''))
                        if 'Please automatically' in ad_copy_value or 'based on the user input' in ad_copy_value or ad_copy_value == user_text:
                            ad_info['ad_copy'] = default_json['ad_copy']
                        
                        # Validate and fix style
                        ad_info['style'] = self._validate_and_fix_style(ad_info.get('style', ''))
                        
                        # Generate text_layout
                        ad_info['text_layout'] = self._generate_text_layout(ad_info.get('ad_copy', user_text))

                        return ad_info
                    else:
                        # If fields missing, try to complete with defaults
                        print(f"Missing required fields, trying to complete. Existing fields: {list(ad_info.keys())}")
                        for key in required_keys:
                            if key not in ad_info:
                                if key == 'ad_copy':
                                    ad_info[key] = default_json['ad_copy']
                                elif key == 'style':
                                    # Use default style only if style field is completely missing
                                    ad_info[key] = self.STYLE_LIST[0]
                                elif key in default_json:
                                    ad_info[key] = default_json[key]
                        
                        # Validate style and generate text_layout
                        ad_info['style'] = self._validate_and_fix_style(ad_info.get('style', ''))
                        ad_info['text_layout'] = self._generate_text_layout(ad_info.get('ad_copy', user_text))
                        
                        # Validate again
                        if all(key in ad_info for key in required_keys + ['text_layout']):
                            return ad_info
                except json.JSONDecodeError as e:
                    print("=" * 80)
                    print(f"⚠ JSON parsing error: {e}")
                    print("=" * 80)
                    print(f"Attempted JSON string (first 500 chars):\n{json_str[:500]}")
                    if len(json_str) > 500:
                        print(f"... (truncated, total JSON length: {len(json_str)} characters)")
                    print("=" * 80)
            
            # If parsing fails, return default JSON with generated layout
            print("=" * 80)
            print("⚠ LLM response parsing failed, using default JSON")
            print("=" * 80)
            print(f"Final output (thinking process removed, first 1000 chars):\n{final_output[:1000]}")
            if len(final_output) > 1000:
                print(f"... (truncated, total length: {len(final_output)} characters)")
            print("=" * 80)
            
            # Still try to generate layout even if LLM failed
            default_json['text_layout'] = self._generate_text_layout(user_text)
            return default_json
            
        except Exception as e:
            print("=" * 80)
            print(f"✗ LLM processing error: {e}")
            print("=" * 80)
            import traceback
            traceback.print_exc()
            print("=" * 80)
            # Return default JSON on error, but still generate layout
            try:
                default_json['text_layout'] = self._generate_text_layout(user_text)
            except Exception as layout_error:
                print(f"Layout generation error: {layout_error}")
                # Use default layout from default_json
            return default_json

