#!/usr/bin/env python3
"""
Inference module for fine-tuned layout LLM.

This module provides functions to generate text_layout JSON from ad copy,
with automatic distribution checking and smart retry mechanism.
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path to import from train_layout_distribution
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_layout_distribution.layout_inference import (
    load_model_and_thresholds,
    is_in_distribution,
    infer_log_prob,
)


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from model output text."""
    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*"text_layout"[^{}]*\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except:
            pass
    
    # Try to find any JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except:
            pass
    
    return None


def validate_and_fix_layout(text_layout: Dict) -> Dict:
    """Validate and fix layout values to be in valid ranges."""
    fixed = {}
    
    # Clamp coordinates to [0, 1]
    fixed["x"] = max(0.0, min(1.0, float(text_layout.get("x", 0.0))))
    fixed["y"] = max(0.0, min(1.0, float(text_layout.get("y", 0.0))))
    fixed["width"] = max(0.0, min(1.0, float(text_layout.get("width", 0.5))))
    fixed["height"] = max(0.0, min(1.0, float(text_layout.get("height", 0.5))))
    
    # Round to 3 decimal places
    fixed["x"] = round(fixed["x"], 3)
    fixed["y"] = round(fixed["y"], 3)
    fixed["width"] = round(fixed["width"], 3)
    fixed["height"] = round(fixed["height"], 3)
    
    # Validate alignment
    alignment = str(text_layout.get("alignment", "center")).lower()
    if alignment not in ["left", "center", "right"]:
        alignment = "center"
    fixed["alignment"] = alignment
    
    # Validate color (keep as-is, but could map to known colors)
    fixed["color"] = str(text_layout.get("color", "white")).lower()
    
    return fixed


def load_model_for_inference(
    model_path: str,
    base_model_name: str = "Qwen/Qwen2.5-1.5B",
) -> tuple:
    """
    Load fine-tuned model for inference.
    
    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading LLM from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Check if it's a LoRA model
    if (Path(model_path) / "adapter_config.json").exists():
        print("Detected LoRA adapter, loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    model.eval()
    print("✓ Model loaded")
    return model, tokenizer


def generate_layout(
    model,
    tokenizer,
    ad_copy: str,
    instruction: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text_layout JSON from ad copy."""
    if instruction is None:
        instruction = (
            "You are an ad design assistant. Given an ad copy or ad design requirement, "
            "generate a text layout JSON for the ad image. The JSON must contain a 'text_layout' object "
            "with fields: x, y, width, height (all floats 0-1, 3 decimal places), "
            "alignment (one of: left, center, right), and color (a color name). "
            "Output only the JSON, no explanations."
        )
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"Ad copy:\n{ad_copy}"},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def infer_with_retry(
    model,
    tokenizer,
    ad_copy: str,
    layout_model,
    thresholds: Dict,
    max_retries: int = 3,
    filter_level: str = "p5",
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict:
    """
    Generate layout with smart retry mechanism if distribution check fails.
    Uses adaptive temperature and rejection sampling for better results.
    
    Returns:
        {
            "text_layout": {...},
            "log_prob": float,
            "in_distribution": bool,
            "retries": int,
            "all_attempts": List[Dict]  # All generated layouts for analysis
        }
    """
    best_layout = None
    best_log_prob = float("-inf")
    all_attempts = []
    retries = 0
    
    # Adaptive temperature schedule: start conservative, become more exploratory
    temp_schedule = [
        temperature * 0.5,  # First attempt: very conservative
        temperature * 0.7,  # Second attempt: slightly more exploratory
        temperature,        # Third attempt: original temperature
    ]
    
    # Generate multiple candidates per attempt for rejection sampling
    candidates_per_attempt = 3 if max_retries > 1 else 1
    
    for attempt in range(max_retries):
        current_temp = temp_schedule[min(attempt, len(temp_schedule) - 1)]
        
        # Generate multiple candidates
        candidates = []
        for candidate_idx in range(candidates_per_attempt):
            # Generate
            response = generate_layout(
                model, tokenizer, ad_copy,
                temperature=current_temp,
                top_p=top_p
            )
            
            # Extract JSON
            result = extract_json_from_text(response)
            if not result:
                continue
            
            # Get text_layout
            text_layout = result.get("text_layout")
            if not text_layout:
                continue
            
            # Validate and fix
            text_layout = validate_and_fix_layout(text_layout)
            
            # Check distribution
            log_prob = infer_log_prob(layout_model, text_layout)
            in_dist = is_in_distribution(layout_model, thresholds, text_layout, level=filter_level)
            
            candidate_info = {
                "text_layout": text_layout,
                "log_prob": log_prob,
                "in_distribution": in_dist,
                "temperature": current_temp,
                "attempt": attempt + 1,
                "candidate": candidate_idx + 1,
            }
            candidates.append(candidate_info)
            all_attempts.append(candidate_info)
            
            # Track best candidate
            if log_prob > best_log_prob:
                best_layout = text_layout
                best_log_prob = log_prob
            
            # If we found a valid one, return immediately
            if in_dist:
                return {
                    "text_layout": text_layout,
                    "log_prob": log_prob,
                    "in_distribution": True,
                    "retries": attempt + 1,
                    "all_attempts": all_attempts,
                }
        
        retries = attempt + 1
        
        # Log attempt results
        valid_count = sum(1 for c in candidates if c["in_distribution"])
        avg_log_prob = np.mean([c["log_prob"] for c in candidates]) if candidates else float("-inf")
        print(f"  Attempt {attempt + 1}: Generated {len(candidates)} candidates, "
              f"{valid_count} in distribution, avg log_prob={avg_log_prob:.2f}")
        
        # If we have multiple candidates, try the best one even if not in distribution
        if candidates and not any(c["in_distribution"] for c in candidates):
            # Sort by log_prob and try the best
            candidates.sort(key=lambda x: x["log_prob"], reverse=True)
            best_candidate = candidates[0]
            if best_candidate["log_prob"] > best_log_prob:
                best_layout = best_candidate["text_layout"]
                best_log_prob = best_candidate["log_prob"]
    
    # Return best attempt even if not in distribution
    return {
        "text_layout": best_layout,
        "log_prob": best_log_prob,
        "in_distribution": False,
        "retries": retries,
        "all_attempts": all_attempts,
    }


# Module functions can be imported and used directly
# Example usage:
#   from infer_layout_llm import load_model_for_inference, infer_with_retry
#   from train_layout_distribution.layout_inference import load_model_and_thresholds
#   
#   # Load models
#   model, tokenizer = load_model_for_inference("output_layout_llm", "Qwen/Qwen2.5-1.5B")
#   ctx = load_model_and_thresholds("layout_prob_model.joblib", "layout_thresholds.json")
#   
#   # Generate
#   result = infer_with_retry(
#       model, tokenizer, "Your ad copy",
#       ctx["model"], ctx["thresholds"]
#   )

