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
    """
    Extract the first complete JSON from model output text.
    Stops at the first valid JSON to avoid processing repeated content.
    """
    # 移除可能的特殊字符和多余空白
    text = text.strip()
    
    # 方法1: 查找第一个完整的包含text_layout的JSON对象
    # 使用更精确的正则表达式，匹配嵌套的JSON结构
    pattern = r'\{\s*"text_layout"\s*:\s*\{[^}]*"x"[^}]*"y"[^}]*"width"[^}]*"height"[^}]*"alignment"[^}]*"color"[^}]*\}[^}]*\}'
    json_match = re.search(pattern, text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            result = json.loads(json_str)
            if "text_layout" in result:
                return result
        except json.JSONDecodeError:
            pass
    
    # 方法2: 查找第一个完整的JSON对象（通过匹配括号）
    # 找到第一个 { 的位置
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # 从第一个 { 开始，找到匹配的 }
    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    if end_idx > start_idx:
        json_str = text[start_idx:end_idx]
        try:
            result = json.loads(json_str)
            # 验证是否包含text_layout
            if isinstance(result, dict) and "text_layout" in result:
                return result
        except json.JSONDecodeError:
            pass
    
    # 方法3: 尝试提取第一个看起来像JSON的部分（更宽松的匹配）
    # 查找 "text_layout" 关键字
    text_layout_idx = text.find('"text_layout"')
    if text_layout_idx != -1:
        # 向前找到最近的 {
        start_idx = text.rfind('{', 0, text_layout_idx)
        if start_idx != -1:
            # 向后找到匹配的 }
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, min(start_idx + 500, len(text))):  # 限制搜索范围
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                try:
                    result = json.loads(json_str)
                    if isinstance(result, dict) and "text_layout" in result:
                        return result
                except json.JSONDecodeError:
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
    
    Supports:
    - Final merged model (output_layout_llm directory)
    - LoRA checkpoint (checkpoint-XXX directory with adapter_config.json)
    
    Returns:
        (model, tokenizer) tuple
    """
    model_path = Path(model_path)
    print(f"Loading LLM from {model_path}...")
    
    # Check if it's a LoRA checkpoint (has adapter_config.json)
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        print("Detected LoRA adapter checkpoint, loading base model first...")
        
        # Load tokenizer from checkpoint if available, otherwise from base model
        tokenizer_path = model_path if (model_path / "tokenizer_config.json").exists() else base_model_name
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model with appropriate dtype based on device
        print(f"Loading base model: {base_model_name}...")
        cuda_available = torch.cuda.is_available()
        
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if cuda_available:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
            print("使用 CUDA (GPU)")
        else:
            # Use CPU (skip MPS)
            model_kwargs["torch_dtype"] = torch.float32
            print("使用 CPU (跳过 MPS)")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map (i.e., using CPU)
        if not cuda_available:
            device = torch.device("cpu")
            base_model = base_model.to(device)
            print(f"模型已移动到 {device}")
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from {model_path}...")
        try:
            model = PeftModel.from_pretrained(base_model, str(model_path))
            # Try to merge for faster inference
            print("Attempting to merge LoRA adapter into base model...")
            try:
                model = model.merge_and_unload()
                print("✓ Successfully merged LoRA adapter")
            except Exception as merge_error:
                print(f"Warning: Failed to merge LoRA adapter: {merge_error}")
                print("Will use adapter without merging (slower but works)")
                # Keep the PeftModel as-is
        except Exception as e:
            print(f"Error loading LoRA adapter: {e}")
            raise
        
    else:
        # Load as full model (merged or final checkpoint)
        print("Loading as full model...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine dtype based on device
        cuda_available = torch.cuda.is_available()
        
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if cuda_available:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
            print("使用 CUDA (GPU)")
        else:
            # Use CPU (skip MPS)
            model_kwargs["torch_dtype"] = torch.float32
            print("使用 CPU (跳过 MPS)")
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **model_kwargs
        )
        
        # Move to device if not using device_map (i.e., using CPU)
        if not cuda_available:
            device = torch.device("cpu")
            model = model.to(device)
            print(f"模型已移动到 {device}")
    
    model.eval()
    print("✓ Model loaded successfully")
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
    
    # 检查ad_copy是否已经包含位置信息和"Ad copy:"前缀
    # 如果已经包含，直接使用；否则添加"Ad copy:"前缀
    if ad_copy.strip().startswith("Ad copy:"):
        user_content = ad_copy
    else:
        user_content = f"Ad copy:\n{ad_copy}"
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 确定设备
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device("cpu")
    if hasattr(model, 'device'):
        device = model.device
    
    # Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # 准备生成参数
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": min(max_new_tokens, 200),  # 限制最大长度，JSON不需要太长
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask
    
    # 添加停止序列：如果生成 } 后遇到换行或其他JSON开始，可以停止
    # 但transformers的generate不直接支持stop sequences，所以我们用max_new_tokens限制
    # 并在后处理中截取第一个JSON
    
    try:
        print(f"      开始生成 (设备={device}, 输入长度={input_ids.shape[1]}, max_new_tokens={generation_kwargs['max_new_tokens']})...")
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
        print(f"      生成完成，输出长度: {outputs.shape[1]}")
    except Exception as e:
        print(f"      生成时出错: {e}")
        print(f"      设备: {device}, 输入长度: {input_ids.shape[1]}")
        import traceback
        traceback.print_exc()
        raise
    
    # Decode response
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    response = response.strip()
    
    # 尝试提取第一个完整的JSON，如果找到就截取
    # 这样可以避免处理重复内容
    json_result = extract_json_from_text(response)
    if json_result:
        # 如果成功提取JSON，尝试找到它在原始响应中的位置并截取
        # 这样可以返回更短的响应
        json_str = json.dumps(json_result, ensure_ascii=False)
        # 在响应中找到JSON字符串的位置
        json_start = response.find('{')
        if json_start != -1:
            # 找到第一个完整JSON的结束位置
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end > json_start:
                # 截取到第一个完整JSON结束
                response = response[:json_end]
    
    return response


def calculate_percentile_info(log_prob: float, thresholds: Dict) -> Dict:
    """
    Calculate percentile information for a log probability value.
    
    Returns:
        {
            "percentile_estimate": float,  # Estimated percentile (0-100)
            "below_p1": bool,
            "below_p5": bool,
            "below_p10": bool,
            "threshold_used": str,  # Which threshold was used
            "threshold_value": float,
            "distance_from_threshold": float,  # How far from threshold
            "percentile_range": str,  # Human-readable range description
        }
    """
    p1 = thresholds.get("p1", float("-inf"))
    p5 = thresholds.get("p5", float("-inf"))
    p10 = thresholds.get("p10", float("-inf"))
    train_mean = thresholds.get("train_mean", 0.0)
    train_var = thresholds.get("train_var", 1.0)
    train_std = np.sqrt(train_var) if train_var > 0 else 1.0
    
    # Determine which percentile range
    below_p1 = log_prob < p1
    below_p5 = log_prob < p5
    below_p10 = log_prob < p10
    
    # Estimate percentile using normal distribution approximation
    # Assuming log probabilities follow approximately normal distribution
    if train_std > 0:
        z_score = (log_prob - train_mean) / train_std
        # Use error function approximation (no scipy dependency)
        import math
        # CDF approximation: 0.5 * (1 + erf(z / sqrt(2)))
        percentile_estimate = 50.0 * (1 + math.erf(z_score / math.sqrt(2)))
    else:
        percentile_estimate = 50.0  # Default if no variance
    
    # Clamp to [0, 100]
    percentile_estimate = max(0.0, min(100.0, percentile_estimate))
    
    # Determine which threshold was used and distance
    if below_p1:
        threshold_used = "p1"
        threshold_value = p1
        distance = log_prob - p1
        percentile_range = "最低 1%"
    elif below_p5:
        threshold_used = "p5"
        threshold_value = p5
        distance = log_prob - p5
        percentile_range = "最低 1-5%"
    elif below_p10:
        threshold_used = "p10"
        threshold_value = p10
        distance = log_prob - p10
        percentile_range = "最低 5-10%"
    else:
        threshold_used = "p10"
        threshold_value = p10
        distance = log_prob - p10
        if percentile_estimate < 25:
            percentile_range = "最低 10-25%"
        elif percentile_estimate < 50:
            percentile_range = "25-50%"
        elif percentile_estimate < 75:
            percentile_range = "50-75%"
        else:
            percentile_range = "75-100%"
    
    return {
        "percentile_estimate": percentile_estimate,
        "below_p1": below_p1,
        "below_p5": below_p5,
        "below_p10": below_p10,
        "threshold_used": threshold_used,
        "threshold_value": threshold_value,
        "distance_from_threshold": distance,
        "percentile_range": percentile_range,
    }


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
            "all_attempts": List[Dict],  # All generated layouts for analysis
            "percentile_info": Dict,  # Percentile and rejection reason info
        }
    """
    best_layout = None
    best_log_prob = float("-inf")
    best_percentile_info = None
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
        print(f"  尝试 {attempt + 1}/{max_retries} (温度={current_temp:.3f}):")
        
        # Generate multiple candidates
        candidates = []
        for candidate_idx in range(candidates_per_attempt):
            # Generate
            response = generate_layout(
                model, tokenizer, ad_copy,
                temperature=current_temp,
                top_p=top_p,
                max_new_tokens=200  # 限制生成长度，JSON通常只需要100-150 tokens
            )
            
            # Extract JSON (函数内部已经优化，会提取第一个完整JSON)
            result = extract_json_from_text(response)
            if not result:
                print(f"      Candidate {candidate_idx + 1}: 无法从响应中提取JSON")
                print(f"        响应预览: {response[:200]}...")
                continue
            
            # Get text_layout
            text_layout = result.get("text_layout")
            if not text_layout:
                print(f"      Candidate {candidate_idx + 1}: 响应中没有text_layout字段")
                continue
            
            # Validate and fix
            text_layout = validate_and_fix_layout(text_layout)
            
            # Check distribution
            log_prob = infer_log_prob(layout_model, text_layout)
            in_dist = is_in_distribution(layout_model, thresholds, text_layout, level=filter_level)
            percentile_info = calculate_percentile_info(log_prob, thresholds)
            
            candidate_info = {
                "text_layout": text_layout,
                "log_prob": log_prob,
                "in_distribution": in_dist,
                "temperature": current_temp,
                "attempt": attempt + 1,
                "candidate": candidate_idx + 1,
                "percentile_info": percentile_info,
            }
            candidates.append(candidate_info)
            all_attempts.append(candidate_info)
            
            # Track best candidate
            if log_prob > best_log_prob:
                best_layout = text_layout
                best_log_prob = log_prob
                best_percentile_info = percentile_info
            
            # Print detailed information for this candidate
            print(f"      Candidate {candidate_idx + 1}:")
            print(f"        text_layout: {json.dumps(text_layout, indent=10, ensure_ascii=False)}")
            print(f"        log_prob: {log_prob:.4f}")
            print(f"        分布状态: {'✓ 符合分布' if in_dist else '✗ 不符合分布'}")
            
            if percentile_info:
                percentile_est = percentile_info.get("percentile_estimate", 50.0)
                percentile_range = percentile_info.get("percentile_range", "未知")
                threshold_used = percentile_info.get("threshold_used", filter_level)
                threshold_value = percentile_info.get("threshold_value", 0.0)
                distance = percentile_info.get("distance_from_threshold", 0.0)
                below_p1 = percentile_info.get("below_p1", False)
                below_p5 = percentile_info.get("below_p5", False)
                below_p10 = percentile_info.get("below_p10", False)
                
                if not in_dist:
                    # 说明为什么被拒绝
                    if below_p1:
                        reason = f"log概率 ({log_prob:.4f}) 低于 p1 阈值 ({threshold_value:.4f})，位于最低 1% 区域"
                    elif below_p5:
                        reason = f"log概率 ({log_prob:.4f}) 低于 p5 阈值 ({threshold_value:.4f})，位于最低 1-5% 区域"
                    elif below_p10:
                        reason = f"log概率 ({log_prob:.4f}) 低于 p10 阈值 ({threshold_value:.4f})，位于最低 5-10% 区域"
                    else:
                        reason = f"log概率 ({log_prob:.4f}) 低于 {filter_level} 阈值 ({threshold_value:.4f})"
                    
                    print(f"        拒绝原因: {reason}")
                    print(f"        距离阈值: {distance:.4f} (负值表示低于阈值)")
                
                print(f"        估计百分位: {percentile_est:.2f}% ({percentile_range})")
                print(f"        使用的阈值: {threshold_used} = {threshold_value:.4f}")
                train_mean = thresholds.get("train_mean", 0.0)
                train_var = thresholds.get("train_var", 1.0)
                train_std = np.sqrt(train_var) if train_var > 0 else 1.0
                print(f"        训练数据: 均值={train_mean:.4f}, 标准差={train_std:.4f}")
            
            # If we found a valid one, return immediately
            if in_dist:
                print(f"      ✓ 找到符合分布的布局，停止重试")
                return {
                    "text_layout": text_layout,
                    "log_prob": log_prob,
                    "in_distribution": True,
                    "retries": attempt + 1,
                    "all_attempts": all_attempts,
                    "percentile_info": percentile_info,
                }
        
        retries = attempt + 1
        
        # Log attempt results
        valid_count = sum(1 for c in candidates if c["in_distribution"])
        avg_log_prob = np.mean([c["log_prob"] for c in candidates]) if candidates else float("-inf")
        print(f"  Attempt {attempt + 1} 总结: 生成 {len(candidates)} 个候选，"
              f"{valid_count} 个符合分布，平均 log_prob={avg_log_prob:.2f}")
        
        # If we have multiple candidates, try the best one even if not in distribution
        if candidates and not any(c["in_distribution"] for c in candidates):
            # Sort by log_prob and try the best
            candidates.sort(key=lambda x: x["log_prob"], reverse=True)
            best_candidate = candidates[0]
            if best_candidate["log_prob"] > best_log_prob:
                best_layout = best_candidate["text_layout"]
                best_log_prob = best_candidate["log_prob"]
                best_percentile_info = best_candidate.get("percentile_info")
    
    # Return best attempt even if not in distribution
    if best_layout is None:
        print("  ✗ 所有尝试都失败，无法生成有效的布局")
        return {
            "text_layout": None,
            "log_prob": float("-inf"),
            "in_distribution": False,
            "retries": retries,
            "all_attempts": all_attempts,
            "percentile_info": None,
        }
    
    # Print final result summary
    print(f"\n  最终结果 (最佳候选，即使不符合分布):")
    print(f"    text_layout: {json.dumps(best_layout, indent=6, ensure_ascii=False)}")
    print(f"    log_prob: {best_log_prob:.4f}")
    print(f"    分布状态: ✗ 不符合分布 (经过 {retries} 次尝试)")
    
    if best_percentile_info:
        percentile_est = best_percentile_info.get("percentile_estimate", 50.0)
        percentile_range = best_percentile_info.get("percentile_range", "未知")
        threshold_used = best_percentile_info.get("threshold_used", filter_level)
        threshold_value = best_percentile_info.get("threshold_value", 0.0)
        distance = best_percentile_info.get("distance_from_threshold", 0.0)
        below_p1 = best_percentile_info.get("below_p1", False)
        below_p5 = best_percentile_info.get("below_p5", False)
        below_p10 = best_percentile_info.get("below_p10", False)
        
        # 说明为什么被拒绝
        if below_p1:
            reason = f"log概率 ({best_log_prob:.4f}) 低于 p1 阈值 ({threshold_value:.4f})，位于最低 1% 区域"
        elif below_p5:
            reason = f"log概率 ({best_log_prob:.4f}) 低于 p5 阈值 ({threshold_value:.4f})，位于最低 1-5% 区域"
        elif below_p10:
            reason = f"log概率 ({best_log_prob:.4f}) 低于 p10 阈值 ({threshold_value:.4f})，位于最低 5-10% 区域"
        else:
            reason = f"log概率 ({best_log_prob:.4f}) 低于 {filter_level} 阈值 ({threshold_value:.4f})"
        
        print(f"    拒绝原因: {reason}")
        print(f"    距离阈值: {distance:.4f} (负值表示低于阈值)")
        print(f"    估计百分位: {percentile_est:.2f}% ({percentile_range})")
        print(f"    使用的阈值: {threshold_used} = {threshold_value:.4f}")
        train_mean = thresholds.get("train_mean", 0.0)
        train_var = thresholds.get("train_var", 1.0)
        train_std = np.sqrt(train_var) if train_var > 0 else 1.0
        print(f"    训练数据: 均值={train_mean:.4f}, 标准差={train_std:.4f}")
    
    return {
        "text_layout": best_layout,
        "log_prob": best_log_prob,
        "in_distribution": False,
        "retries": retries,
        "all_attempts": all_attempts,
        "percentile_info": best_percentile_info,
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

