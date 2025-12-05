#!/usr/bin/env python3
"""
Prepare SFT (Supervised Fine-Tuning) dataset from train_layout.json.

This module provides functions to:
1. Load train_layout.json
2. Filter samples using layout_prob_model to keep only in-distribution layouts
3. Convert to instruction-following format for LLM fine-tuning
4. Save as JSONL format (one example per line)
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path to import from train_layout_distribution
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_layout_distribution.layout_inference import (
    load_model_and_thresholds,
    is_in_distribution,
    infer_log_prob,
)
import numpy as np
from collections import Counter


def format_text_layout_json(text_layout: Dict) -> str:
    """Format text_layout dict as a clean JSON string."""
    return json.dumps({"text_layout": text_layout}, indent=2, ensure_ascii=False)


def analyze_filtering_impact(
    raw_data: List[Dict],
    filtered_data: List[Dict],
    model: Dict,
    thresholds: Dict,
) -> None:
    """
    Analyze the impact of filtering on data distribution.
    Compares statistics between raw and filtered data.
    """
    print("\n" + "="*60)
    print("Data Filtering Impact Analysis")
    print("="*60)
    
    # Basic statistics
    raw_count = len(raw_data)
    filtered_count = len(filtered_data)
    retention_rate = filtered_count / raw_count * 100 if raw_count > 0 else 0
    
    print(f"\nSample Count:")
    print(f"  Raw samples: {raw_count}")
    print(f"  Filtered samples: {filtered_count}")
    print(f"  Retention rate: {retention_rate:.2f}%")
    print(f"  Filtered out: {raw_count - filtered_count} ({100 - retention_rate:.2f}%)")
    
    # Analyze alignment distribution
    raw_alignments = [s.get("text_layout", {}).get("alignment", "unknown") for s in raw_data if s.get("text_layout")]
    filtered_alignments = [s.get("text_layout", {}).get("alignment", "unknown") for s in filtered_data if s.get("text_layout")]
    
    raw_align_counter = Counter(raw_alignments)
    filtered_align_counter = Counter(filtered_alignments)
    
    print(f"\nAlignment Distribution:")
    for align in ["left", "center", "right"]:
        raw_count_align = raw_align_counter.get(align, 0)
        filtered_count_align = filtered_align_counter.get(align, 0)
        raw_pct = raw_count_align / len(raw_alignments) * 100 if raw_alignments else 0
        filtered_pct = filtered_count_align / len(filtered_alignments) * 100 if filtered_alignments else 0
        print(f"  {align:6s}: {raw_count_align:5d} ({raw_pct:5.1f}%) -> {filtered_count_align:5d} ({filtered_pct:5.1f}%)")
    
    # Analyze color distribution
    raw_colors = [s.get("text_layout", {}).get("color", "unknown") for s in raw_data if s.get("text_layout")]
    filtered_colors = [s.get("text_layout", {}).get("color", "unknown") for s in filtered_data if s.get("text_layout")]
    
    raw_color_counter = Counter(raw_colors)
    filtered_color_counter = Counter(filtered_colors)
    
    print(f"\nTop 5 Colors (Raw -> Filtered):")
    top_colors = sorted(raw_color_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    for color, raw_count_color in top_colors:
        filtered_count_color = filtered_color_counter.get(color, 0)
        raw_pct = raw_count_color / len(raw_colors) * 100 if raw_colors else 0
        filtered_pct = filtered_count_color / len(filtered_colors) * 100 if filtered_colors else 0
        print(f"  {color:10s}: {raw_count_color:5d} ({raw_pct:5.1f}%) -> {filtered_count_color:5d} ({filtered_pct:5.1f}%)")
    
    # Analyze coordinate statistics
    def extract_coords(data):
        coords = []
        for s in data:
            tl = s.get("text_layout", {})
            if tl:
                coords.append({
                    "x": float(tl.get("x", 0)),
                    "y": float(tl.get("y", 0)),
                    "width": float(tl.get("width", 0)),
                    "height": float(tl.get("height", 0)),
                })
        return coords
    
    raw_coords = extract_coords(raw_data)
    filtered_coords = extract_coords(filtered_data)
    
    if raw_coords and filtered_coords:
        print(f"\nCoordinate Statistics (mean ± std):")
        for coord_name in ["x", "y", "width", "height"]:
            raw_vals = [c[coord_name] for c in raw_coords]
            filtered_vals = [c[coord_name] for c in filtered_coords]
            raw_mean = np.mean(raw_vals)
            raw_std = np.std(raw_vals)
            filtered_mean = np.mean(filtered_vals)
            filtered_std = np.std(filtered_vals)
            print(f"  {coord_name:6s}: {raw_mean:.3f} ± {raw_std:.3f} -> {filtered_mean:.3f} ± {filtered_std:.3f}")
    
    # Analyze log probability distribution
    raw_log_probs = []
    filtered_log_probs = []
    
    for s in raw_data:
        tl = s.get("text_layout")
        if tl:
            try:
                lp = infer_log_prob(model, tl)
                raw_log_probs.append(lp)
            except:
                pass
    
    for s in filtered_data:
        tl = s.get("text_layout")
        if tl:
            try:
                lp = infer_log_prob(model, tl)
                filtered_log_probs.append(lp)
            except:
                pass
    
    if raw_log_probs and filtered_log_probs:
        print(f"\nLog Probability Statistics:")
        raw_mean_lp = np.mean(raw_log_probs)
        raw_std_lp = np.std(raw_log_probs)
        filtered_mean_lp = np.mean(filtered_log_probs)
        filtered_std_lp = np.std(filtered_log_probs)
        print(f"  Raw:      {raw_mean_lp:.3f} ± {raw_std_lp:.3f}")
        print(f"  Filtered: {filtered_mean_lp:.3f} ± {filtered_std_lp:.3f}")
        print(f"  Threshold (p5): {thresholds.get('p5', 'N/A')}")
    
    print("="*60 + "\n")


def create_sft_example(ad_copy: str, text_layout: Dict) -> Dict:
    """
    Create a single SFT training example.
    
    Format:
    {
        "instruction": "...",
        "input": "Ad copy: ...",
        "output": "{\"text_layout\": {...}}"
    }
    """
    instruction = (
        "You are an ad design assistant. Given an ad copy or ad design requirement, "
        "generate a text layout JSON for the ad image. The JSON must contain a 'text_layout' object "
        "with fields: x, y, width, height (all floats 0-1, 3 decimal places), "
        "alignment (one of: left, center, right), and color (a color name). "
        "Output only the JSON, no explanations."
    )
    
    input_text = f"Ad copy:\n{ad_copy}"
    output_text = format_text_layout_json(text_layout)
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }


def prepare_dataset(
    train_json_path: str,
    output_path: str,
    model_path: str = "layout_prob_model.joblib",
    thresholds_path: str = "layout_thresholds.json",
    filter_level: str = "p5",
    min_log_prob: float = None,
) -> None:
    """
    Prepare SFT dataset from train_layout.json.
    
    Args:
        train_json_path: Path to train_layout.json
        output_path: Output JSONL file path
        model_path: Path to layout probability model
        thresholds_path: Path to layout_thresholds.json
        filter_level: Distribution filter level ("p1", "p5", "p10")
        min_log_prob: Optional minimum log probability threshold (overrides filter_level)
    """
    print(f"Loading layout model and thresholds...")
    ctx = load_model_and_thresholds(model_path, thresholds_path)
    model = ctx["model"]
    thresholds = ctx["thresholds"]
    
    print(f"Loading training data from {train_json_path}...")
    with open(train_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} raw samples")
    
    # Filter by distribution
    filtered_data = []
    skipped_count = 0
    
    for idx, sample in enumerate(raw_data):
        if idx % 1000 == 0:
            print(f"Processing {idx}/{len(raw_data)}...")
        
        text_layout = sample.get("text_layout")
        if not text_layout:
            skipped_count += 1
            continue
        
        # Check if in distribution
        if min_log_prob is not None:
            log_p = infer_log_prob(model, text_layout)
            if log_p < min_log_prob:
                skipped_count += 1
                continue
        else:
            if not is_in_distribution(model, thresholds, text_layout, level=filter_level):
                skipped_count += 1
                continue
        
        ad_copy = sample.get("ad_copy", "")
        if not ad_copy:
            skipped_count += 1
            continue
        
        filtered_data.append(sample)
    
    print(f"\nFiltered: {len(filtered_data)} samples kept, {skipped_count} skipped")
    
    # Analyze filtering impact
    analyze_filtering_impact(raw_data, filtered_data, model, thresholds)
    
    # Convert to SFT format
    print("Converting to SFT format...")
    sft_examples = []
    for sample in filtered_data:
        example = create_sft_example(
            sample["ad_copy"],
            sample["text_layout"]
        )
        sft_examples.append(example)
    
    # Save as JSONL (one example per line)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(sft_examples)} examples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for example in sft_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"✓ Dataset saved to {output_path}")
    print(f"  Total examples: {len(sft_examples)}")
    
    # Print a sample
    if sft_examples:
        print("\nSample example:")
        print(json.dumps(sft_examples[0], indent=2, ensure_ascii=False))


# Module functions can be imported and used directly
# Example usage:
#   from prepare_sft_dataset import prepare_dataset
#   prepare_dataset(
#       train_json_path="train_layout.json",
#       output_path="sft_dataset.jsonl",
#       model_path="layout_prob_model.joblib",
#       thresholds_path="layout_thresholds.json",
#       filter_level="p5"
#   )

