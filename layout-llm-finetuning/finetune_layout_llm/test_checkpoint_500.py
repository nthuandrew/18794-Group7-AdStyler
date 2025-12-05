#!/usr/bin/env python3
"""
使用训练好的checkpoint-500进行20条布局生成测试

用法:
    python test_checkpoint_500.py \
        --checkpoint_path output_layout_llm/checkpoint-500 \
        --layout_model_path ../train_layout_distribution/layout_prob_model.joblib \
        --thresholds_path ../train_layout_distribution/layout_thresholds.json
"""
import json
import argparse
from pathlib import Path
import sys
import random
import numpy as np

# 添加路径以便导入模块
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from infer_layout_llm import load_model_for_inference, infer_with_retry
from train_layout_distribution.layout_inference import load_model_and_thresholds


def generate_random_layout_hint():
    """生成随机的位置信息提示"""
    positions = [
        "Place text at the top center",
        "Position text in the bottom left corner",
        "Center the text horizontally",
        "Place text at the top right",
        "Position text at the bottom center",
        "Place text in the middle left",
        "Center text both horizontally and vertically",
        "Position text at the top left",
        "Place text at the bottom right",
        "Position text slightly above center",
    ]
    
    colors = [
        "use white color",
        "use black color",
        "use red color",
        "use blue color",
        "use yellow color",
        "use a bright color",
        "use a dark color",
        "use a contrasting color",
    ]
    
    alignments = [
        "left aligned",
        "center aligned",
        "right aligned",
    ]
    
    # 随机组合位置、颜色和对齐方式
    position = random.choice(positions)
    color = random.choice(colors)
    alignment = random.choice(alignments)
    
    # 随机决定是否包含所有信息
    if random.random() < 0.5:
        # 只包含位置
        return f"{position}"
    elif random.random() < 0.7:
        # 位置 + 颜色
        return f"{position}, {color}"
    else:
        # 位置 + 颜色 + 对齐
        return f"{position}, {color}, {alignment}"


def add_layout_hint_to_ad_copy(ad_copy: str) -> str:
    """在广告文案前添加位置信息"""
    layout_hint = generate_random_layout_hint()
    return f"{layout_hint}\nAd copy:\n{ad_copy}"


def load_test_ad_copies(test_json_path: str = None, num_samples: int = 20, add_layout_hints: bool = True):
    """
    加载测试广告文案
    
    Args:
        test_json_path: 测试数据JSON文件路径，如果为None则使用默认测试文案
        num_samples: 需要生成的样本数量
        add_layout_hints: 是否添加随机位置信息提示
    """
    if test_json_path and Path(test_json_path).exists():
        print(f"从 {test_json_path} 加载测试数据...")
        with open(test_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 提取前num_samples条广告文案
        ad_copies = [item.get("ad_copy", "") for item in data[:num_samples] if item.get("ad_copy")]
        print(f"✓ 加载了 {len(ad_copies)} 条测试文案")
    else:
        # 使用默认测试文案
        print("使用默认测试文案...")
        ad_copies = [
            "Summer Sale! Up to 50% OFF. Limited time only.",
            "New Arrival: Premium Quality Products",
            "Buy Now and Get Free Shipping",
            "Special Offer: 30% Discount Today",
            "Join Our Membership Program",
            "Limited Edition Collection Available Now",
            "Flash Sale: Everything Must Go!",
            "Premium Quality at Affordable Prices",
            "Shop Now and Save Big",
            "Exclusive Deal for New Customers",
            "Best Sellers: Top Rated Products",
            "Holiday Special: Extra 20% Off",
            "Free Trial: No Credit Card Required",
            "Premium Service: Satisfaction Guaranteed",
            "Early Bird Special: Book Now",
            "Weekend Sale: Up to 40% Off",
            "New Collection: Spring 2024",
            "Customer Favorite: Highly Rated",
            "Limited Stock: Order Now",
            "Special Promotion: Buy 2 Get 1 Free",
        ][:num_samples]
    
    # 如果启用，为每个广告文案添加位置信息提示
    if add_layout_hints:
        print("为测试文案添加随机位置信息提示...")
        ad_copies = [add_layout_hint_to_ad_copy(ad_copy) for ad_copy in ad_copies]
        print("✓ 已添加位置信息提示")
    
    return ad_copies


def main():
    parser = argparse.ArgumentParser(description="使用checkpoint-500进行布局生成测试")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="output_layout_llm/checkpoint-500",
        help="checkpoint-500的路径"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="基础模型名称"
    )
    parser.add_argument(
        "--layout_model_path",
        type=str,
        default="train_layout_distribution/layout_prob_model.joblib",
        help="布局概率模型路径（相对于layout-llm-finetuning目录）"
    )
    parser.add_argument(
        "--thresholds_path",
        type=str,
        default="train_layout_distribution/layout_thresholds.json",
        help="布局阈值文件路径（相对于layout-llm-finetuning目录）"
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default=None,
        help="测试数据JSON文件路径（可选，如果不提供则使用默认测试文案）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="生成样本数量"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="最大重试次数"
    )
    parser.add_argument(
        "--filter_level",
        type=str,
        default="p5",
        choices=["p1", "p5", "p10"],
        help="分布过滤级别"
    )
    parser.add_argument(
        "--add_layout_hints",
        action="store_true",
        default=True,
        help="是否在广告文案前添加随机位置信息提示（默认启用）"
    )
    parser.add_argument(
        "--no_layout_hints",
        action="store_false",
        dest="add_layout_hints",
        help="禁用位置信息提示"
    )
    
    args = parser.parse_args()
    
    # 检查checkpoint路径
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"错误: checkpoint路径不存在: {checkpoint_path}")
        print(f"请确保checkpoint-500已训练并保存在该路径")
        return
    
    print("="*80)
    print("布局生成测试 - Checkpoint 500")
    print("="*80)
    print(f"Checkpoint路径: {checkpoint_path}")
    print(f"基础模型: {args.base_model}")
    print(f"布局模型: {args.layout_model_path}")
    print(f"阈值文件: {args.thresholds_path}")
    print(f"生成样本数: {args.num_samples}")
    print("="*80)
    print()
    
    # 1. 加载微调模型
    print("步骤1: 加载微调模型...")
    try:
        model, tokenizer = load_model_for_inference(
            str(checkpoint_path),
            args.base_model
        )
        print("✓ 模型加载成功\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 2. 加载布局概率模型和阈值
    print("步骤2: 加载布局概率模型和阈值...")
    try:
        layout_model_path = Path(args.layout_model_path)
        thresholds_path = Path(args.thresholds_path)
        
        # 如果路径是相对路径，从layout-llm-finetuning目录解析
        if not layout_model_path.is_absolute():
            # 获取layout-llm-finetuning目录（脚本所在目录的父目录）
            project_root = Path(__file__).parent.parent
            # 处理相对路径（如 ../train_layout_distribution/...）
            layout_model_path = (project_root / layout_model_path).resolve()
        
        if not thresholds_path.is_absolute():
            project_root = Path(__file__).parent.parent
            thresholds_path = (project_root / thresholds_path).resolve()
        
        print(f"布局模型路径: {layout_model_path}")
        print(f"阈值文件路径: {thresholds_path}")
        
        # 检查文件是否存在
        if not layout_model_path.exists():
            print(f"错误: 布局模型文件不存在: {layout_model_path}")
            print(f"请检查路径是否正确")
            return
        
        if not thresholds_path.exists():
            print(f"错误: 阈值文件不存在: {thresholds_path}")
            print(f"请检查路径是否正确")
            return
        
        ctx = load_model_and_thresholds(
            str(layout_model_path),
            str(thresholds_path)
        )
        layout_model = ctx["model"]
        thresholds = ctx["thresholds"]
        print("✓ 布局模型和阈值加载成功\n")
    except Exception as e:
        print(f"✗ 布局模型加载失败: {e}")
        print("提示: 请确保layout_prob_model.joblib和layout_thresholds.json存在")
        print(f"当前工作目录: {Path.cwd()}")
        return
    
    # 3. 加载测试广告文案
    print("步骤3: 加载测试广告文案...")
    test_ad_copies = load_test_ad_copies(
        args.test_json_path, 
        args.num_samples,
        add_layout_hints=args.add_layout_hints
    )
    print(f"✓ 加载了 {len(test_ad_copies)} 条测试文案")
    if args.add_layout_hints:
        print("  (已添加随机位置信息提示)\n")
    else:
        print("  (未添加位置信息提示)\n")
    
    # 4. 进行生成
    print("步骤4: 开始生成布局...")
    print("="*80)
    
    results = []
    in_distribution_count = 0
    
    for idx, ad_copy in enumerate(test_ad_copies, 1):
        print(f"\n[{idx}/{len(test_ad_copies)}] 输入内容:")
        # 显示完整输入（可能包含位置信息）
        display_text = ad_copy[:150] + ('...' if len(ad_copy) > 150 else '')
        print(f"  {display_text}")
        
        try:
            result = infer_with_retry(
                model=model,
                tokenizer=tokenizer,
                ad_copy=ad_copy,
                layout_model=layout_model,
                thresholds=thresholds,
                max_retries=args.max_retries,
                filter_level=args.filter_level,
                temperature=0.7,
                top_p=0.9,
            )
            
            text_layout = result.get("text_layout")
            log_prob = result.get("log_prob", float("-inf"))
            in_dist = result.get("in_distribution", False)
            retries = result.get("retries", 0)
            percentile_info = result.get("percentile_info", {})
            
            if in_dist:
                in_distribution_count += 1
            
            results.append({
                "ad_copy": ad_copy,
                "text_layout": text_layout,
                "log_prob": log_prob,
                "in_distribution": in_dist,
                "retries": retries,
                "percentile_info": percentile_info,
            })
            
            # 详细信息已经在 infer_with_retry 中打印了
            # 这里只打印简要总结
            status_icon = "✓" if in_dist else "✗"
            print(f"  结果: {status_icon} {'符合分布' if in_dist else '不符合分布'} | log_prob={log_prob:.4f} | 重试={retries}次")
            
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            results.append({
                "ad_copy": ad_copy,
                "error": str(e),
            })
    
    # 5. 打印统计信息
    print("\n" + "="*80)
    print("生成统计")
    print("="*80)
    print(f"总生成数: {len(results)}")
    print(f"符合分布数: {in_distribution_count} ({in_distribution_count/len(results)*100:.1f}%)")
    print(f"不符合分布数: {len(results) - in_distribution_count} ({(len(results) - in_distribution_count)/len(results)*100:.1f}%)")
    
    # 计算平均log概率
    valid_results = [r for r in results if "text_layout" in r]
    if valid_results:
        avg_log_prob = sum(r["log_prob"] for r in valid_results) / len(valid_results)
        print(f"平均log概率: {avg_log_prob:.4f}")
    
    # 计算平均重试次数
    valid_results_with_retries = [r for r in valid_results if "retries" in r]
    if valid_results_with_retries:
        avg_retries = sum(r["retries"] for r in valid_results_with_retries) / len(valid_results_with_retries)
        print(f"平均重试次数: {avg_retries:.2f}")
    
    print("="*80)
    
    # 6. 保存结果到JSON文件
    output_file = Path(__file__).parent / "test_checkpoint_500_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

