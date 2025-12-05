#!/usr/bin/env python3
"""
推理辅助脚本：加载已经训练好的 layout 概率模型，
并提供简单接口来：
- 计算单个 text_layout 的 log 概率
- 根据训练集分位阈值判断是否“符合分布”
"""

from pathlib import Path
from typing import Dict, Literal
import sys
import os

import json

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from train_layout_model import load_model, layout_log_prob


def load_model_and_thresholds(
    model_path: str = "layout_prob_model.joblib",
    thresholds_path: str = "layout_thresholds.json",
) -> Dict:
    """
    加载训练好的模型和阈值配置。

    Returns:
        一个 dict，包含:
        - "model": 训练好的概率模型
        - "thresholds": 阈值信息（train 统计 + p1/p5/p10）
    """
    model = load_model(Path(model_path))
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    return {"model": model, "thresholds": thresholds}


def infer_log_prob(
    model: Dict,
    text_layout: Dict,
) -> float:
    """计算单个 text_layout 的 log 概率"""
    return layout_log_prob(model, text_layout)


def is_in_distribution(
    model: Dict,
    thresholds: Dict,
    text_layout: Dict,
    level: Literal["p1", "p5", "p10"] = "p5",
) -> bool:
    """
    判断给定 text_layout 是否符合训练分布。

    Args:
        model: 概率模型（layout_prob_model.joblib 加载的对象）
        thresholds: 阈值信息（layout_thresholds.json 加载的 dict）
        text_layout: 要评估的布局
        level: 使用哪个分位作为阈值:
            - "p1": 最严格，只允许最中间 99% 的样本
            - "p5": 默认，允许中间 95%
            - "p10": 更宽松，允许中间 90%
    """
    log_p = infer_log_prob(model, text_layout)
    th = float(thresholds.get(level))
    return log_p >= th


if __name__ == "__main__":
    # 简单示例：命令行快速测试
    ctx = load_model_and_thresholds()
    model = ctx["model"]
    thresholds = ctx["thresholds"]

    example = {
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.5,
        "alignment": "left",
        "color": "white",
    }

    lp = infer_log_prob(model, example)
    print("Example layout:", example)
    print("log prob:", lp)
    for lvl in ["p1", "p5", "p10"]:
        print(f"in_distribution @ {lvl}:", is_in_distribution(model, thresholds, example, level=lvl))


