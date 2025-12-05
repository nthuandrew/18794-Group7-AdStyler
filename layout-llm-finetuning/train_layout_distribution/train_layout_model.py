#!/usr/bin/env python3
"""
使用 train_layout.json 和 test_layout.json 训练 text_layout 概率分布模型。

思路：
- 连续变量: x, y, width, height -> 用高斯混合模型(Gaussian Mixture)建模 p_continuous
- 离散变量: alignment, color -> 用经验频率估计 p(alignment), p(color)

最终:
  p(text_layout) = p_continuous(x, y, width, height) * p(alignment) * p(color)

注意：对连续变量来说这是概率密度，不是严格的“概率”，但可以用来比较不同布局的相对合理程度。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
import joblib


def load_layouts(json_paths: List[Path]) -> Tuple[np.ndarray, List[str], List[str]]:
    """从多个 JSON 文件中加载 text_layout 数据"""
    xs = []
    alignments: List[str] = []
    colors: List[str] = []

    for path in json_paths:
        print(f"Loading {path} ...")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            tl = item.get("text_layout", {})
            try:
                x = float(tl.get("x", 0.0))
                y = float(tl.get("y", 0.0))
                w = float(tl.get("width", 0.0))
                h = float(tl.get("height", 0.0))
            except (TypeError, ValueError):
                continue

            xs.append([x, y, w, h])
            alignments.append(str(tl.get("alignment", "center")).lower())
            colors.append(str(tl.get("color", "white")).lower())

    X = np.asarray(xs, dtype=np.float32)
    print(f"Loaded {len(X)} layouts from {len(json_paths)} files")
    return X, alignments, colors


def load_layouts_only_continuous(json_path: Path) -> List[Dict]:
    """
    仅加载 text_layout 字典列表，用于评估 log 概率。
    保留原始 dict，便于后续做百分位映射等。
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    layouts: List[Dict] = []
    for item in data:
        tl = item.get("text_layout")
        if not tl:
            continue
        layouts.append(tl)
    return layouts


def estimate_categorical_probs(values: List[str]) -> Dict[str, float]:
    """基于频率估计离散变量的概率分布，加入简单平滑避免为 0"""
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    total = sum(counts.values())
    # Laplace smoothing
    k = len(counts)
    probs = {v: (c + 1) / (total + k) for v, c in counts.items()}
    return probs


def train_model(
    train_json: Path,
    n_components: int = 5,
    random_state: int = 42,
) -> Dict:
    """只用 train_json 训练概率模型并返回模型对象 dict（方便保存）"""
    X, alignments, colors = load_layouts([train_json])

    print(f"Fitting Gaussian Mixture with {n_components} components on {X.shape[0]} samples ...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
    )
    gmm.fit(X)

    alignment_probs = estimate_categorical_probs(alignments)
    color_probs = estimate_categorical_probs(colors)

    print("Estimated alignment distribution:", alignment_probs)
    print("Estimated color distribution:", color_probs)

    model = {
        "gmm": gmm,
        "alignment_probs": alignment_probs,
        "color_probs": color_probs,
    }
    return model


def save_model(model: Dict, path: Path) -> None:
    """保存模型到磁盘"""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: Path) -> Dict:
    """从磁盘加载模型"""
    return joblib.load(path)


def layout_log_prob(model: Dict, text_layout: Dict) -> float:
    """
    计算给定 text_layout 的对数概率（log probability）。

    text_layout: {
        "x": float,
        "y": float,
        "width": float,
        "height": float,
        "alignment": "left|center|right",
        "color": "white|black|..."
    }
    """
    gmm: GaussianMixture = model["gmm"]
    alignment_probs: Dict[str, float] = model["alignment_probs"]
    color_probs: Dict[str, float] = model["color_probs"]

    x = float(text_layout["x"])
    y = float(text_layout["y"])
    w = float(text_layout["width"])
    h = float(text_layout["height"])
    X = np.array([[x, y, w, h]], dtype=np.float32)

    # 连续部分的 log pdf
    log_p_cont = float(gmm.score_samples(X)[0])

    # 离散部分的 log prob，加上简单平滑（未出现的给一个很小的概率）
    align = str(text_layout.get("alignment", "center")).lower()
    color = str(text_layout.get("color", "white")).lower()

    eps = 1e-6
    p_align = alignment_probs.get(align, eps)
    p_color = color_probs.get(color, eps)

    log_p_cat = float(np.log(p_align) + np.log(p_color))

    return log_p_cont + log_p_cat


def compute_log_probs_for_layouts(model: Dict, layouts: List[Dict]) -> np.ndarray:
    """对一批 text_layout 计算 log 概率，返回 numpy 数组"""
    log_probs: List[float] = []
    for tl in layouts:
        try:
            lp = layout_log_prob(model, tl)
            log_probs.append(lp)
        except Exception:
            continue
    return np.asarray(log_probs, dtype=np.float32)


def percentile_scores_from_train(train_log_probs: np.ndarray, test_log_probs: np.ndarray) -> np.ndarray:
    """
    基于 train 的 log 概率分布，为每个 test log 概率计算 0-1 百分位分数：
      score(x) = P_train( log_p <= log_p(x) )
    """
    sorted_train = np.sort(train_log_probs)
    n = len(sorted_train)
    if n == 0:
        return np.zeros_like(test_log_probs)

    scores = []
    for lp in test_log_probs:
        # 找到在 train 中 <= lp 的数量
        idx = np.searchsorted(sorted_train, lp, side="right")
        score = idx / n
        scores.append(score)
    return np.asarray(scores, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "训练 text_layout 概率分布模型（仅使用 train_layout.json），"
            "并在 test_layout.json 上计算 log 概率的均值和方差。"
        )
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="train_layout.json",
        help="train 布局 JSON 文件路径",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="test_layout.json",
        help="test 布局 JSON 文件路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="layout_prob_model.joblib",
        help="保存模型的文件路径",
    )
    parser.add_argument(
        "--thresholds_path",
        type=str,
        default="layout_thresholds.json",
        help="保存基于 train log 概率计算的阈值（分位数）的 JSON 文件路径",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=5,
        help="GMM 组件数（越大越灵活，但也越容易过拟合）",
    )
    parser.add_argument(
        "--eval_example",
        action="store_true",
        help="训练完后用一个示例 text_layout 计算概率",
    )

    args = parser.parse_args()

    train_json = Path(args.train_json)
    test_json = Path(args.test_json)
    model_path = Path(args.model_path)
    thresholds_path = Path(args.thresholds_path)

    # 1) 训练模型（仅用 train）
    model = train_model(
        train_json=train_json,
        n_components=args.n_components,
    )
    save_model(model, model_path)

    # 2) 在 train / test 上评估 log 概率和百分位分数
    print("\nEvaluating log probabilities on train & test data ...")

    train_layouts = load_layouts_only_continuous(train_json)
    test_layouts = load_layouts_only_continuous(test_json)

    train_log_probs = compute_log_probs_for_layouts(model, train_layouts)
    test_log_probs = compute_log_probs_for_layouts(model, test_layouts)

    if len(train_log_probs) == 0:
        print("Warning: no valid text_layout found in train_json for evaluation.")
    else:
        mean_lp_tr = float(train_log_probs.mean())
        var_lp_tr = float(train_log_probs.var())
        print(f"Train log prob count: {len(train_log_probs)}")
        print(f"Train log prob mean: {mean_lp_tr:.4f}")
        print(f"Train log prob var : {var_lp_tr:.4f}")

        # 3) 基于 train log_probs 计算一些常用分位阈值并保存
        p1 = float(np.quantile(train_log_probs, 0.01))
        p5 = float(np.quantile(train_log_probs, 0.05))
        p10 = float(np.quantile(train_log_probs, 0.10))
        thresholds = {
            "train_count": int(len(train_log_probs)),
            "train_mean": mean_lp_tr,
            "train_var": var_lp_tr,
            "p1": p1,
            "p5": p5,
            "p10": p10,
        }
        with thresholds_path.open("w", encoding="utf-8") as f:
            json.dump(thresholds, f, ensure_ascii=False, indent=2)
        print(f"\nSaved thresholds to {thresholds_path}:")
        print(f"  p1  (1%): {p1:.4f}")
        print(f"  p5  (5%): {p5:.4f}")
        print(f"  p10(10%): {p10:.4f}")

    if len(test_log_probs) == 0:
        print("Warning: no valid text_layout found in test_json for evaluation.")
    else:
        mean_lp_te = float(test_log_probs.mean())
        var_lp_te = float(test_log_probs.var())
        print(f"\nTest log prob count: {len(test_log_probs)}")
        print(f"Test log prob mean: {mean_lp_te:.4f}")
        print(f"Test log prob var : {var_lp_te:.4f}")

        # 4) 基于 train log_probs 计算 test 的 0-1 百分位分数
        if len(train_log_probs) > 0:
            test_scores = percentile_scores_from_train(train_log_probs, test_log_probs)
            mean_score = float(test_scores.mean())
            var_score = float(test_scores.var())
            print("\nTest percentile scores relative to train log-prob distribution:")
            print(f"Test score mean (0-1): {mean_score:.4f}")
            print(f"Test score var  (0-1): {var_score:.4f}")

    if args.eval_example:
        example = {
            "x": 0.1,
            "y": 0.2,
            "width": 0.3,
            "height": 0.5,
            "alignment": "left",
            "color": "white",
        }
        log_p = layout_log_prob(model, example)
        print("Example layout:", example)
        print("Log probability:", log_p)


if __name__ == "__main__":
    main()


