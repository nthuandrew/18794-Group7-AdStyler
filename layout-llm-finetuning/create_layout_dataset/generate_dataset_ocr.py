#!/usr/bin/env python3
"""
图像到Prompt数据库生成工具（OCR版本）
使用OCR模型从广告图像中提取文本布局信息
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
import re


class AdImageAnalyzerOCR:
    """使用OCR的广告图像分析器"""
    
    def __init__(self, ocr_library: str = "easyocr"):
        """
        初始化分析器
        
        Args:
            ocr_library: OCR库名称，支持 "easyocr" 或 "paddleocr"
        """
        self.ocr_library = ocr_library.lower()
        print(f"Initializing {ocr_library} OCR...")
        
        if self.ocr_library == "easyocr":
            try:
                import easyocr
                self.reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)
                print("✓ EasyOCR initialized")
            except ImportError:
                print("Error: EasyOCR not installed. Run: pip install easyocr")
                raise
            except Exception as e:
                print(f"Error initializing EasyOCR: {e}")
                print("Trying CPU mode...")
                self.reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)
                print("✓ EasyOCR initialized (CPU mode)")
        
        elif self.ocr_library == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
                print("✓ PaddleOCR initialized")
            except ImportError:
                print("Error: PaddleOCR not installed. Run: pip install paddlepaddle paddleocr")
                raise
        else:
            raise ValueError(f"Unsupported OCR library: {ocr_library}")
    
    def analyze_image(self, image_input, ad_copy: str) -> Dict:
        """
        分析图像并生成结构化数据
        
        Args:
            image_input: 图像输入，可以是路径字符串或PIL图像对象
            ad_copy: 已有的广告文案
            
        Returns:
            包含text_layout的字典
        """
        # 加载图像（支持路径或PIL对象）
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        img_array = np.array(image)
        img_width, img_height = image.size
        
        # 使用OCR检测文本
        if self.ocr_library == "easyocr":
            ocr_results = self.reader.readtext(img_array)
            text_boxes = self._parse_easyocr_results(ocr_results, ad_copy)
        else:  # paddleocr
            ocr_results = self.ocr.ocr(img_array, cls=True)
            text_boxes = self._parse_paddleocr_results(ocr_results, ad_copy)
        
        # 分析文本布局
        text_layout = self._analyze_text_layout(
            text_boxes, img_width, img_height, img_array, ad_copy
        )
        
        return {
            "ad_copy": ad_copy,
            "text_layout": text_layout
        }
    
    def _parse_easyocr_results(self, results: List, ad_copy: str) -> List[Dict]:
        """解析EasyOCR结果"""
        text_boxes = []
        for (bbox, text, confidence) in results:
            # bbox格式: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            # 转换为 [x_min, y_min, x_max, y_max]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            text_boxes.append({
                "text": text.strip(),
                "bbox": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                "confidence": confidence
            })
        
        return text_boxes
    
    def _parse_paddleocr_results(self, results: List, ad_copy: str) -> List[Dict]:
        """解析PaddleOCR结果"""
        text_boxes = []
        if results and results[0]:
            for line in results[0]:
                if line:
                    bbox = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    text_info = line[1]  # (text, confidence)
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    text_boxes.append({
                        "text": text.strip(),
                        "bbox": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                        "confidence": confidence
                    })
        
        return text_boxes
    
    def _find_matching_text_boxes(self, text_boxes: List[Dict], ad_copy: str) -> List[Dict]:
        """
        找到与ad_copy匹配的所有文本框（可能有多行）
        
        Returns:
            匹配的文本框列表
        """
        if not text_boxes:
            return []
        
        # 清理ad_copy用于匹配
        ad_copy_clean = re.sub(r'[^\w\s]', '', ad_copy.lower())
        ad_copy_words = set(ad_copy_clean.split())
        
        matched_boxes = []
        box_scores = []
        
        # 方法1: 找到包含ad_copy中大部分单词的文本框
        for box in text_boxes:
            text_clean = re.sub(r'[^\w\s]', '', box["text"].lower())
            text_words = set(text_clean.split())
            
            # 计算重叠度
            score = 0.0
            if text_words and ad_copy_words:
                # 单词重叠度
                word_overlap = len(text_words & ad_copy_words) / len(ad_copy_words)
                score = word_overlap
            
            # 字符串包含检查
            if len(text_clean) > 0 and len(ad_copy_clean) > 0:
                if ad_copy_clean in text_clean or text_clean in ad_copy_clean:
                    score = max(score, 0.8)  # 高优先级
                else:
                    # 字符重叠度
                    common_chars = len(set(text_clean) & set(ad_copy_clean))
                    char_overlap = common_chars / max(len(text_clean), len(ad_copy_clean))
                    score = max(score, char_overlap)
            
            if score > 0.2:  # 至少20%的匹配度
                matched_boxes.append(box)
                box_scores.append(score)
        
        # 如果找到匹配的文本框，尝试找到相关的文本框（可能是同一文本块的其他行）
        if matched_boxes:
            # 找到所有匹配文本框的边界
            if len(matched_boxes) > 1:
                x_mins = [b["bbox"][0] for b in matched_boxes]
                y_mins = [b["bbox"][1] for b in matched_boxes]
                x_maxs = [b["bbox"][2] for b in matched_boxes]
                y_maxs = [b["bbox"][3] for b in matched_boxes]
                
                region_x_min = min(x_mins)
                region_y_min = min(y_mins)
                region_x_max = max(x_maxs)
                region_y_max = max(y_maxs)
                
                # 查找在附近的其他文本框（可能是同一文本块的其他行）
                for box in text_boxes:
                    if box in matched_boxes:
                        continue
                    
                    bbox = box["bbox"]
                    box_x_center = (bbox[0] + bbox[2]) / 2
                    box_y_center = (bbox[1] + bbox[3]) / 2
                    
                    # 检查是否在匹配区域的附近（允许一定的扩展）
                    margin = max(region_x_max - region_x_min, region_y_max - region_y_min) * 0.3
                    
                    if (region_x_min - margin <= box_x_center <= region_x_max + margin and
                        region_y_min - margin <= box_y_center <= region_y_max + margin):
                        matched_boxes.append(box)
        
        # 方法2: 如果方法1没找到，返回所有文本框（可能是整个广告都是文本）
        if not matched_boxes:
            matched_boxes = text_boxes
        
        return matched_boxes
    
    def _merge_boxes(self, boxes: List[Dict]) -> Dict:
        """
        合并多个文本框为一个边界框
        
        Args:
            boxes: 文本框列表
            
        Returns:
            合并后的边界框信息
        """
        if not boxes:
            return None
        
        if len(boxes) == 1:
            return boxes[0]
        
        # 找到所有边界框的最小外接矩形
        x_mins = [box["bbox"][0] for box in boxes]
        y_mins = [box["bbox"][1] for box in boxes]
        x_maxs = [box["bbox"][2] for box in boxes]
        y_maxs = [box["bbox"][3] for box in boxes]
        
        merged_bbox = [
            min(x_mins),  # x_min
            min(y_mins),  # y_min
            max(x_maxs),  # x_max
            max(y_maxs)   # y_max
        ]
        
        # 合并所有文本
        merged_text = "\n".join([box["text"] for box in boxes])
        
        return {
            "text": merged_text,
            "bbox": merged_bbox,
            "confidence": sum([box.get("confidence", 0.5) for box in boxes]) / len(boxes)
        }
    
    def _analyze_text_layout(
        self, 
        text_boxes: List[Dict], 
        img_width: int, 
        img_height: int,
        img_array: np.ndarray,
        ad_copy: str
    ) -> Dict:
        """分析文本布局（支持多行文本）"""
        # 找到所有匹配的文本框（可能有多行）
        matched_boxes = self._find_matching_text_boxes(text_boxes, ad_copy)
        
        if not matched_boxes:
            # 如果没有找到匹配的文本，返回默认值
            return {
                "x": 0.1,
                "y": 0.2,
                "width": 0.3,
                "height": 0.5,
                "alignment": "left",
                "color": "white"
            }
        
        # 合并所有匹配的文本框（处理多行情况）
        merged_box = self._merge_boxes(matched_boxes)
        
        if not merged_box:
            return {
                "x": 0.1,
                "y": 0.2,
                "width": 0.3,
                "height": 0.5,
                "alignment": "left",
                "color": "white"
            }
        
        bbox = merged_box["bbox"]
        x_min, y_min, x_max, y_max = bbox
        
        # 转换为相对坐标 (0-1)
        x = max(0.0, min(1.0, x_min / img_width))
        y = max(0.0, min(1.0, y_min / img_height))
        width = max(0.0, min(1.0, (x_max - x_min) / img_width))
        height = max(0.0, min(1.0, (y_max - y_min) / img_height))
        
        # 判断对齐方式
        alignment = self._determine_alignment(x, width, img_width)
        
        # 提取文本颜色
        color = self._extract_text_color(img_array, bbox)
        
        return {
            "x": round(x, 3),
            "y": round(y, 3),
            "width": round(width, 3),
            "height": round(height, 3),
            "alignment": alignment,
            "color": color
        }
    
    def _determine_alignment(self, x: float, width: float, img_width: int) -> str:
        """判断文本对齐方式"""
        # 计算文本中心位置
        text_center_x = x + width / 2
        
        if text_center_x < 0.33:
            return "left"
        elif text_center_x > 0.67:
            return "right"
        else:
            return "center"
    
    def _extract_text_color(self, img_array: np.ndarray, bbox: List[float]) -> str:
        """从图像中提取文本颜色"""
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
        
        # 确保坐标在图像范围内
        x_min = max(0, min(x_min, img_array.shape[1] - 1))
        y_min = max(0, min(y_min, img_array.shape[0] - 1))
        x_max = max(0, min(x_max, img_array.shape[1] - 1))
        y_max = max(0, min(y_max, img_array.shape[0] - 1))
        
        if x_max <= x_min or y_max <= y_min:
            return "white"  # 默认值
        
        # 提取文本区域
        text_region = img_array[y_min:y_max, x_min:x_max]
        
        if text_region.size == 0:
            return "white"
        
        # 计算平均颜色
        avg_color = np.mean(text_region.reshape(-1, 3), axis=0)
        
        # 转换为颜色名称
        color_name = self._rgb_to_color_name(avg_color)
        return color_name
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """将RGB值转换为颜色名称"""
        r, g, b = rgb
        
        # 常见颜色定义
        colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
            "cyan": (0, 255, 255),
            "gray": (128, 128, 128),
            "brown": (165, 42, 42),
        }
        
        # 找到最接近的颜色
        min_distance = float('inf')
        closest_color = "white"
        
        for color_name, color_rgb in colors.items():
            distance = np.sqrt(
                (r - color_rgb[0])**2 + 
                (g - color_rgb[1])**2 + 
                (b - color_rgb[2])**2
            )
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        # 如果距离太远，可能是自定义颜色，返回最接近的基础颜色
        if min_distance > 100:
            # 根据亮度判断
            brightness = (r + g + b) / 3
            if brightness > 200:
                return "white"
            elif brightness < 50:
                return "black"
            else:
                return closest_color
        
        return closest_color


def load_ad_copy_mapping(copy_file: str) -> Dict[str, str]:
    """加载广告文案映射"""
    mapping = {}
    
    if not os.path.exists(copy_file):
        print(f"Warning: Ad copy file not found: {copy_file}")
        return mapping
    
    ext = Path(copy_file).suffix.lower()
    
    if ext == '.json':
        with open(copy_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    elif ext == '.csv':
        import csv
        with open(copy_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'image_path' in row and 'ad_copy' in row:
                    mapping[row['image_path']] = row['ad_copy']
    else:
        with open(copy_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        mapping[parts[0].strip()] = parts[1].strip()
    
    return mapping


def process_from_dataset(
    dataset_path: str,
    output_path: str,
    ocr_library: str = "easyocr",
    split: str = "train",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    output_mode: str = "compact"
):
    """
    从HuggingFace datasets加载并处理图像
    
    Args:
        dataset_path: 数据集路径
        output_path: 输出路径（文件或文件夹）
        ocr_library: OCR库
        split: 数据集split
        start_idx: 起始索引
        end_idx: 结束索引
        output_mode: 输出模式，"compact"（单个JSON）或 "separate"（每个文件单独保存）
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        print("Error: datasets library not installed. Run: pip install datasets")
        raise
    
    print(f"Loading dataset from {dataset_path}...")
    ad_data = load_from_disk(dataset_path)
    
    if split not in ad_data:
        available_splits = list(ad_data.keys())
        raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")
    
    dataset = ad_data[split]
    total_samples = len(dataset)
    
    if end_idx is None:
        end_idx = total_samples
    
    end_idx = min(end_idx, total_samples)
    num_samples = end_idx - start_idx
    
    print(f"Processing samples {start_idx} to {end_idx} (total: {total_samples})")
    print(f"Output mode: {output_mode}")
    
    analyzer = AdImageAnalyzerOCR(ocr_library=ocr_library)
    
    # 如果是separate模式，创建输出文件夹
    if output_mode == "separate":
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        json_dir = output_dir / "json"
        images_dir.mkdir(exist_ok=True)
        json_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir}")
        print(f"  Images: {images_dir}")
        print(f"  JSONs: {json_dir}")
    
    results = []
    for idx in range(start_idx, end_idx):
        sample = dataset[idx]
        
        # 获取图像和文本
        image = sample.get("image")
        ad_copy = sample.get("text", "")
        
        if image is None:
            print(f"Warning: Sample {idx} has no image, skipping...")
            continue
        
        if not ad_copy:
            print(f"Warning: Sample {idx} has no text, skipping...")
            continue
        
        print(f"[{idx - start_idx + 1}/{num_samples}] Processing sample {idx}...")
        
        try:
            result = analyzer.analyze_image(image, ad_copy)
            result["sample_idx"] = idx
            
            if output_mode == "separate":
                # 保存图像和JSON到单独文件
                image_filename = f"sample_{idx:06d}.jpg"
                json_filename = f"sample_{idx:06d}.json"
                
                image_path = images_dir / image_filename
                json_path = json_dir / json_filename
                
                # 保存图像
                image.save(image_path, "JPEG", quality=95)
                
                # 保存JSON
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ Saved: {image_filename} + {json_filename}")
            else:
                # compact模式，添加到结果列表
                results.append(result)
                print(f"  ✓ Completed - Layout: x={result['text_layout']['x']:.2f}, "
                      f"y={result['text_layout']['y']:.2f}, "
                      f"color={result['text_layout']['color']}")
                      
        except Exception as e:
            print(f"  ✗ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # compact模式：保存所有结果到一个JSON文件
    if output_mode == "compact":
        print(f"\nSaving results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Generated {len(results)} entries in {output_path}")
    else:
        print(f"\n✓ Generated {num_samples} separate files in {output_path}")
        print(f"  - Images: {images_dir}")
        print(f"  - JSONs: {json_dir}")


def process_images(
    image_dir: str,
    ad_copy_file: str,
    output_file: str,
    ocr_library: str = "easyocr"
):
    """处理目录中的图像文件（旧方法，保留兼容性）"""
    print(f"Loading ad copy mapping from {ad_copy_file}...")
    ad_copy_mapping = load_ad_copy_mapping(ad_copy_file)
    print(f"Loaded {len(ad_copy_mapping)} ad copy entries")
    
    analyzer = AdImageAnalyzerOCR(ocr_library=ocr_library)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        
        ad_copy = ad_copy_mapping.get(image_file, "")
        if not ad_copy:
            base_name = Path(image_file).stem
            ad_copy = ad_copy_mapping.get(base_name, "")
        
        if not ad_copy:
            print(f"Warning: No ad copy found for {image_file}, skipping...")
            continue
        
        print(f"[{idx}/{len(image_files)}] Processing {image_file}...")
        
        try:
            result = analyzer.analyze_image(image_path, ad_copy)
            result["image_path"] = image_file
            results.append(result)
            print(f"  ✓ Completed - Layout: x={result['text_layout']['x']:.2f}, "
                  f"y={result['text_layout']['y']:.2f}, "
                  f"color={result['text_layout']['color']}")
        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Generated {len(results)} entries in {output_file}")


def main():
    parser = argparse.ArgumentParser(description="生成图像到Prompt数据库（OCR版本）")
    
    # 数据集模式
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="HuggingFace数据集路径（使用此参数时，忽略image_dir和ad_copy_file）")
    parser.add_argument("--split", type=str, default="train",
                       help="数据集split名称 (默认: train)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="起始样本索引 (默认: 0)")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="结束样本索引 (默认: 全部)")
    
    # 文件模式（旧方法）
    parser.add_argument("--image_dir", type=str, default=None,
                       help="图像目录路径")
    parser.add_argument("--ad_copy_file", type=str, default=None,
                       help="广告文案文件路径 (JSON/CSV/TXT)")
    
    # 通用参数
    parser.add_argument("--output", type=str, default="dataset.json",
                       help="输出路径（文件或文件夹，取决于output_mode）")
    parser.add_argument("--output_mode", type=str, default="compact",
                       choices=["compact", "separate"],
                       help="输出模式: compact（单个JSON文件）或 separate（每个文件单独保存）")
    parser.add_argument("--ocr", type=str, default="easyocr",
                       choices=["easyocr", "paddleocr"],
                       help="OCR库选择 (默认: easyocr)")
    
    args = parser.parse_args()
    
    # 判断使用哪种模式
    if args.dataset_path:
        # 使用数据集模式
        process_from_dataset(
            dataset_path=args.dataset_path,
            output_path=args.output,
            ocr_library=args.ocr,
            split=args.split,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            output_mode=args.output_mode
        )
    elif args.image_dir and args.ad_copy_file:
        # 使用文件模式（暂时只支持compact模式）
        if args.output_mode == "separate":
            print("Warning: separate mode not yet supported for file mode, using compact mode")
        process_images(
            image_dir=args.image_dir,
            ad_copy_file=args.ad_copy_file,
            output_file=args.output,
            ocr_library=args.ocr
        )
    else:
        parser.error("必须提供 --dataset_path 或同时提供 --image_dir 和 --ad_copy_file")


if __name__ == "__main__":
    main()

