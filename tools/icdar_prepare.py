#!/usr/bin/env python3
"""
ICDAR 数据集处理工具
====================
用于准备 ICDAR（国际文档分析与识别会议）数据集进行 OCR 训练。

支持的 ICDAR 任务：
- ICDAR 2013/2015 场景文本检测
- ICDAR 2017/2019 鲁棒阅读竞赛
- ICDAR 2021/2023 文档文字识别

功能：
1. 下载和解压官方数据集
2. 解析标注文件（JSON/XML/Text 格式）
3. 生成 CRNN 训练所需的合成数据
4. 创建训练/验证/测试集划分
5. 导出为 ApexOCR 可读取的格式

用法：
    # 下载并准备 ICDAR 2013 数据集
    python icdar_prepare.py --dataset icdar2013 --output data/icdar2013

    # 准备自定义数据集
    python icdar_prepare.py --input /path/to/images --output data/custom

    # 生成合成训练数据
    python icdar_prepare.py --synthesize --output data/synthetic --num-samples 10000

作者： ApexOCR 团队
"""

import argparse
import json
import os
import random
import string
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import fontconfig
    HAS_FONTCONFIG = True
except ImportError:
    HAS_FONTCONFIG = False


@dataclass
class OCRSample:
    """OCR 样本数据类"""
    image_path: str
    text: str
    transcription: Optional[List[Tuple]] = None  # 字符级位置信息
    language: str = 'en'
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'image_path': self.image_path,
            'text': self.text,
            'transcription': self.transcription,
            'language': self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OCRSample':
        """从字典创建实例"""
        return cls(
            image_path=data['image_path'],
            text=data['text'],
            transcription=data.get('transcription'),
            language=data.get('language', 'en')
        )


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    url: str
    download_name: str
    annotation_format: str  # 'json', 'xml', 'text', 'cropped'
    train_images: str
    train_gt: str
    test_images: str = ''
    test_gt: str = ''
    num_train_samples: int = 0
    num_test_samples: int = 0


# ICDAR 数据集配置
ICDAR_CONFIGS = {
    'icdar2013': DatasetConfig(
        name='ICDAR 2013',
        url='http://rrc.cvc.uab.es/downloads/ch4_training_images.zip',
        download_name='icdar2013',
        annotation_format='cropped',
        train_images='ch4_training_images',
        train_gt='ch4_training_gt',
        test_images='ch4_test_images',
        test_gt='',
        num_train_samples=229,
        num_test_samples=233
    ),
    'icdar2015': DatasetConfig(
        name='ICDAR 2015',
        url='http://rrc.cvc.uab.es/downloads/ch4_training_images.zip',
        download_name='icdar2015',
        annotation_format='cropped',
        train_images='ch4_training_images',
        train_gt='ch4_training_gt',
        test_images='ch4_test_images',
        test_gt='',
        num_train_samples=1000,
        num_test_samples=500
    ),
    'icdar2017': DatasetConfig(
        name='ICDAR 2017',
        url='http://rrc.cvc.uab.es/downloads/train.zip',
        download_name='icdar2017',
        annotation_format='json',
        train_images='train_images',
        train_gt='train_gts',
        test_images='test_images',
        test_gt='test_gts',
        num_train_samples=72000,
        num_test_samples=8000
    ),
    'icdar2019': DatasetConfig(
        name='ICDAR 2019',
        url='http://rrc.cvc.uab.es/downloads/ch8_training_images.zip',
        download_name='icdar2019',
        annotation_format='json',
        train_images='ch8_training_images',
        train_gt='ch8_training_gt',
        test_images='ch8_test_images',
        test_gt='',
        num_train_samples=10000,
        num_test_samples=10000
    ),
    'synth90k': DatasetConfig(
        name='Synth90k (MJSynth)',
        url='http://www.robots.ox.ac.uk/~ankush/data.tar.gz',
        download_name='synth90k',
        annotation_format='text',
        train_images='mnt/ramdisk/max/90kDICT32px',
        train_gt='annotation.txt',
        test_images='',
        test_gt='',
        num_train_samples=9000000,
        num_test_samples=0
    )
}


class ICDARDataPreparer:
    """
    ICDAR 数据集准备器
    
    处理数据集下载、解析、划分和导出。
    """
    
    # ICDAR 2017 及以后版本的 GT 格式
    GT_FORMAT_V2 = '''
    {
        "GT": [
            {
                "image_path": "xxx.jpg",
                "gt": [
                    {"points": [[0,0],[10,0],[10,10],[0,10]], "text": "hello", "language": "en"}
                ]
            }
        ]
    }
    '''
    
    def __init__(self, output_dir: str, verbose: bool = False):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'validation'
        self.test_dir = self.output_dir / 'test'
        
        for dir_path in [self.images_dir, self.labels_dir, 
                        self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(exist_ok=True)
            
    def log(self, message: str):
        """打印日志"""
        if self.verbose:
            print(f"  [INFO] {message}")
            
    def download_dataset(self, config: DatasetConfig, download_dir: str) -> bool:
        """下载数据集"""
        download_path = Path(download_dir) / f"{config.download_name}.zip"
        
        if download_path.exists():
            self.log(f"数据集已存在: {download_path}")
            return True
            
        self.log(f"下载数据集: {config.name}")
        self.log(f"URL: {config.url}")
        
        try:
            urlretrieve(config.url, download_path)
            self.log(f"下载完成: {download_path}")
            return True
        except Exception as e:
            self.log(f"下载失败: {e}")
            return False
            
    def extract_dataset(self, zip_path: str, extract_dir: str) -> bool:
        """解压数据集"""
        zip_path = Path(zip_path)
        extract_path = Path(extract_dir)
        
        if not zip_path.exists():
            self.log(f"文件不存在: {zip_path}")
            return False
            
        self.log(f"解压数据集: {zip_path}")
        
        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            self.log(f"解压完成: {extract_path}")
            return True
        except Exception as e:
            self.log(f"解压失败: {e}")
            return False
            
    def parse_icdar2013_gt(self, gt_dir: str, images_dir: str) -> List[OCRSample]:
        """
        解析 ICDAR 2013 格式的标注
        
        格式：每行一个样本，格式为 "filename\ttranscription"
        """
        samples = []
        gt_path = Path(gt_dir)
        
        # 查找 GT 文件
        gt_files = list(gt_path.glob('*.txt'))
        if not gt_files:
            # 可能是子目录格式
            gt_files = list(gt_path.rglob('*.txt'))
            
        for gt_file in gt_files:
            self.log(f"解析标注文件: {gt_file}")
            
            with open(gt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        filename = parts[0]
                        text = parts[1]
                        
                        # 查找对应的图像文件
                        image_path = Path(images_dir) / filename
                        if image_path.exists():
                            samples.append(OCRSample(
                                image_path=str(image_path),
                                text=text
                            ))
                            
        self.log(f"解析完成: {len(samples)} 个样本")
        return samples
        
    def parse_icdar2017_gt(self, gt_file: str, images_dir: str) -> List[OCRSample]:
        """
        解析 ICDAR 2017+ 格式的 JSON 标注
        
        格式：
        {
            "GT": [
                {
                    "image_path": "xxx.jpg",
                    "gt": [
                        {"points": [[x1,y1], [x2,y2], ...], "text": "hello", "language": "en"}
                    ]
                }
            ]
        }
        """
        samples = []
        
        self.log(f"解析标注文件: {gt_file}")
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for image_data in data.get('GT', []):
            image_path = image_data.get('image_path', '')
            full_path = Path(images_dir) / image_path
            
            if not full_path.exists():
                continue
                
            for gt_item in image_data.get('gt', []):
                text = gt_item.get('text', '')
                language = gt_item.get('language', 'en')
                
                samples.append(OCRSample(
                    image_path=str(full_path),
                    text=text,
                    transcription=gt_item.get('points'),
                    language=language
                ))
                
        self.log(f"解析完成: {len(samples)} 个样本")
        return samples
        
    def parse_cropped_format(self, gt_dir: str, images_dir: str) -> List[OCRSample]:
        """
        解析裁剪图像格式的标注
        
        文件名即为标注内容，如 "hello_1.jpg"
        """
        samples = []
        images_path = Path(images_dir)
        
        # 查找所有图像文件
        image_files = list(images_path.glob('*.jpg'))
        image_files.extend(images_path.glob('*.png'))
        image_files.extend(images_path.glob('*.jpeg'))
        
        for img_file in image_files:
            # 从文件名提取文本（假设格式为 "text_xxx.jpg"）
            name = img_file.stem
            
            # 尝试不同的命名格式
            # 格式1: "word_1" -> "word"
            # 格式2: "word.jpg" -> "word"
            
            text = name
            if '_' in name:
                # 尝试分割
                parts = name.rsplit('_', 1)
                if parts[-1].isdigit():
                    text = '_'.join(parts[:-1])
                    
            samples.append(OCRSample(
                image_path=str(img_file),
                text=text
            ))
            
        self.log(f"解析完成: {len(samples)} 个样本")
        return samples
        
    def prepare_dataset(self, config: DatasetConfig, download_dir: str = 'downloads',
                       split_ratio: float = 0.9) -> Tuple[List[OCRSample], List[OCRSample], List[OCRSample]]:
        """
        准备完整数据集
        
        Args:
            config: 数据集配置
            download_dir: 下载目录
            split_ratio: 训练/验证划分比例
            
        Returns:
            train_samples, val_samples, test_samples
        """
        download_dir = Path(download_dir)
        download_dir.mkdir(exist_ok=True)
        
        # 下载数据集
        if not self.download_dataset(config, str(download_dir)):
            raise RuntimeError(f"无法下载数据集: {config.name}")
            
        # 解压数据集
        zip_path = download_dir / f"{config.download_name}.zip"
        extract_dir = download_dir / config.download_name
        
        if not extract_dir.exists():
            if not self.extract_dataset(str(zip_path), str(download_dir)):
                raise RuntimeError(f"无法解压数据集: {config.name}")
                
        # 解析标注
        images_dir = extract_dir / config.train_images
        gt_dir = extract_dir / config.train_gt
        
        if not gt_dir.exists():
            gt_dir = extract_dir
            
        if config.annotation_format == 'json':
            samples = self.parse_icdar2017_gt(str(gt_dir), str(images_dir))
        elif config.annotation_format == 'cropped':
            samples = self.parse_cropped_format(str(gt_dir), str(images_dir))
        else:
            # 默认使用文本格式
            samples = self.parse_icdar2013_gt(str(gt_dir), str(images_dir))
            
        # 划分数据集
        random.shuffle(samples)
        
        split_idx = int(len(samples) * split_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # 测试集
        test_samples = []
        if config.test_images:
            test_images_dir = extract_dir / config.test_images
            test_gt_dir = extract_dir / config.test_gt
            
            if test_gt_dir.exists():
                if config.annotation_format == 'json':
                    test_samples = self.parse_icdar2017_gt(str(test_gt_dir), str(test_images_dir))
                else:
                    test_samples = self.parse_cropped_format(str(test_gt_dir), str(test_images_dir))
                    
        self.log(f"数据集划分: 训练={len(train_samples)}, 验证={len(val_samples)}, 测试={len(test_samples)}")
        
        return train_samples, val_samples, test_samples
        
    def save_samples(self, samples: List[OCRSample], subset: str):
        """保存样本到指定子目录"""
        subset_dir = self.output_dir / subset
        images_dir = subset_dir / 'images'
        labels_dir = subset_dir / 'labels'
        
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # 保存图像和标注
        for i, sample in enumerate(samples):
            # 复制图像
            src_path = Path(sample.image_path)
            if src_path.exists():
                dst_name = f"{i:08d}_{src_path.suffix}"
                dst_path = images_dir / dst_name
                
                try:
                    import shutil
                    shutil.copy2(str(src_path), str(dst_path))
                except Exception as e:
                    self.log(f"复制图像失败: {e}")
                    continue
                    
                # 保存标注
                label_data = {
                    'text': sample.text,
                    'language': sample.language,
                    'original_path': sample.image_path
                }
                
                label_path = labels_dir / f"{i:08d}.json"
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, ensure_ascii=False, indent=2)
                    
        self.log(f"保存 {subset} 集: {len(samples)} 个样本")
        
    def export_for_apexocr(self, samples: List[OCRSample], output_file: str):
        """
        导出为 ApexOCR 训练格式
        
        输出格式：每行一个 JSON 对象
        {
            "image_path": "path/to/image.jpg",
            "text": "hello world"
        }
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                line = json.dumps(sample.to_dict(), ensure_ascii=False)
                f.write(line + '\n')
                
        self.log(f"导出完成: {output_file} ({len(samples)} 个样本)")
        
    def create_vocabulary(self, samples: List[OCRSample], output_file: str,
                         include_special: bool = True) -> List[str]:
        """
        从样本创建词汇表
        
        Args:
            samples: 样本列表
            output_file: 输出文件路径
            include_special: 是否包含特殊字符（空格、blank 等）
            
        Returns:
            词汇表列表
        """
        char_count = {}
        
        for sample in samples:
            for char in sample.text:
                char_count[char] = char_count.get(char, 0) + 1
                
        # 按频率排序
        sorted_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
        vocabulary = [char for char, count in sorted_chars]
        
        # 添加特殊字符
        if include_special:
            vocabulary = [' ', '<blank>', '<unk>', '<sos>', '<eos>'] + vocabulary
            
        # 保存词汇表
        with open(output_file, 'w', encoding='utf-8') as f:
            for char in vocabulary:
                f.write(char + '\n')
                
        self.log(f"词汇表保存完成: {output_file} ({len(vocabulary)} 个字符)")
        
        return vocabulary


class SyntheticDataGenerator:
    """
    合成数据生成器
    
    使用各种字体和背景生成合成文本图像，用于数据增强。
    """
    
    # 常用字体列表（系统依赖）
    DEFAULT_FONTS = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        '/System/Library/Fonts/Helvetica.ttc',  # macOS
        'C:\\Windows\\Fonts\\arial.ttf',  # Windows
    ]
    
    # 默认字符集（英文）
    DEFAULT_CHARS = string.ascii_letters + string.digits + string.punctuation + ' '
    
    # 常用中文字符
    CHINESE_CHARS = (
        '的一是在不了有和人这中大为上个国我以要他'
        '时来用们生到作地于出就分对成会可主发年动'
        '同工也能下过子说产种面而方后多定行学法所'
        '民也经子之进等部度家将里两相一开新力十'
    )
    
    def __init__(self, output_dir: str, font_path: Optional[str] = None,
                 image_size: Tuple[int, int] = (128, 32)):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找可用字体
        self.font_path = font_path or self._find_available_font()
        
    def _find_available_font(self) -> Optional[str]:
        """查找可用的字体文件"""
        for font_path in self.DEFAULT_FONTS:
            if Path(font_path).exists():
                self.log(f"使用字体: {font_path}")
                return font_path
                
        self.log("警告：未找到系统字体，将使用默认渲染")
        return None
        
    def log(self, message: str):
        """打印日志"""
        print(f"  [INFO] {message}")
        
    def generate_sample(self, text: str, background_color: Tuple[int, int, int] = (255, 255, 255),
                       text_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
        """
        生成单个文本图像
        
        Args:
            text: 要渲染的文本
            background_color: 背景颜色 (R, G, B)
            text_color: 文字颜色 (R, G, B)
            
        Returns:
            PIL Image 对象
        """
        # 创建背景
        img = Image.new('RGB', self.image_size, background_color)
        draw = ImageDraw.Draw(img)
        
        # 估算文本大小
        if self.font_path and HAS_PIL:
            try:
                font_size = int(self.image_size[1] * 0.7)
                font = ImageFont.truetype(self.font_path, font_size)
                
                # 测量文本宽度
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 计算位置（居中）
                x = (self.image_size[0] - text_width) // 2
                y = (self.image_size[1] - text_height) // 2
                
                draw.text((x, y), text, font=font, fill=text_color)
                
            except Exception as e:
                self.log(f"字体渲染失败: {e}")
                # 使用默认渲染
                self._draw_text_fallback(draw, text, text_color)
        else:
            self._draw_text_fallback(draw, text, text_color)
            
        return img
    
    def _draw_text_fallback(self, draw: ImageDraw.ImageDraw, text: str, color: Tuple[int, int, int]):
        """使用基本绘制（无字体）"""
        # 这是一个简化的回退方案
        x = 10
        y = self.image_size[1] // 2 - 5
        for char in text:
            draw.text((x, y), char, fill=color)
            x += 8
            
    def generate_dataset(self, num_samples: int, text_source: str = 'random',
                        min_length: int = 1, max_length: int = 10,
                        output_prefix: str = 'synth') -> Tuple[List[OCRSample], str]:
        """
        生成合成数据集
        
        Args:
            num_samples: 生成样本数量
            text_source: 文本来源 ('random', 'dictionary', 'icdar')
            min_length: 文本最小长度
            max_length: 文本最大长度
            output_prefix: 输出文件前缀
            
        Returns:
            样本列表, 标注文件路径
        """
        if not HAS_PIL:
            raise ImportError("PIL is required for synthetic data generation. Install with: pip install pillow")
            
        samples = []
        
        # 文本来源处理
        if text_source == 'random':
            char_pool = self.DEFAULT_CHARS
        elif text_source == 'chinese':
            char_pool = self.CHINESE_CHARS
        else:
            char_pool = self.DEFAULT_CHARS
            
        for i in range(num_samples):
            # 生成随机文本
            length = random.randint(min_length, max_length)
            text = ''.join(random.choice(char_pool) for _ in range(length))
            
            # 生成图像
            img = self.generate_sample(text)
            
            # 保存图像
            img_filename = f"{output_prefix}_{i:08d}.png"
            img_path = self.output_dir / img_filename
            img.save(str(img_path))
            
            # 创建样本
            sample = OCRSample(
                image_path=str(img_path),
                text=text
            )
            samples.append(sample)
            
            if (i + 1) % 1000 == 0:
                self.log(f"已生成 {i + 1}/{num_samples} 个样本")
                
        # 保存标注文件
        labels_file = self.output_dir / f"{output_prefix}_labels.txt"
        with open(labels_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                img_name = Path(sample.image_path).name
                f.write(f"{img_name}\t{sample.text}\n")
                
        self.log(f"合成数据生成完成: {num_samples} 个样本")
        self.log(f"标注文件: {labels_file}")
        
        return samples, str(labels_file)


def main():
    parser = argparse.ArgumentParser(
        description='ICDAR 数据集准备工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 下载并准备 ICDAR 2017 数据集
  python icdar_prepare.py --dataset icdar2017 --output data/icdar2017
  
  # 准备自定义数据集
  python icdar_prepare.py --input /path/to/images --output data/custom --split 0.8
  
  # 生成合成数据
  python icdar_prepare.py --synthesize --output data/synthetic --num-samples 10000
  
  # 创建词汇表
  python icdar_prepare.py --vocabulary --input data/train.txt --output data/vocab.txt
        """
    )
    
    # 数据集参数
    parser.add_argument('--dataset', '-d', choices=list(ICDAR_CONFIGS.keys()),
                       help='预定义数据集名称')
    parser.add_argument('--input', '-i', help='自定义数据集输入目录')
    parser.add_argument('--output', '-o', required=True, help='输出目录')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    # 合成数据参数
    parser.add_argument('--synthesize', action='store_true', help='生成合成数据')
    parser.add_argument('--num-samples', type=int, default=10000, help='合成样本数量')
    parser.add_argument('--min-length', type=int, default=1, help='文本最小长度')
    parser.add_argument('--max-length', type=int, default=10, help='文本最大长度')
    parser.add_argument('--font', help='字体文件路径')
    
    # 数据集划分参数
    parser.add_argument('--split', type=float, default=0.9, help='训练/验证划分比例')
    
    # 词汇表参数
    parser.add_argument('--vocabulary', action='store_true', help='从输入文件创建词汇表')
    
    args = parser.parse_args()
    
    if not any([args.dataset, args.input, args.synthesize, args.vocabulary]):
        parser.print_help()
        print("\n错误：请指定 --dataset、--input、--synthesize 或 --vocabulary 参数")
        return 1
        
    # 创建准备器
    preparer = ICDARDataPreparer(args.output, args.verbose)
    
    try:
        # 准备预定义数据集
        if args.dataset:
            config = ICDAR_CONFIGS[args.dataset]
            print(f"准备数据集: {config.name}")
            
            train_samples, val_samples, test_samples = preparer.prepare_dataset(config, split_ratio=args.split)
            
            # 保存各子集
            preparer.save_samples(train_samples, 'train')
            preparer.save_samples(val_samples, 'validation')
            preparer.save_samples(test_samples, 'test')
            
            # 导出为 ApexOCR 格式
            preparer.export_for_apexocr(train_samples, str(Path(args.output) / 'train_apexocr.txt'))
            preparer.export_for_apexocr(val_samples, str(Path(args.output) / 'val_apexocr.txt'))
            
            # 创建词汇表
            preparer.create_vocabulary(train_samples, str(Path(args.output) / 'vocabulary.txt'))
            
        # 准备自定义数据集
        elif args.input:
            input_path = Path(args.input)
            
            if not input_path.exists():
                print(f"错误：输入目录不存在: {input_path}")
                return 1
                
            # 查找图像和标注
            image_files = []
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                image_files.extend(input_path.rglob(ext))
                
            if not image_files:
                print("错误：未找到图像文件")
                return 1
                
            # 创建样本
            samples = []
            for img_path in image_files:
                # 尝试查找对应的标注文件
                base_name = img_path.stem
                gt_paths = [
                    img_path.with_suffix('.txt'),
                    img_path.with_suffix('.json'),
                    input_path / f"{base_name}.txt",
                    input_path / f"{base_name}.json",
                ]
                
                text = ''
                for gt_path in gt_paths:
                    if gt_path.exists():
                        with open(gt_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        break
                        
                samples.append(OCRSample(
                    image_path=str(img_path),
                    text=text
                ))
                
            # 划分数据集
            random.shuffle(samples)
            split_idx = int(len(samples) * args.split)
            train_samples = samples[:split_idx]
            val_samples = samples[split_idx:]
            
            # 保存
            preparer.save_samples(train_samples, 'train')
            preparer.save_samples(val_samples, 'validation')
            
            # 导出
            preparer.export_for_apexocr(train_samples, str(Path(args.output) / 'train_apexocr.txt'))
            preparer.export_for_apexocr(val_samples, str(Path(args.output) / 'val_apexocr.txt'))
            
            print(f"处理完成: {len(samples)} 个样本 (训练: {len(train_samples)}, 验证: {len(val_samples)})")
            
        # 生成合成数据
        elif args.synthesize:
            if not HAS_PIL:
                print("错误：需要安装 PIL 库 (pip install pillow)")
                return 1
                
            generator = SyntheticDataGenerator(
                args.output,
                font_path=args.font,
                image_size=(256, 32)
            )
            
            samples, labels_file = generator.generate_dataset(
                args.num_samples,
                min_length=args.min_length,
                max_length=args.max_length
            )
            
            # 导出为 ApexOCR 格式
            preparer.export_for_apexocr(samples, str(Path(args.output) / 'synthetic_apexocr.txt'))
            
        # 创建词汇表
        elif args.vocabulary:
            if not args.input:
                print("错误：--vocabulary 需要 --input 参数指定标注文件")
                return 1
                
            # 解析标注文件
            samples = []
            with open(args.input, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        samples.append(OCRSample.from_dict(data))
                    except json.JSONDecodeError:
                        # 可能是旧格式 "image_path\ttext"
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            samples.append(OCRSample(
                                image_path=parts[0],
                                text=parts[1]
                            ))
                            
            vocabulary = preparer.create_vocabulary(
                samples,
                str(Path(args.output) / 'vocabulary.txt')
            )
            
            print(f"词汇表创建完成: {len(vocabulary)} 个字符")
            
        return 0
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
