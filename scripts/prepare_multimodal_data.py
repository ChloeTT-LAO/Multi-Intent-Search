#!/usr/bin/env python3
"""
多模态数据准备脚本 - 处理多模态数据集并生成训练用的搜索轨迹
"""

import os
import sys
import argparse
import json
import subprocess
import gdown
import zipfile
import shutil
import base64
import io
import time
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Union

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config import CONFIG
from src.multimodal.multimodal_data import MultimodalDataPipeline
from src.utils.logging_utils import setup_logger


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Prepare Multimodal StepSearch training data")

    parser.add_argument('--raw_data_dir', type=str, default='./data/multimodal/raw',
                        help='Directory containing raw multimodal datasets')
    parser.add_argument('--output_dir', type=str, default='./data/multimodal/processed',
                        help='Output directory for processed multimodal data')
    parser.add_argument('--max_train_samples', type=int, default=1000,
                        help='Maximum training samples to process')
    parser.add_argument('--max_dev_samples', type=int, default=200,
                        help='Maximum development samples to process')
    parser.add_argument('--dataset', type=str, default='vqa',
                        choices=['vqa', 'mmqa', 'okvqa', 'gqa', 'visual_reasoning', 'custom'],
                        help='Multimodal dataset to process')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key for GPT-4V processing')
    parser.add_argument('--deepseek_api_key', type=str, default='sk-anbsoppiznxtiuhzxdibxuvpnhsxoabbsderulnnzsfduyrq',
                        help='DeepSeek API key for processing')
    parser.add_argument('--skip_gpt4v', action='store_true',
                        help='Skip GPT-4V processing and use simple decomposition')
    parser.add_argument('--skip_deepseek', action='store_true',
                        help='Skip DeepSeek processing')
    parser.add_argument('--image_analysis', action='store_true',
                        help='Enable image analysis for better query generation')
    parser.add_argument('--download_images', action='store_true',
                        help='Download images from URLs if needed')
    parser.add_argument('--image_format', type=str, default='base64',
                        choices=['base64', 'path', 'url'],
                        help='How to store image information')

    return parser.parse_args()


def setup_environment(args):
    """设置环境"""
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建图像目录
    image_dir = output_dir / 'images'
    image_dir.mkdir(exist_ok=True)

    # 设置日志
    logger = setup_logger('prepare_multimodal_data', output_dir / 'prepare_multimodal_data.log')

    # 设置API keys
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
        logger.info("OpenAI API key set from command line argument")
    elif os.getenv('OPENAI_API_KEY'):
        logger.info("Using OpenAI API key from environment variable")
    elif not args.skip_gpt4v:
        logger.warning("No OpenAI API key provided. Use --skip-gpt4v for simple processing.")

    if args.deepseek_api_key:
        os.environ['DEEPSEEK_API_KEY'] = args.deepseek_api_key
        logger.info("DeepSeek API key set from command line argument")
    elif os.getenv('DEEPSEEK_API_KEY'):
        logger.info("Using DeepSeek API key from environment variable")
    elif not args.skip_deepseek:
        logger.warning("No DeepSeek API key provided. Use --skip-deepseek to disable DeepSeek processing.")

    return logger, output_dir, image_dir


def download_vqa_dataset(raw_data_dir: str, logger):
    """下载VQA数据集"""
    try:
        from datasets import load_dataset

        raw_data_dir = Path(raw_data_dir)
        vqa_dir = raw_data_dir / 'vqa'

        if vqa_dir.exists():
            logger.info("VQA dataset already exists")
            return True

        vqa_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading VQA dataset...")
        dataset = load_dataset("HuggingFaceM4/VQAv2", split="train[:1000]")  # 只下载1000个样本用于测试

        # 保存数据
        data = []
        for item in dataset:
            sample = {
                'id': item.get('question_id', ''),
                'question': item['question'],
                'image': item['image'],  # PIL Image
                'answer': item['multiple_choice_answer'],
                'answers': item.get('answers', [])
            }
            data.append(sample)

        # 保存为JSON（图像转换为base64）
        processed_data = []
        for item in data:
            processed_item = item.copy()

            # 将PIL图像转换为base64
            if item['image']:
                buffered = io.BytesIO()
                item['image'].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                processed_item['image'] = f"data:image/png;base64,{img_str}"

            processed_data.append(processed_item)

        with open(vqa_dir / 'train.json', 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        logger.info(f"VQA dataset downloaded: {len(data)} samples")
        return True

    except Exception as e:
        logger.error(f"Failed to download VQA dataset: {e}")
        return False


def download_custom_dataset(raw_data_dir: str, dataset_url: str, logger):
    """下载自定义多模态数据集"""
    try:
        raw_data_dir = Path(raw_data_dir)

        if dataset_url.endswith('.zip'):
            # 下载并解压zip文件
            zip_path = raw_data_dir / 'dataset.zip'

            logger.info(f"Downloading dataset from {dataset_url}")
            if dataset_url.startswith('https://drive.google.com'):
                # Google Drive链接
                file_id = dataset_url.split('/')[-2]
                gdown.download(f"https://drive.google.com/uc?id={file_id}", str(zip_path), quiet=False)
            else:
                # 普通HTTP链接
                import requests
                response = requests.get(dataset_url)
                with open(zip_path, 'wb') as f:
                    f.write(response.content)

            # 解压
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_data_dir)

            zip_path.unlink()  # 删除zip文件

        return True

    except Exception as e:
        logger.error(f"Failed to download custom dataset: {e}")
        return False


def download_datasets(raw_data_dir: str, dataset: str, logger):
    """下载多模态数据集"""
    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    if dataset == 'vqa':
        return download_vqa_dataset(raw_data_dir, logger)
    elif dataset == 'mmqa':
        logger.warning("MMQA dataset download not implemented yet")
        return False
    elif dataset == 'okvqa':
        logger.warning("OK-VQA dataset download not implemented yet")
        return False
    elif dataset == 'custom':
        # 自定义数据集需要提供URL
        logger.info("Custom dataset - please ensure data is in the raw_data_dir")
        return True
    else:
        logger.error(f"Unknown dataset: {dataset}")
        return False


def process_image(image_input: Union[str, Image.Image, dict], image_dir: Path,
                  image_format: str = 'base64') -> Optional[Dict[str, Any]]:
    """处理图像输入"""
    if image_input is None:
        return None

    try:
        # 转换为PIL图像
        if isinstance(image_input, str):
            if image_input.startswith('data:image'):
                # Base64编码
                header, data = image_input.split(',', 1)
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
            elif image_input.startswith('http'):
                # URL
                import requests
                response = requests.get(image_input)
                image = Image.open(io.BytesIO(response.content))
            else:
                # 文件路径
                image = Image.open(image_input)
        elif isinstance(image_input, dict):
            if 'data' in image_input:
                image_data = base64.b64decode(image_input['data'])
                image = Image.open(io.BytesIO(image_data))
            elif 'path' in image_input:
                image = Image.open(image_input['path'])
            else:
                return None
        else:
            image = image_input

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 根据格式保存/编码图像
        if image_format == 'base64':
            # 转换为base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {
                'format': 'base64',
                'data': f"data:image/png;base64,{img_str}",
                'width': image.width,
                'height': image.height
            }
        elif image_format == 'path':
            # 保存到文件
            import uuid
            filename = f"{uuid.uuid4()}.png"
            filepath = image_dir / filename
            image.save(filepath)
            return {
                'format': 'path',
                'path': str(filepath),
                'width': image.width,
                'height': image.height
            }
        else:  # url格式，保持原样
            if isinstance(image_input, str) and image_input.startswith('http'):
                return {
                    'format': 'url',
                    'url': image_input,
                    'width': image.width,
                    'height': image.height
                }
            else:
                # 回退到base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return {
                    'format': 'base64',
                    'data': f"data:image/png;base64,{img_str}",
                    'width': image.width,
                    'height': image.height
                }

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def analyze_image_with_gpt4v(image: Image.Image, question: str, openai_api_key: str) -> Dict[str, Any]:
    """使用GPT-4V分析图像"""
    if not openai_api_key:
        return {}

    try:
        import openai

        # 将图像转换为base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        client = openai.OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this image in the context of the question: "{question}"

Please provide:
1. A detailed description of what you see
2. Key objects, people, or concepts visible
3. Relevant visual information for answering the question
4. Suggested search terms based on the image content

Format your response as JSON:
{{
    "description": "detailed description",
    "objects": ["object1", "object2", ...],
    "concepts": ["concept1", "concept2", ...],
    "search_terms": ["term1", "term2", ...]
}}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        # 解析响应
        content = response.choices[0].message.content
        try:
            analysis = json.loads(content)
            return analysis
        except json.JSONDecodeError:
            # 如果不是JSON格式，提取关键信息
            return {
                "description": content,
                "objects": [],
                "concepts": [],
                "search_terms": []
            }

    except Exception as e:
        print(f"Error in GPT-4V analysis: {e}")
        return {}


def generate_multimodal_search_queries(question: str, image_analysis: Dict[str, Any],
                                       use_api: bool = False, api_key: str = None) -> List[str]:

    # 使用API生成查询
    try:
        import requests

        prompt = f"""Generate search queries for this multimodal question:

Question: {question}
Image Analysis: {json.dumps(image_analysis, indent=2)}

Generate 5 diverse search queries that combine text and visual information:
1. Text-only query based on the question
2. Object-based query from image analysis
3. Concept-based query combining question and image
4. Specific visual feature query
5. Contextual query

Format as JSON list: ["query1", "query2", ...]"""

        payload = {
            "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "max_tokens": 300,
            "temperature": 0.7
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post("https://api.siliconflow.cn/v1/chat/completions",
                                 json=payload, headers=headers)

        if response.status_code == 200:
            result_data = response.json()
            result = result_data['choices'][0]['message']['content']

            try:
                queries = json.loads(result)
                return queries[:5] if isinstance(queries, list) else []
            except json.JSONDecodeError:
                # 提取JSON部分
                import re
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    queries = json.loads(json_match.group())
                    return queries[:5] if isinstance(queries, list) else []

    except Exception as e:
        print(f"Error generating queries with API: {e}")

    # 回退到简单方法
    return []


def process_multimodal_sample(sample: Dict[str, Any], image_dir: Path, args, logger) -> Optional[Dict[str, Any]]:
    """处理单个多模态样本"""
    try:
        question = sample['question']
        answer = sample['answer']
        image = sample.get('image')

        # 处理图像
        if image:
            image_info = process_image(image, image_dir, args.image_format)
            if not image_info:
                logger.warning(f"Failed to process image for sample {sample.get('id', 'unknown')}")
                return None
        else:
            image_info = None

        # 图像分析
        image_analysis = {}
        if image_info and args.image_analysis:
            if args.openai_api_key and not args.skip_gpt4v:
                # 重新获取PIL图像用于分析
                if image_info['format'] == 'base64':
                    header, data = image_info['data'].split(',', 1)
                    image_data = base64.b64decode(data)
                    pil_image = Image.open(io.BytesIO(image_data))
                elif image_info['format'] == 'path':
                    pil_image = Image.open(image_info['path'])
                else:
                    pil_image = None

                if pil_image:
                    image_analysis = analyze_image_with_gpt4v(pil_image, question, args.openai_api_key)
            else:
                # 简单分析
                if image_info['format'] == 'base64':
                    header, data = image_info['data'].split(',', 1)
                    image_data = base64.b64decode(data)
                    pil_image = Image.open(io.BytesIO(image_data))
                elif image_info['format'] == 'path':
                    pil_image = Image.open(image_info['path'])
                else:
                    pil_image = None

        # 生成搜索查询
        use_api = not args.skip_deepseek and bool(args.deepseek_api_key)
        search_queries = generate_multimodal_search_queries(
            question, image_analysis, use_api, args.deepseek_api_key
        )

        # 创建多模态样本
        processed_sample = {
            'id': sample.get('id', ''),
            'question': question,
            'answer': answer,
            'image': image_info,
            'image_analysis': image_analysis,
            'subquestions': [
                {
                    'sub_question': question,
                    'reasoning': 'Multimodal question requiring both text and visual understanding',
                    'search_queries': search_queries[:3],  # 文本搜索查询
                    'image_search_queries': search_queries[3:] if len(search_queries) > 3 else []  # 图像搜索查询
                }
            ],
            'original_sample': sample
        }

        return processed_sample

    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        return None


def process_multimodal_dataset(raw_data_dir: str, output_dir: str, image_dir: Path,
                               dataset: str, max_train: int, max_dev: int,
                               args, logger):
    """处理多模态数据集"""
    logger.info(f"Processing {dataset} dataset...")

    raw_data_dir = Path(raw_data_dir)

    # 加载原始数据
    if dataset == 'vqa':
        train_file = raw_data_dir / 'vqa' / 'train.json'
        if not train_file.exists():
            raise FileNotFoundError(f"VQA training data not found: {train_file}")

        with open(train_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

    elif dataset == 'custom':
        # 查找自定义数据文件
        data_files = list(raw_data_dir.glob('**/*.json'))
        if not data_files:
            raise FileNotFoundError(f"No JSON files found in {raw_data_dir}")

        raw_data = []
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    raw_data.extend(data)
                else:
                    raw_data.append(data)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 处理训练数据
    logger.info(f"Processing {min(len(raw_data), max_train)} training samples...")
    processed_train = []

    for i, sample in enumerate(raw_data[:max_train]):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{min(len(raw_data), max_train)} samples")

        processed_sample = process_multimodal_sample(sample, image_dir, args, logger)
        if processed_sample:
            processed_train.append(processed_sample)

        # 添加延迟避免API限制
        if args.image_analysis and not args.skip_gpt4v:
            time.sleep(1)

    # 保存训练数据
    train_output = Path(output_dir) / 'train_multimodal_processed.json'
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(processed_train, f, ensure_ascii=False, indent=2)

    logger.info(f"Processed {len(processed_train)} training samples -> {train_output}")

    # 处理验证数据（如果有的话）
    processed_dev = []
    if len(raw_data) > max_train and max_dev > 0:
        logger.info(f"Processing {max_dev} dev samples...")

        for i, sample in enumerate(raw_data[max_train:max_train + max_dev]):
            processed_sample = process_multimodal_sample(sample, image_dir, args, logger)
            if processed_sample:
                processed_sample['id'] = f"dev_{i}"
                processed_dev.append(processed_sample)

        # 保存验证数据
        dev_output = Path(output_dir) / 'dev_multimodal_processed.json'
        with open(dev_output, 'w', encoding='utf-8') as f:
            json.dump(processed_dev, f, ensure_ascii=False, indent=2)

        logger.info(f"Processed {len(processed_dev)} dev samples -> {dev_output}")

    return len(processed_train), len(processed_dev)


def validate_multimodal_data(processed_data_dir: str, logger):
    """验证处理后的多模态数据"""
    logger.info("Validating processed multimodal data...")

    processed_data_dir = Path(processed_data_dir)

    # 检查必需文件
    required_files = ['train_multimodal_processed.json']
    optional_files = ['dev_multimodal_processed.json']

    for filename in required_files:
        filepath = processed_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")

        # 验证数据格式
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Invalid data format in {filepath}: expected list")

        if len(data) == 0:
            raise ValueError(f"Empty data in {filepath}")

        # 验证样本格式
        sample = data[0]
        required_keys = ['id', 'question', 'answer', 'subquestions']
        for key in required_keys:
            if key not in sample:
                raise ValueError(f"Missing key '{key}' in sample from {filepath}")

        # 验证多模态特定字段
        if 'subquestions' in sample:
            subq = sample['subquestions'][0]
            if 'image_search_queries' not in subq:
                logger.warning(f"Missing 'image_search_queries' in subquestion from {filepath}")

        logger.info(f"Validated {filename}: {len(data)} samples")

    # 检查可选文件
    for filename in optional_files:
        filepath = processed_data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Found {filename}: {len(data)} samples")

    logger.info("Multimodal data validation completed successfully!")


def main():
    """主函数"""
    args = parse_arguments()

    # 设置环境
    logger, output_dir, image_dir = setup_environment(args)
    logger.info("Starting multimodal data preparation...")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # 下载数据集
        if not download_datasets(args.raw_data_dir, args.dataset, logger):
            raise Exception(f"Failed to download {args.dataset} dataset")

        # 处理数据集
        train_count, dev_count = process_multimodal_dataset(
            args.raw_data_dir,
            args.output_dir,
            image_dir,
            args.dataset,
            args.max_train_samples,
            args.max_dev_samples,
            args,
            logger
        )

        # 验证处理后的数据
        validate_multimodal_data(args.output_dir, logger)


        # 打印总结
        logger.info("Multimodal data preparation completed successfully!")
        logger.info(f"Training samples: {train_count}")
        logger.info(f"Development samples: {dev_count}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Image directory: {image_dir}")

        print("\nMultimodal Data Preparation Summary:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Training samples: {train_count}")
        print(f"  Development samples: {dev_count}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Image format: {args.image_format}")

        print("\nNext steps:")
        print("1. Review the processed multimodal data files")
        print("2. Update config.yaml with correct data paths")
        print(
            "3. Run multimodal training: python scripts/train_multimodal.py --train-data data/multimodal/processed/train_multimodal_processed.json")

    except Exception as e:
        logger.error(f"Multimodal data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()