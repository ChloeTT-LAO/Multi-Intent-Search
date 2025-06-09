#!/usr/bin/env python3
"""
数据准备脚本 - 处理原始数据并生成训练用的搜索轨迹
"""

import os
import sys
import argparse
import json
import subprocess
import gdown
import zipfile
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config import CONFIG
from src.data.data_pipeline import DataPipeline
from src.utils.logging_utils import setup_logger


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Prepare StepSearch training musique")

    parser.add_argument('--raw_data_dir', type=str, default='./data/raw',
                        help='Directory containing raw datasets')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for processed musique')
    parser.add_argument('--max_train_samples', type=int, default=1,
                        help='Maximum training samples to process')
    parser.add_argument('--max_dev_samples', type=int, default=1,
                        help='Maximum development samples to process')
    parser.add_argument('--dataset', type=str, default='musique',
                        choices=['musique', 'hotpotqa', '2wiki'],
                        help='Dataset to process')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key for GPT-4o processing')
    parser.add_argument('--deepseek_api_key', type=str, default='sk-anbsoppiznxtiuhzxdibxuvpnhsxoabbsderulnnzsfduyrq',
                        help='DeepSeek API key for DeepSeek processing')
    parser.add_argument('--skip_gpt4o', action='store_true',
                        help='Skip GPT-4o processing and use simple decomposition')
    parser.add_argument('--skip_deepseek', action='store_true',
                        help='Skip DeepSeek processing')
    parser.add_argument('--build_knowledge_base', action='store_true',
                        help='Build knowledge base from processed musique')

    return parser.parse_args()


def setup_environment(args):
    """设置环境"""
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger('prepare_data', output_dir / 'prepare_data.log')

    # 设置API keys
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
        logger.info("OpenAI API key set from command line argument")
    elif os.getenv('OPENAI_API_KEY'):
        logger.info("Using OpenAI API key from environment variable")
    elif not args.skip_gpt4o:
        logger.warning("No OpenAI API key provided. Use --skip-gpt4o for simple processing.")

    if args.deepseek_api_key:
        os.environ['DEEPSEEK_API_KEY'] = args.deepseek_api_key
        logger.info("DeepSeek API key set from command line argument")
    elif os.getenv('DEEPSEEK_API_KEY'):
        logger.info("Using DeepSeek API key from environment variable")
    elif not args.skip_deepseek:
        logger.warning("No DeepSeek API key provided. Use --skip-deepseek to disable DeepSeek processing.")

    return logger, output_dir


def download_musique_from_gdrive(raw_data_dir: str, logger):
    """从Google Drive下载MuSiQue数据集"""

    try:
        # 确保gdown已安装
        try:
            import gdown
        except ImportError:
            logger.info("Installing gdown...")
            subprocess.check_call(["pip", "install", "gdown"])
            import gdown

        # 设置文件路径
        raw_data_dir = Path(raw_data_dir)
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        zip_name = "musique_v1.0.zip"
        zip_path = raw_data_dir / zip_name

        # Google Drive文件ID
        file_id = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"

        # 检查是否已经下载并解压
        extracted_dir = raw_data_dir / "musique_v1.0"
        if extracted_dir.exists():
            logger.info("Dataset already downloaded and extracted")
            return True

        # 下载文件
        logger.info(f"Downloading {zip_name} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(zip_path), quiet=False)

        # 解压文件
        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)

        # 删除zip文件
        zip_path.unlink()
        logger.info("Removed zip file")

        # 删除MacOS相关文件（如果存在）
        macos_dir = raw_data_dir / "__MACOSX"
        if macos_dir.exists():
            shutil.rmtree(macos_dir)
            logger.info("Removed __MACOSX directory")

        return True

    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
        return False


def download_datasets(raw_data_dir: str, logger):
    """下载数据集"""
    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # 检查并下载MuSiQue数据集
    musique_dir = raw_data_dir / 'musique'
    if not musique_dir.exists():
        logger.info("Downloading MuSiQue dataset...")

        try:
            from datasets import load_dataset

            # 方法1: 尝试使用Hugging Face Hub
            dataset_names = ["tau/musique", "musique", "microsoft/musique"]
            dataset = None

            for name in dataset_names:
                try:
                    logger.info(f"Trying to load dataset: {name}")
                    dataset = load_dataset(name, "musique_v1.0")
                    logger.info(f"Successfully loaded from Hub: {name}")
                    break
                except:
                    try:
                        dataset = load_dataset(name)
                        logger.info(f"Successfully loaded from Hub: {name}")
                        break
                    except:
                        continue

            # 方法2: 如果Hub失败，从Google Drive下载
            if dataset is None:
                logger.info("Attempting download from Google Drive...")
                if download_musique_from_gdrive(raw_data_dir, logger):
                    logger.info("Successfully downloaded from Google Drive")
                    return  # 成功下载，直接返回
                else:
                    raise Exception("Both Hub and Google Drive download failed")

            # 如果从Hub成功加载，保存到本地
            musique_dir.mkdir(exist_ok=True)

            for split in ['train', 'validation']:
                if split in dataset:
                    output_file = musique_dir / f'{split}.json'
                    data = []
                    for item in dataset[split]:
                        data.append(item)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    logger.info(f"Saved {len(data)} samples to {output_file}")

        except Exception as e:
            logger.error(f"Failed to download MuSiQue: {e}")
            raise
    else:
        logger.info("MuSiQue dataset already exists")


def process_musique_simple(raw_data_dir: str, output_dir: str,
                           max_train: int, max_dev: int, logger):
    """简单处理MuSiQue数据（不使用GPT-4o/DeepSeek）"""
    logger.info("Processing MuSiQue musique with simple decomposition...")

    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    # 检查是否有从Google Drive下载的数据
    gdrive_extracted_dir = raw_data_dir / "musique_v1.0"
    if gdrive_extracted_dir.exists():
        logger.info("Using musique downloaded from Google Drive")
        # 寻找数据文件
        data_files = {}
        for root, dirs, files in os.walk(gdrive_extracted_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    if 'train' in file:
                        data_files['train'] = file_path
                    elif 'dev' in file:
                        data_files['validation'] = file_path

        # 处理JSONL文件
        processed_train = []
        processed_dev = []

        if 'train' in data_files:
            with open(data_files['train'], 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_train:
                        break
                    if line.strip():
                        item = json.loads(line)
                        processed_item = create_simple_processed_item(item, i)
                        processed_train.append(processed_item)

        if 'validation' in data_files:
            with open(data_files['validation'], 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_dev:
                        break
                    if line.strip():
                        item = json.loads(line)
                        processed_item = create_simple_processed_item(item, f'dev_{i}')
                        processed_dev.append(processed_item)

    else:
        # 使用原来的逻辑处理JSON文件
        train_file = raw_data_dir / 'musique' / 'musique_full_v1.0_train.jsonl'
        dev_file = raw_data_dir / 'musique' / 'musique_full_v1.0_dev.jsonl'

        if not train_file.exists():
            raise FileNotFoundError(f"Training musique not found: {train_file}")

        # 处理训练数据
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        processed_train = []
        for i, item in enumerate(train_data[:max_train]):
            processed_item = create_simple_processed_item(item, i)
            processed_train.append(processed_item)

        # 处理验证数据
        processed_dev = []
        if dev_file.exists():
            with open(dev_file, 'r', encoding='utf-8') as f:
                dev_data = json.load(f)

            for i, item in enumerate(dev_data[:max_dev]):
                processed_item = create_simple_processed_item(item, f'dev_{i}')
                processed_dev.append(processed_item)

    # 保存处理后的数据
    train_output = output_dir / 'train_processed.json'
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(processed_train, f, ensure_ascii=False, indent=2)
    logger.info(f"Processed {len(processed_train)} training samples -> {train_output}")

    if processed_dev:
        dev_output = output_dir / 'dev_processed.json'
        with open(dev_output, 'w', encoding='utf-8') as f:
            json.dump(processed_dev, f, ensure_ascii=False, indent=2)
        logger.info(f"Processed {len(processed_dev)} dev samples -> {dev_output}")

    return len(processed_train), len(processed_dev)


def create_simple_processed_item(item: dict, item_id):
    """创建简单处理的项目"""
    processed_item = {
        'id': item.get('id', str(item_id)),
        'question': item['question'],
        'answer': item['answer'],
        'subquestions': [
            {
                'sub_question': item['question'],  # 简化：使用原问题
                'reasoning': 'Direct question without decomposition',
                'search_queries': [
                    item['question'],
                    # 简单的查询变体
                    ' '.join(item['question'].split()[:5]),  # 前5个词
                    item['answer'] if len(item['answer']) < 50 else item['answer'][:50]  # 答案作为查询
                ]
            }
        ],
        'original_sample': item
    }

    # 如果有decomposition信息，使用它
    if 'decomposition' in item and item['decomposition']:
        subquestions = []
        for step in item['decomposition']:
            subq = {
                'sub_question': step['question'],
                'reasoning': f"Step {step['id']}: {step['question']}",
                'search_queries': [
                    step['question'],
                    step.get('answer', '')[:50] if step.get('answer') else ''
                ]
            }
            subquestions.append(subq)

        if subquestions:
            processed_item['subquestions'] = subquestions

    return processed_item


def build_knowledge_base(processed_data_dir: str, logger):
    """从处理的数据构建知识库"""
    logger.info("Building knowledge base...")

    processed_data_dir = Path(processed_data_dir)
    knowledge_base = []

    # 从训练数据中提取知识
    train_file = processed_data_dir / 'train_processed.json'
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        for item in train_data:
            # 添加问答对作为知识
            qa_text = f"Question: {item['question']} Answer: {item['answer']}"
            knowledge_base.append(qa_text)

            # 添加子问题作为知识
            for subq in item.get('subquestions', []):
                knowledge_base.append(subq['sub_question'])

    # 添加一些通用知识
    general_knowledge = [
        "Beijing is the capital of China and has a population of over 21 million people.",
        "Paris is the capital of France and is famous for the Eiffel Tower.",
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world.",
        "London is the capital of the United Kingdom and England.",
        "New York City is the most populous city in the United States.",
        "The Great Wall of China is a fortification built across northern China.",
        "The Amazon River is the longest river in South America.",
        "Mount Everest is the highest mountain in the world.",
        "The Pacific Ocean is the largest ocean on Earth."
    ]

    knowledge_base.extend(general_knowledge)

    # 保存知识库
    kb_dir = processed_data_dir.parent / 'knowledge_base'
    kb_dir.mkdir(exist_ok=True)

    kb_file = kb_dir / 'knowledge_base.json'
    with open(kb_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    logger.info(f"Knowledge base created with {len(knowledge_base)} entries -> {kb_file}")

    return kb_file


def validate_processed_data(processed_data_dir: str, logger):
    """验证处理后的数据"""
    logger.info("Validating processed musique...")

    processed_data_dir = Path(processed_data_dir)

    # 检查必需文件
    required_files = ['train_processed.json']
    optional_files = ['dev_processed.json']

    for filename in required_files:
        filepath = processed_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")

        # 验证数据格式
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Invalid musique format in {filepath}: expected list")

        if len(data) == 0:
            raise ValueError(f"Empty musique in {filepath}")

        # 验证样本格式
        sample = data[0]
        required_keys = ['id', 'question', 'answer', 'subquestions']
        for key in required_keys:
            if key not in sample:
                raise ValueError(f"Missing key '{key}' in sample from {filepath}")

        logger.info(f"Validated {filename}: {len(data)} samples")

    # 检查可选文件
    for filename in optional_files:
        filepath = processed_data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Found {filename}: {len(data)} samples")

    logger.info("Data validation completed successfully!")


def main():
    """主函数"""
    args = parse_arguments()

    # 设置环境
    logger, output_dir = setup_environment(args)
    logger.info("Starting musique preparation...")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # 下载数据集
        download_datasets(args.raw_data_dir, logger)

        # 检查API可用性
        has_openai = bool(os.getenv('OPENAI_API_KEY')) and not args.skip_gpt4o
        has_deepseek = bool(os.getenv('DEEPSEEK_API_KEY')) and not args.skip_deepseek

        if not has_openai and not has_deepseek:
            # 简单处理
            logger.info("Using simple musique processing (no API models)")
            train_count, dev_count = process_musique_simple(
                args.raw_data_dir,
                args.output_dir,
                args.max_train_samples,
                args.max_dev_samples,
                logger
            )
        else:
            # 使用API模型处理
            if has_deepseek:
                logger.info("Using DeepSeek for musique processing")
            elif has_openai:
                logger.info("Using GPT-4o for musique processing")

            # 更新配置
            config = CONFIG.copy()
            config['data']['output_path'] = args.output_dir

            # 创建数据管道
            pipeline = DataPipeline(config)

            # 运行管道
            result = pipeline.run_pipeline(
                max_samples_train=args.max_train_samples,
                max_samples_dev=args.max_dev_samples,
                raw_data_dir=args.raw_data_dir
            )

            train_count = len(result['train_data'])
            dev_count = len(result['dev_data'])

        # 验证处理后的数据
        validate_processed_data(args.output_dir, logger)

        # 构建知识库
        if args.build_knowledge_base:
            kb_file = build_knowledge_base(args.output_dir, logger)

        # 打印总结
        logger.info("Data preparation completed successfully!")
        logger.info(f"Training samples: {train_count}")
        logger.info(f"Development samples: {dev_count}")
        logger.info(f"Output directory: {args.output_dir}")

        print("\nData Preparation Summary:")
        print(f"  Training samples: {train_count}")
        print(f"  Development samples: {dev_count}")
        print(f"  Output directory: {args.output_dir}")
        if args.build_knowledge_base:
            print(f"  Knowledge base: {kb_file}")

        print("\nNext steps:")
        print("1. Review the processed musique files")
        print("2. Update config.yaml with correct musique paths")
        print("3. Run training: python scripts/train.py --musique-path musique/processed/train_processed.json")

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()