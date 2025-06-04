#!/usr/bin/env python3
"""
数据准备脚本 - 处理原始数据并生成训练用的搜索轨迹
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config import CONFIG
from src.data.data_pipeline import DataPipeline
from src.utils.logging_utils import setup_logger


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Prepare StepSearch training data")

    parser.add_argument('--raw-data-dir', type=str, default='./data/raw',
                        help='Directory containing raw datasets')
    parser.add_argument('--output-dir', type=str, default='./data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--max-train-samples', type=int, default=1000,
                        help='Maximum training samples to process')
    parser.add_argument('--max-dev-samples', type=int, default=200,
                        help='Maximum development samples to process')
    parser.add_argument('--dataset', type=str, default='musique',
                        choices=['musique', 'hotpotqa', '2wiki'],
                        help='Dataset to process')
    parser.add_argument('--openai-api-key', type=str, default=None,
                        help='OpenAI API key for GPT-4o processing')
    parser.add_argument('--skip-gpt4o', action='store_true',
                        help='Skip GPT-4o processing and use simple decomposition')
    parser.add_argument('--build-knowledge-base', action='store_true',
                        help='Build knowledge base from processed data')

    return parser.parse_args()


def setup_environment(args):
    """设置环境"""
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger('prepare_data', output_dir / 'prepare_data.log')

    # 设置OpenAI API key
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
    elif not os.getenv('OPENAI_API_KEY') and not args.skip_gpt4o:
        logger.warning("No OpenAI API key provided. Use --skip-gpt4o for simple processing.")

    return logger, output_dir


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
            dataset = load_dataset("tau/musique", "musique_v1.0")

            # 保存到本地
            musique_dir.mkdir(exist_ok=True)

            for split in ['train', 'validation']:
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
    """简单处理MuSiQue数据（不使用GPT-4o）"""
    logger.info("Processing MuSiQue data with simple decomposition...")

    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    # 读取原始数据
    train_file = raw_data_dir / 'musique' / 'train.json'
    dev_file = raw_data_dir / 'musique' / 'validation.json'

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    # 处理训练数据
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    processed_train = []
    for i, item in enumerate(train_data[:max_train]):
        processed_item = {
            'id': item.get('id', str(i)),
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

        processed_train.append(processed_item)

    # 保存处理后的训练数据
    train_output = output_dir / 'train_processed.json'
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(processed_train, f, ensure_ascii=False, indent=2)

    logger.info(f"Processed {len(processed_train)} training samples -> {train_output}")

    # 处理验证数据
    if dev_file.exists():
        with open(dev_file, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)

        processed_dev = []
        for i, item in enumerate(dev_data[:max_dev]):
            processed_item = {
                'id': item.get('id', f'dev_{i}'),
                'question': item['question'],
                'answer': item['answer'],
                'subquestions': [
                    {
                        'sub_question': item['question'],
                        'reasoning': 'Direct question without decomposition',
                        'search_queries': [
                            item['question'],
                            ' '.join(item['question'].split()[:5]),
                            item['answer'] if len(item['answer']) < 50 else item['answer'][:50]
                        ]
                    }
                ],
                'original_sample': item
            }
            processed_dev.append(processed_item)

        dev_output = output_dir / 'dev_processed.json'
        with open(dev_output, 'w', encoding='utf-8') as f:
            json.dump(processed_dev, f, ensure_ascii=False, indent=2)

        logger.info(f"Processed {len(processed_dev)} dev samples -> {dev_output}")

    return len(processed_train), len(processed_dev) if dev_file.exists() else 0


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
    logger.info("Validating processed data...")

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
            raise ValueError(f"Invalid data format in {filepath}: expected list")

        if len(data) == 0:
            raise ValueError(f"Empty data in {filepath}")

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
    logger.info("Starting data preparation...")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # 下载数据集
        download_datasets(args.raw_data_dir, logger)

        if args.skip_gpt4o or not os.getenv('OPENAI_API_KEY'):
            # 简单处理
            logger.info("Using simple data processing (no GPT-4o)")
            train_count, dev_count = process_musique_simple(
                args.raw_data_dir,
                args.output_dir,
                args.max_train_samples,
                args.max_dev_samples,
                logger
            )
        else:
            # 使用GPT-4o处理
            logger.info("Using GPT-4o for data processing")

            # 更新配置
            config = CONFIG.copy()
            config['data']['output_path'] = args.output_dir

            # 创建数据管道
            pipeline = DataPipeline(config)

            # 运行管道
            result = pipeline.run_pipeline(
                max_samples_train=args.max_train_samples,
                max_samples_dev=args.max_dev_samples
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
        print("1. Review the processed data files")
        print("2. Update config.yaml with correct data paths")
        print("3. Run training: python scripts/train.py --data-path data/processed/train_processed.json")

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()