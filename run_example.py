#!/usr/bin/env python3
"""
StepSearch完整运行示例
演示如何使用StepSearch进行训练、评估和推理
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from src.utils.common import set_random_seed, Timer
from src.utils.logging_utils import setup_logger


def setup_environment():
    """设置运行环境"""
    # 设置随机种子
    set_random_seed(42)

    # 创建必要目录
    directories = [
        'data/raw',
        'data/processed',
        'data/knowledge_base',
        'checkpoints',
        'logs',
        'results',
        'cache',
        'temp'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger('stepsearch_example', 'logs/example.log')
    return logger


def run_data_preparation(logger):
    """运行数据准备步骤"""
    logger.info("=== Step 1: Data Preparation ===")

    try:
        # 检查是否有OpenAI API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            logger.warning("No OpenAI API key found. Using simple data processing.")
            use_gpt4o = False
        else:
            logger.info("OpenAI API key found. Using GPT-4o for data processing.")
            use_gpt4o = True

        # 运行数据准备
        from scripts.prepare_data import main as prepare_main

        # 模拟命令行参数
        sys.argv = [
            'prepare_data.py',
            '--raw-data-dir', './data/raw',
            '--output-dir', './data/processed',
            '--max-train-samples', '100',  # 小数据集用于演示
            '--max-dev-samples', '20',
            '--build-knowledge-base'
        ]

        if not use_gpt4o:
            sys.argv.append('--skip-gpt4o')

        with Timer("Data preparation"):
            prepare_main()

        logger.info("Data preparation completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return False


def run_training(logger):
    """运行训练步骤"""
    logger.info("=== Step 2: Model Training ===")

    try:
        # 检查数据是否存在
        train_data_path = Path('data/processed/train_processed.json')
        if not train_data_path.exists():
            logger.error("Training data not found. Please run data preparation first.")
            return False

        # 运行训练
        from scripts.train import main as train_main

        # 模拟命令行参数
        sys.argv = [
            'train.py',
            '--data-path', str(train_data_path),
            '--output-dir', './checkpoints',
            '--max-samples', '50',  # 更小的样本数用于演示
            '--eval-data-path', 'data/processed/dev_processed.json',
            '--eval-interval', '10',
            '--save-interval', '20',
            '--log-interval', '5'
        ]

        with Timer("Model training"):
            train_main()

        logger.info("Model training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        logger.info("This might be due to hardware limitations or missing dependencies.")
        return False


def run_evaluation(logger):
    """运行评估步骤"""
    logger.info("=== Step 3: Model Evaluation ===")

    try:
        # 检查模型是否存在
        model_path = Path('checkpoints/best_model')
        if not model_path.exists():
            # 尝试使用最终模型
            model_path = Path('checkpoints/final_model')
            if not model_path.exists():
                logger.error("No trained model found. Please run training first.")
                return False

        # 创建示例评估数据
        create_sample_eval_data()

        # 运行评估
        from scripts.evaluate import main as eval_main

        # 模拟命令行参数
        sys.argv = [
            'evaluate.py',
            '--model-path', str(model_path),
            '--datasets', 'musique',  # 只评估一个数据集用于演示
            '--data-dir', './data/eval',
            '--output-dir', './results',
            '--max-samples', '10',  # 小样本评估
            '--save-predictions',
            '--compare-baselines'
        ]

        with Timer("Model evaluation"):
            eval_main()

        logger.info("Model evaluation completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return False


def run_inference_demo(logger):
    """运行推理演示"""
    logger.info("=== Step 4: Inference Demo ===")

    try:
        # 检查模型是否存在
        model_path = Path('checkpoints/best_model')
        if not model_path.exists():
            model_path = Path('checkpoints/final_model')
            if not model_path.exists():
                logger.error("No trained model found. Please run training first.")
                return False

        # 运行推理演示
        from scripts.inference import StepSearchInference

        # 创建推理器
        inferencer = StepSearchInference(str(model_path), CONFIG)

        # 示例问题
        demo_questions = [
            "What is the capital of France?",
            "Where is the Eiffel Tower located?",
            "What is the population of Tokyo?"
        ]

        logger.info("Running inference on demo questions...")

        for i, question in enumerate(demo_questions, 1):
            logger.info(f"\nDemo {i}: {question}")

            try:
                result = inferencer.answer_question(question, verbose=False)

                logger.info(f"Answer: {result['final_answer']}")
                logger.info(f"Search steps: {result['num_search_steps']}")
                logger.info(f"Search queries: {result['search_queries']}")

            except Exception as e:
                logger.warning(f"Inference failed for question {i}: {e}")

        logger.info("Inference demo completed!")
        return True

    except Exception as e:
        logger.error(f"Inference demo failed: {e}")
        return False


def create_sample_eval_data():
    """创建示例评估数据"""
    eval_dir = Path('data/eval')
    eval_dir.mkdir(exist_ok=True)

    # 创建示例MuSiQue评估数据
    sample_data = [
        {
            'id': 'eval_001',
            'question': 'What is the capital of France?',
            'answer': 'Paris'
        },
        {
            'id': 'eval_002',
            'question': 'Where is the Eiffel Tower located?',
            'answer': 'Paris'
        },
        {
            'id': 'eval_003',
            'question': 'What is the largest city in Japan?',
            'answer': 'Tokyo'
        }
    ]

    import json
    with open(eval_dir / 'musique_eval.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)


def print_summary(logger, results):
    """打印运行总结"""
    logger.info("\n" + "=" * 60)
    logger.info("STEPSEARCH EXAMPLE RUN SUMMARY")
    logger.info("=" * 60)

    steps = [
        ("Data Preparation", results.get('data_prep', False)),
        ("Model Training", results.get('training', False)),
        ("Model Evaluation", results.get('evaluation', False)),
        ("Inference Demo", results.get('inference', False))
    ]

    for step_name, success in steps:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{step_name:<20} {status}")

    logger.info("=" * 60)

    if all(result for result in results.values()):
        logger.info("🎉 All steps completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Check ./results/ for evaluation results")
        logger.info("2. Check ./checkpoints/ for trained models")
        logger.info("3. Check ./logs/ for detailed logs")
        logger.info(
            "4. Try interactive inference with: python scripts/inference.py --model-path ./checkpoints/best_model --mode interactive")
    else:
        logger.info("⚠️  Some steps failed. Check the logs for details.")
        logger.info("This is normal for a first run - you may need to:")
        logger.info("1. Install additional dependencies")
        logger.info("2. Set up API keys (OpenAI for data processing)")
        logger.info("3. Ensure sufficient computational resources")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="StepSearch Complete Example")
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training step (for testing)')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation step')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip inference demo')
    parser.add_argument('--quick-mode', action='store_true',
                        help='Run in quick mode with minimal data')

    args = parser.parse_args()

    # 设置环境
    logger = setup_environment()

    logger.info("Starting StepSearch complete example...")
    logger.info(f"Arguments: {vars(args)}")

    # 如果是快速模式，减少数据量
    if args.quick_mode:
        logger.info("Running in quick mode with minimal data")

    # 运行各个步骤
    results = {}

    # 1. 数据准备
    results['data_prep'] = run_data_preparation(logger)

    # 2. 训练（可选跳过）
    if not args.skip_training and results['data_prep']:
        results['training'] = run_training(logger)
    else:
        if args.skip_training:
            logger.info("Skipping training step as requested")
        results['training'] = False

    # 3. 评估（可选跳过）
    if not args.skip_evaluation and results.get('training', False):
        results['evaluation'] = run_evaluation(logger)
    else:
        if args.skip_evaluation:
            logger.info("Skipping evaluation step as requested")
        results['evaluation'] = False

    # 4. 推理演示（可选跳过）
    if not args.skip_inference and results.get('training', False):
        results['inference'] = run_inference_demo(logger)
    else:
        if args.skip_inference:
            logger.info("Skipping inference demo as requested")
        results['inference'] = False

    # 打印总结
    print_summary(logger, results)


if __name__ == "__main__":
    main()