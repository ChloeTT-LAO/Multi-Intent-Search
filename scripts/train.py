#!/usr/bin/env python3
"""
StepSearch主训练脚本
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config import CONFIG
from src.models.step_search_model import create_step_search_model
from src.data.dataset import load_train_dataset
from src.training.steppo_trainer import StePPOTrainer
from src.training.reward_calculator import StepSearchRewardCalculator
from src.search.search_engine import create_search_engine
from src.utils.logging_utils import setup_logger
from src.utils.common import set_random_seed, get_device_info


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train StepSearch model")

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to processed training data')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of training samples')
    parser.add_argument('--eval-data-path', type=str, default=None,
                        help='Path to evaluation data')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Evaluation interval (steps)')
    parser.add_argument('--save-interval', type=int, default=200,
                        help='Save interval (steps)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval (steps)')

    return parser.parse_args()


def setup_training_environment(args):
    """设置训练环境"""
    # 设置随机种子
    set_random_seed(42)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger('train', output_dir / 'train.log')

    # 打印设备信息
    device_info = get_device_info()
    logger.info(f"Training on device: {device_info}")

    return logger, output_dir


def create_data_loader(data_path: str, config: dict, max_samples: int = None):
    """创建数据加载器"""
    # 创建临时tokenizer用于数据集创建
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # 加载数据集
    dataset = load_train_dataset(data_path, tokenizer, config)

    # 限制样本数量
    if max_samples and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=dataset.dataset.collate_fn if hasattr(dataset, 'dataset') else dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    return data_loader


def evaluate_model(trainer, eval_data_path: str, logger):
    """评估模型"""
    if not eval_data_path or not Path(eval_data_path).exists():
        return {}

    try:
        from src.evaluation.evaluator import create_evaluator
        from src.data.dataset import load_eval_dataset

        # 创建评估器
        evaluator = create_evaluator(trainer.model, trainer.search_engine, trainer.config)

        # 加载评估数据
        eval_dataset = load_eval_dataset('musique', eval_data_path)

        # 评估
        result = evaluator.evaluate_dataset(eval_dataset, max_samples=100)

        metrics = result['metrics']
        logger.info(f"Evaluation - EM: {metrics['exact_match']:.3f}, F1: {metrics['f1_score']:.3f}")

        return metrics

    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        return {}


def main():
    """主函数"""
    args = parse_arguments()

    # 设置训练环境
    logger, output_dir = setup_training_environment(args)
    logger.info("Starting StepSearch training...")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # 创建模型
        logger.info("Creating model...")
        model = create_step_search_model(CONFIG, with_value_head=True)
        logger.info(f"Model created: {CONFIG['model']['name']}")

        # 创建搜索引擎
        logger.info("Creating search engine...")
        search_engine = create_search_engine(CONFIG)
        logger.info("Search engine created")

        # 创建奖励计算器
        logger.info("Creating reward calculator...")
        reward_calculator = StepSearchRewardCalculator(CONFIG)

        # 创建训练器
        logger.info("Creating trainer...")
        trainer = StePPOTrainer(model, reward_calculator, search_engine, CONFIG)

        # 从检查点恢复（如果指定）
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            start_epoch = trainer.load_checkpoint(args.resume)

        # 创建数据加载器
        logger.info("Loading training data...")
        train_loader = create_data_loader(args.data_path, CONFIG, args.max_samples)
        logger.info(f"Training data loaded: {len(train_loader)} batches")

        # 训练循环
        logger.info("Starting training loop...")
        global_step = 0
        best_f1 = 0.0

        for epoch in range(start_epoch, CONFIG['training']['num_epochs']):
            logger.info(f"=== Epoch {epoch + 1}/{CONFIG['training']['num_epochs']} ===")

            model.train()
            epoch_stats = {
                'total_loss': [],
                'policy_loss': [],
                'value_loss': [],
                'kl_div': [],
                'avg_reward': [],
                'avg_trajectory_length': []
            }

            for batch_idx, batch in enumerate(train_loader):
                # 训练步骤
                stats = trainer.train_step(batch)

                # 记录统计信息
                for key, value in stats.items():
                    epoch_stats[key].append(value)

                global_step += 1

                # 日志记录
                if global_step % args.log_interval == 0:
                    avg_stats = {k: sum(v[-args.log_interval:]) / len(v[-args.log_interval:])
                                 for k, v in epoch_stats.items() if v}

                    logger.info(f"Step {global_step} - "
                                f"Loss: {avg_stats['total_loss']:.4f}, "
                                f"Reward: {avg_stats['avg_reward']:.4f}, "
                                f"Traj Len: {avg_stats['avg_trajectory_length']:.1f}")

                # 评估
                if args.eval_data_path and global_step % args.eval_interval == 0:
                    model.eval()
                    eval_metrics = evaluate_model(trainer, args.eval_data_path, logger)

                    # 保存最佳模型
                    if eval_metrics and eval_metrics.get('f1_score', 0) > best_f1:
                        best_f1 = eval_metrics['f1_score']
                        best_model_path = output_dir / 'best_model'
                        model.save_model(str(best_model_path))
                        logger.info(f"New best model saved (F1: {best_f1:.3f})")

                    model.train()

                # 保存检查点
                if global_step % args.save_interval == 0:
                    checkpoint_path = output_dir / f'checkpoint_step_{global_step}.pt'
                    trainer.save_checkpoint(str(checkpoint_path), epoch)

                # 检查是否达到最大步数
                if global_step >= CONFIG['training']['max_steps']:
                    logger.info(f"Reached maximum steps: {global_step}")
                    break

            # Epoch结束统计
            avg_epoch_stats = {k: sum(v) / len(v) for k, v in epoch_stats.items() if v}
            logger.info(f"Epoch {epoch + 1} completed - Avg Loss: {avg_epoch_stats['total_loss']:.4f}")

            # 保存epoch检查点
            epoch_checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            trainer.save_checkpoint(str(epoch_checkpoint_path), epoch + 1)

            if global_step >= CONFIG['training']['max_steps']:
                break

        # 训练完成
        logger.info("Training completed!")

        # 保存最终模型
        final_model_path = output_dir / 'final_model'
        model.save_model(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")

        # 最终评估
        if args.eval_data_path:
            logger.info("Running final evaluation...")
            model.eval()
            final_metrics = evaluate_model(trainer, args.eval_data_path, logger)
            logger.info(f"Final evaluation results: {final_metrics}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("Training script completed!")


if __name__ == "__main__":
    main()