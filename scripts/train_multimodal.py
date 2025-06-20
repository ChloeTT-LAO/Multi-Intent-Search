# scripts/train_multimodal.py
# !/usr/bin/env python3
import os
import sys
import argparse
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import CONFIG
from src.multimodal.multimodal_model import create_multimodal_step_search_model
from src.multimodal.multimodal_search import create_multimodal_search_engine
from src.multimodal.multimodal_trainer import MultimodalStePPOTrainer
from src.multimodal.multimodal_reward import MultimodalRewardCalculator
from src.multimodal.multimodal_data import load_multimodal_train_dataset
from src.utils.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal StepSearch")
    parser.add_argument('--config', default='config/multimodal_config.yaml')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--output-dir', default='./checkpoints/multimodal')
    parser.add_argument('--resume', default=None)

    args = parser.parse_args()

    # 设置日志
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('multimodal_train', output_dir / 'train.log')

    logger.info("Starting multimodal StepSearch training...")

    # 创建多模态模型
    logger.info("Creating multimodal model...")
    model = create_multimodal_step_search_model(CONFIG, with_value_head=True)

    # 创建多模态搜索引擎
    logger.info("Creating multimodal search engine...")
    search_engine = create_multimodal_search_engine(CONFIG)

    # 创建多模态奖励计算器
    logger.info("Creating multimodal reward calculator...")
    reward_calculator = MultimodalRewardCalculator(CONFIG)

    # 创建训练器
    logger.info("Creating multimodal trainer...")
    trainer = MultimodalStePPOTrainer(model, reward_calculator, search_engine, CONFIG)

    # 加载训练数据
    logger.info("Loading training data...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(CONFIG['model']['name'])

    train_dataset = load_multimodal_train_dataset(args.train_data, processor, CONFIG)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    logger.info(f"Training data loaded: {len(train_loader)} batches")

    # 训练循环
    logger.info("Starting training loop...")
    global_step = 0

    for epoch in range(CONFIG['training']['num_epochs']):
        logger.info(f"=== Epoch {epoch + 1} ===")

        for batch_idx, batch in enumerate(train_loader):
            # 训练步骤
            stats = trainer.train_step(batch)
            global_step += 1

            # 记录统计信息
            if global_step % 10 == 0:
                logger.info(
                    f"Step {global_step} - "
                    f"Loss: {stats['total_loss']:.4f}, "
                    f"Reward: {stats['avg_reward']:.4f}, "
                    f"Traj Len: {stats['avg_trajectory_length']:.1f}, "
                    f"IMG Steps: {stats.get('avg_image_analysis_steps', 0):.1f}"
                )

            # 保存检查点
            if global_step % 100 == 0:
                checkpoint_path = output_dir / f'checkpoint_step_{global_step}.pt'
                trainer.save_checkpoint(str(checkpoint_path), epoch)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

            if global_step >= CONFIG['training']['max_steps']:
                break

        if global_step >= CONFIG['training']['max_steps']:
            break

    # 保存最终模型
    final_model_path = output_dir / 'final_model'
    model.save_model(str(final_model_path))
    logger.info(f"Training completed! Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()