# scripts/monitor_training.py
import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_metrics():
    """绘制训练指标"""

    # 读取训练日志
    log_file = './logs/train.log'

    losses = []
    rewards = []
    steps = []

    with open(log_file, 'r') as f:
        for line in f:
            if 'Step' in line and 'Loss:' in line:
                # 解析日志行
                parts = line.strip().split(' - ')[1]
                step = int(parts.split()[1])
                loss = float(parts.split('Loss: ')[1].split(',')[0])
                reward = float(parts.split('Reward: ')[1].split(',')[0])

                steps.append(step)
                losses.append(loss)
                rewards.append(reward)

    # 绘制图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(steps, losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')

    ax2.plot(steps, rewards)
    ax2.set_title('Average Reward')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Reward')

    plt.tight_layout()
    plt.savefig('./results/training_metrics.png')
    plt.show()


if __name__ == "__main__":
    plot_training_metrics()