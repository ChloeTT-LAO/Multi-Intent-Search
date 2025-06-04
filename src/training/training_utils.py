"""
训练工具函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """经验缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Dict[str, Any]):
        """添加经验"""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """采样经验"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        import random
        return random.sample(list(self.buffer), batch_size)

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class TrainingScheduler:
    """训练调度器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0

        # 调度参数
        self.max_epochs = config.get('max_epochs', 100)
        self.max_steps = config.get('max_steps', 10000)
        self.patience = config.get('patience', 10)
        self.min_improvement = config.get('min_improvement', 1e-4)

    def should_stop(self) -> bool:
        """是否应该停止训练"""
        # 检查最大轮数
        if self.current_epoch >= self.max_epochs:
            return True

        # 检查最大步数
        if self.current_step >= self.max_steps:
            return True

        # 检查早停
        if self.patience_counter >= self.patience:
            return True

        return False

    def update(self, metric: float) -> bool:
        """更新调度器状态"""
        improved = False

        if metric > self.best_metric + self.min_improvement:
            self.best_metric = metric
            self.patience_counter = 0
            improved = True
        else:
            self.patience_counter += 1

        return improved

    def step_epoch(self):
        """步进epoch"""
        self.current_epoch += 1

    def step(self):
        """步进step"""
        self.current_step += 1

    def get_progress(self) -> Dict[str, Any]:
        """获取训练进度"""
        return {
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'progress_ratio': min(1.0, self.current_step / self.max_steps)
        }


class LearningRateScheduler:
    """学习率调度器"""

    def __init__(self, optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
        self.optimizer = optimizer
        self.config = config
        self.scheduler_type = config.get('scheduler_type', 'cosine')

        # 创建调度器
        if self.scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.get('max_steps', 1000),
                eta_min=config.get('min_lr', 1e-8)
            )
        elif self.scheduler_type == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=config.get('end_factor', 0.1),
                total_iters=config.get('max_steps', 1000)
            )
        elif self.scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(
                optimizer,
                step_size=config.get('step_size', 100),
                gamma=config.get('gamma', 0.9)
            )
        else:
            self.scheduler = None

    def step(self):
        """更新学习率"""
        if self.scheduler is not None:
            self.scheduler.step()

    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class GradientClipper:
    """梯度裁剪器"""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.gradient_norms = []

    def clip_gradients(self, model: nn.Module) -> float:
        """裁剪梯度并返回裁剪前的梯度范数"""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            norm_type=self.norm_type
        )

        self.gradient_norms.append(total_norm.item())
        return total_norm.item()

    def get_gradient_statistics(self) -> Dict[str, float]:
        """获取梯度统计信息"""
        if not self.gradient_norms:
            return {}

        return {
            'mean_grad_norm': np.mean(self.gradient_norms),
            'max_grad_norm': np.max(self.gradient_norms),
            'min_grad_norm': np.min(self.gradient_norms),
            'std_grad_norm': np.std(self.gradient_norms)
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.gradient_norms.clear()


class TrainingMetrics:
    """训练指标追踪器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.global_metrics = defaultdict(list)

    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.global_metrics[key].append(value)

    def get_recent_average(self, key: str) -> float:
        """获取最近的平均值"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return np.mean(list(self.metrics[key]))

    def get_global_average(self, key: str) -> float:
        """获取全局平均值"""
        if key not in self.global_metrics or len(self.global_metrics[key]) == 0:
            return 0.0
        return np.mean(self.global_metrics[key])

    def get_trend(self, key: str, window: int = 50) -> str:
        """获取指标趋势"""
        if key not in self.global_metrics or len(self.global_metrics[key]) < window * 2:
            return "insufficient_data"

        recent_data = self.global_metrics[key][-window:]
        prev_data = self.global_metrics[key][-window * 2:-window]

        recent_avg = np.mean(recent_data)
        prev_avg = np.mean(prev_data)

        if recent_avg > prev_avg * 1.05:
            return "improving"
        elif recent_avg < prev_avg * 0.95:
            return "degrading"
        else:
            return "stable"

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {}
        for key in self.metrics.keys():
            summary[key] = {
                'recent_avg': self.get_recent_average(key),
                'global_avg': self.get_global_average(key),
                'trend': self.get_trend(key),
                'count': len(self.global_metrics[key])
            }
        return summary


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save_checkpoint(self,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: Any,
                        epoch: int,
                        step: int,
                        metrics: Dict[str, float],
                        is_best: bool = False) -> str:
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': time.time()
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # 保存路径
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'

        torch.save(checkpoint, checkpoint_path)

        # 管理检查点数量
        if not is_best:
            self.checkpoints.append(checkpoint_path)
            self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        if len(self.checkpoints) > self.max_checkpoints:
            # 按修改时间排序
            self.checkpoints.sort(key=lambda x: x.stat().st_mtime)

            # 删除最旧的检查点
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新检查点"""
        if not self.checkpoints:
            return None

        # 按修改时间排序，返回最新的
        self.checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(self.checkpoints[0])


class TrainingTimer:
    """训练计时器"""

    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        self.step_times = deque(maxlen=100)

    def start_training(self):
        """开始训练计时"""
        self.start_time = time.time()

    def start_epoch(self):
        """开始epoch计时"""
        self.epoch_start_time = time.time()

    def step(self):
        """记录步骤时间"""
        if self.epoch_start_time is not None:
            step_time = time.time() - self.epoch_start_time
            self.step_times.append(step_time)
            self.epoch_start_time = time.time()

    def get_training_time(self) -> float:
        """获取总训练时间"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_average_step_time(self) -> float:
        """获取平均步骤时间"""
        if not self.step_times:
            return 0.0
        return np.mean(list(self.step_times))

    def estimate_remaining_time(self, current_step: int, total_steps: int) -> float:
        """估计剩余时间"""
        if current_step >= total_steps or not self.step_times:
            return 0.0

        avg_step_time = self.get_average_step_time()
        remaining_steps = total_steps - current_step
        return avg_step_time * remaining_steps

    def get_time_summary(self, current_step: int, total_steps: int) -> Dict[str, str]:
        """获取时间摘要"""
        training_time = self.get_training_time()
        avg_step_time = self.get_average_step_time()
        remaining_time = self.estimate_remaining_time(current_step, total_steps)

        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return {
            'total_time': format_time(training_time),
            'avg_step_time': f"{avg_step_time:.2f}s",
            'estimated_remaining': format_time(remaining_time)
        }


class MemoryMonitor:
    """内存监控器"""

    def __init__(self):
        self.memory_usage = []
        self.peak_memory = 0

    def update(self):
        """更新内存使用情况"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            self.memory_usage.append(current_memory)
            self.peak_memory = max(self.peak_memory, current_memory)
        else:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss / 1024 ** 3  # GB
            self.memory_usage.append(current_memory)
            self.peak_memory = max(self.peak_memory, current_memory)

    def get_statistics(self) -> Dict[str, float]:
        """获取内存统计"""
        if not self.memory_usage:
            return {}

        return {
            'current_memory_gb': self.memory_usage[-1] if self.memory_usage else 0.0,
            'peak_memory_gb': self.peak_memory,
            'avg_memory_gb': np.mean(self.memory_usage),
            'memory_trend': 'increasing' if len(self.memory_usage) > 10 and
                                            self.memory_usage[-1] > np.mean(self.memory_usage[-10:]) else 'stable'
        }


class TrainingProfiler:
    """训练性能分析器"""

    def __init__(self):
        self.profiling_data = defaultdict(list)
        self.current_timers = {}

    def start_timer(self, name: str):
        """开始计时"""
        self.current_timers[name] = time.time()

    def end_timer(self, name: str):
        """结束计时"""
        if name in self.current_timers:
            elapsed = time.time() - self.current_timers[name]
            self.profiling_data[name].append(elapsed)
            del self.current_timers[name]
            return elapsed
        return 0.0

    def get_profile_summary(self) -> Dict[str, Dict[str, float]]:
        """获取性能分析摘要"""
        summary = {}
        for name, times in self.profiling_data.items():
            if times:
                summary[name] = {
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'count': len(times)
                }
        return summary


def compute_advantages(rewards: List[float],
                       values: List[float],
                       gamma: float = 0.99,
                       lam: float = 0.95) -> Tuple[List[float], List[float]]:
    """计算GAE优势和回报"""
    advantages = []
    returns = []

    # 计算GAE
    gae = 0
    for i in reversed(range(len(rewards))):
        next_value = values[i + 1] if i + 1 < len(values) else 0
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    # 计算回报
    for i in range(len(rewards)):
        returns.append(advantages[i] + values[i])

    return advantages, returns


def normalize_advantages(advantages: List[float], eps: float = 1e-8) -> List[float]:
    """标准化优势"""
    if len(advantages) <= 1:
        return advantages

    mean_adv = np.mean(advantages)
    std_adv = np.std(advantages)

    if std_adv < eps:
        return [0.0] * len(advantages)

    normalized = [(adv - mean_adv) / (std_adv + eps) for adv in advantages]
    return normalized


def compute_policy_loss(log_probs: torch.Tensor,
                        old_log_probs: torch.Tensor,
                        advantages: torch.Tensor,
                        clip_range: float = 0.2) -> torch.Tensor:
    """计算PPO策略损失"""
    # 计算比率
    ratio = torch.exp(log_probs - old_log_probs)

    # 计算裁剪损失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss


def compute_value_loss(values: torch.Tensor,
                       returns: torch.Tensor,
                       old_values: torch.Tensor = None,
                       clip_range: float = 0.2) -> torch.Tensor:
    """计算价值函数损失"""
    if old_values is not None:
        # 使用裁剪的价值损失
        value_pred_clipped = old_values + torch.clamp(
            values - old_values, -clip_range, clip_range
        )
        value_loss1 = F.mse_loss(values, returns)
        value_loss2 = F.mse_loss(value_pred_clipped, returns)
        value_loss = torch.max(value_loss1, value_loss2)
    else:
        # 标准MSE损失
        value_loss = F.mse_loss(values, returns)

    return value_loss


def save_training_state(state: Dict[str, Any], save_path: str):
    """保存训练状态"""
    torch.save(state, save_path)
    logger.info(f"Training state saved to {save_path}")


def load_training_state(load_path: str) -> Dict[str, Any]:
    """加载训练状态"""
    state = torch.load(load_path, map_location='cpu')
    logger.info(f"Training state loaded from {load_path}")
    return state


class TrainingContext:
    """训练上下文管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scheduler = TrainingScheduler(config)
        self.metrics = TrainingMetrics()
        self.timer = TrainingTimer()
        self.memory_monitor = MemoryMonitor()
        self.profiler = TrainingProfiler()

        # 检查点管理
        if 'checkpoint_dir' in config:
            self.checkpoint_manager = CheckpointManager(
                config['checkpoint_dir'],
                config.get('max_checkpoints', 5)
            )
        else:
            self.checkpoint_manager = None

    def start_training(self):
        """开始训练"""
        self.timer.start_training()
        logger.info("Training started")

    def start_epoch(self):
        """开始epoch"""
        self.timer.start_epoch()
        self.scheduler.step_epoch()

    def step(self, metrics: Dict[str, float]):
        """执行训练步骤"""
        self.timer.step()
        self.scheduler.step()
        self.metrics.update(metrics)
        self.memory_monitor.update()

    def should_stop(self) -> bool:
        """是否应该停止训练"""
        return self.scheduler.should_stop()

    def save_checkpoint(self, model, optimizer, scheduler=None, is_best=False):
        """保存检查点"""
        if self.checkpoint_manager is not None:
            metrics = self.metrics.get_summary()
            return self.checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler,
                self.scheduler.current_epoch,
                self.scheduler.current_step,
                metrics, is_best
            )
        return None

    def get_full_summary(self) -> Dict[str, Any]:
        """获取完整训练摘要"""
        return {
            'progress': self.scheduler.get_progress(),
            'metrics': self.metrics.get_summary(),
            'timing': self.timer.get_time_summary(
                self.scheduler.current_step,
                self.scheduler.max_steps
            ),
            'memory': self.memory_monitor.get_statistics(),
            'profiling': self.profiler.get_profile_summary()
        }