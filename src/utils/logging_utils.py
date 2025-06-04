"""
日志工具
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logger(name: str, log_file: Optional[Union[str, Path]] = None,
                 level: int = logging.INFO, format_string: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        format_string: 自定义格式字符串

    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的处理器
    logger.handlers.clear()

    # 设置格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果指定了文件路径）
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir: Union[str, Path], experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建训练日志
        self.train_logger = setup_logger(
            f"{experiment_name}_train",
            self.log_dir / f"{experiment_name}_train_{timestamp}.log"
        )

        # 创建评估日志
        self.eval_logger = setup_logger(
            f"{experiment_name}_eval",
            self.log_dir / f"{experiment_name}_eval_{timestamp}.log"
        )

        # 统计信息
        self.stats = {
            'train_steps': 0,
            'eval_steps': 0,
            'best_metrics': {}
        }

    def log_train_step(self, step: int, metrics: dict, learning_rate: float = None):
        """记录训练步骤"""
        self.stats['train_steps'] += 1

        log_msg = f"Train Step {step}"
        if learning_rate:
            log_msg += f" | LR: {learning_rate:.6f}"

        for key, value in metrics.items():
            log_msg += f" | {key}: {value:.4f}"

        self.train_logger.info(log_msg)

    def log_eval_step(self, step: int, metrics: dict, dataset: str = "eval"):
        """记录评估步骤"""
        self.stats['eval_steps'] += 1

        log_msg = f"Eval Step {step} ({dataset})"
        for key, value in metrics.items():
            log_msg += f" | {key}: {value:.4f}"

        self.eval_logger.info(log_msg)

        # 更新最佳指标
        for key, value in metrics.items():
            best_key = f"best_{key}"
            if best_key not in self.stats['best_metrics'] or value > self.stats['best_metrics'][best_key]:
                self.stats['best_metrics'][best_key] = value
                self.eval_logger.info(f"New best {key}: {value:.4f}")

    def log_epoch_summary(self, epoch: int, train_metrics: dict, eval_metrics: dict = None):
        """记录epoch总结"""
        log_msg = f"Epoch {epoch} Summary - Train:"
        for key, value in train_metrics.items():
            log_msg += f" {key}={value:.4f}"

        if eval_metrics:
            log_msg += " | Eval:"
            for key, value in eval_metrics.items():
                log_msg += f" {key}={value:.4f}"

        self.train_logger.info(log_msg)

    def log_model_info(self, model_info: dict):
        """记录模型信息"""
        self.train_logger.info("Model Information:")
        for key, value in model_info.items():
            self.train_logger.info(f"  {key}: {value}")

    def log_config(self, config: dict):
        """记录配置信息"""
        self.train_logger.info("Configuration:")
        for key, value in config.items():
            self.train_logger.info(f"  {key}: {value}")


class MetricsLogger:
    """指标记录器"""

    def __init__(self, log_file: Union[str, Path]):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []

    def log_metrics(self, step: int, metrics: dict, prefix: str = ""):
        """记录指标"""
        timestamp = datetime.now().isoformat()

        entry = {
            'timestamp': timestamp,
            'step': step,
            'prefix': prefix
        }
        entry.update(metrics)

        self.metrics_history.append(entry)

        # 写入文件
        import json
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)

    def get_best_metrics(self, metric_name: str, mode: str = 'max'):
        """获取最佳指标"""
        if not self.metrics_history:
            return None

        values = [entry.get(metric_name) for entry in self.metrics_history if metric_name in entry]
        if not values:
            return None

        if mode == 'max':
            best_value = max(values)
        else:
            best_value = min(values)

        # 找到对应的entry
        for entry in self.metrics_history:
            if entry.get(metric_name) == best_value:
                return entry

        return None

    def get_latest_metrics(self):
        """获取最新指标"""
        return self.metrics_history[-1] if self.metrics_history else None


def setup_experiment_logging(experiment_dir: Union[str, Path],
                             experiment_name: str) -> TrainingLogger:
    """设置实验日志"""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    return TrainingLogger(experiment_dir, experiment_name)


class ProgressLogger:
    """进度日志记录器"""

    def __init__(self, total_steps: int, log_interval: int = 100):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.start_time = None

        self.logger = setup_logger("progress")

    def start(self):
        """开始进度记录"""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting progress tracking for {self.total_steps} steps")

    def update(self, step: int, metrics: dict = None):
        """更新进度"""
        self.current_step = step

        if step % self.log_interval == 0 or step == self.total_steps:
            progress_pct = (step / self.total_steps) * 100

            log_msg = f"Progress: {step}/{self.total_steps} ({progress_pct:.1f}%)"

            if self.start_time:
                import time
                elapsed = time.time() - self.start_time
                if step > 0:
                    eta = (elapsed / step) * (self.total_steps - step)
                    log_msg += f" | ETA: {eta / 60:.1f}min"

            if metrics:
                for key, value in metrics.items():
                    log_msg += f" | {key}: {value:.4f}"

            self.logger.info(log_msg)

    def finish(self):
        """完成进度记录"""
        if self.start_time:
            import time
            total_time = time.time() - self.start_time
            self.logger.info(f"Completed {self.total_steps} steps in {total_time / 60:.1f} minutes")


class RichLogger:
    """富文本日志记录器（使用rich库）"""

    def __init__(self, name: str = "rich_logger"):
        try:
            from rich.console import Console
            from rich.logging import RichHandler
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

            self.console = Console()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )

            # 设置rich日志处理器
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(console=self.console)]
            )

            self.logger = logging.getLogger(name)
            self.rich_available = True

        except ImportError:
            # 如果rich不可用，回退到标准日志
            self.logger = setup_logger(name)
            self.rich_available = False

    def info(self, message: str):
        """信息日志"""
        self.logger.info(message)

    def warning(self, message: str):
        """警告日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """错误日志"""
        self.logger.error(message)

    def print_table(self, data: list, headers: list, title: str = None):
        """打印表格"""
        if self.rich_available:
            from rich.table import Table

            table = Table(title=title)

            for header in headers:
                table.add_column(header)

            for row in data:
                table.add_row(*[str(cell) for cell in row])

            self.console.print(table)
        else:
            # 简单表格打印
            if title:
                print(f"\n{title}")
            print("-" * 60)
            print(" | ".join(headers))
            print("-" * 60)
            for row in data:
                print(" | ".join(str(cell) for cell in row))
            print("-" * 60)

    def create_progress_bar(self, description: str, total: int):
        """创建进度条"""
        if self.rich_available:
            return self.progress.add_task(description, total=total)
        else:
            return None

    def update_progress(self, task_id, advance: int = 1):
        """更新进度条"""
        if self.rich_available and task_id is not None:
            self.progress.update(task_id, advance=advance)