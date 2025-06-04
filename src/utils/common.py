"""
通用工具函数
"""

import os
import random
import numpy as np
import torch
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time
from contextlib import contextmanager


def set_random_seed(seed: int = 42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保CUDNN的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device_info() -> str:
    """获取设备信息"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory = torch.cuda.get_device_properties(current_device).total_memory

        return f"CUDA {current_device} ({device_name}, {memory // 1024 ** 3}GB, {device_count} devices)"
    else:
        return "CPU"


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """获取模型大小信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def format_number(num: Union[int, float]) -> str:
    """格式化数字显示（如1.2M, 3.4B等）"""
    if num >= 1e9:
        return f"{num / 1e9:.1f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    else:
        return str(num)


def create_directories(paths: List[Union[str, Path]]):
    """创建多个目录"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: Union[str, Path]) -> Any:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, file_path: Union[str, Path]):
    """保存pickle文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """加载pickle文件"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class Timer:
    """计时器类"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self

    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        return self

    def elapsed(self) -> float:
        """获取经过时间"""
        if self.start_time is None:
            return 0.0

        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@contextmanager
def timer(description: str = "Operation"):
    """计时上下文管理器"""
    start_time = time.time()
    print(f"{description} started...")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"{description} completed in {elapsed:.2f} seconds")


class MovingAverage:
    """移动平均计算器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []

    def update(self, value: float):
        """更新值"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def average(self) -> float:
        """获取平均值"""
        return sum(self.values) / len(self.values) if self.values else 0.0

    def reset(self):
        """重置"""
        self.values = []


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if self.config_path.suffix == '.json':
            return load_json(self.config_path)
        elif self.config_path.suffix in ['.yml', '.yaml']:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")

    def save_config(self, config: Dict[str, Any] = None):
        """保存配置"""
        config = config or self.config

        if self.config_path.suffix == '.json':
            save_json(config, self.config_path)
        elif self.config_path.suffix in ['.yml', '.yaml']:
            import yaml
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点分隔的嵌套键）"""
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """设置配置值（支持点分隔的嵌套键）"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value


def check_file_exists(file_path: Union[str, Path], raise_error: bool = True) -> bool:
    """检查文件是否存在"""
    exists = Path(file_path).exists()
    if not exists and raise_error:
        raise FileNotFoundError(f"File not found: {file_path}")
    return exists


def get_file_size(file_path: Union[str, Path]) -> int:
    """获取文件大小（字节）"""
    return Path(file_path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / 1024 ** 3:.1f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """确保目录存在，如果不存在则创建"""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def cleanup_old_files(directory: Union[str, Path], pattern: str = "*",
                      keep_last: int = 5, dry_run: bool = False) -> List[Path]:
    """清理旧文件，保留最新的几个"""
    directory = Path(directory)
    files = list(directory.glob(pattern))

    # 按修改时间排序
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # 要删除的文件
    files_to_delete = files[keep_last:]

    if not dry_run:
        for file_path in files_to_delete:
            file_path.unlink()

    return files_to_delete


class MemoryTracker:
    """内存使用跟踪器"""

    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0

    def start(self):
        """开始跟踪"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        return self

    def current_usage(self) -> Dict[str, float]:
        """获取当前内存使用"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()

            return {
                'current_mb': current / 1024 ** 2,
                'peak_mb': peak / 1024 ** 2,
                'allocated_mb': (current - self.start_memory) / 1024 ** 2
            }
        else:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'current_mb': memory_info.rss / 1024 ** 2,
                'peak_mb': memory_info.rss / 1024 ** 2,
                'allocated_mb': 0
            }

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        usage = self.current_usage()
        print(f"Memory usage - Current: {usage['current_mb']:.1f}MB, "
              f"Peak: {usage['peak_mb']:.1f}MB")


def batch_iterator(items: List[Any], batch_size: int):
    """批处理迭代器"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """展平嵌套字典"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """还原展平的字典"""
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result