import yaml
import os
from pathlib import Path


def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# 全局配置对象
CONFIG = load_config()