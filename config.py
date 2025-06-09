import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen2.5-3B-Base"
    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    
@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 3
    learning_rate: float = 7e-7
    value_learning_rate: float = 7e-6
    batch_size: int = 256
    mini_batch_size: int = 64
    micro_batch_size: int = 32
    max_steps: int = 500
    warmup_ratio: float = 0.285
    value_warmup_ratio: float = 0.015
    clip_epsilon: float = 0.2
    kl_coeff: float = 1e-3
    max_search_steps: int = 5
    
@dataclass
class RewardConfig:
    """奖励配置"""
    gamma_key: float = 0.1
    redundancy_threshold: float = 0.8
    
@dataclass
class DataConfig:
    """数据配置"""
    train_data_path: str = "src/data/enhanced_musique_train.json"
    test_data_paths: dict = field(default_factory=lambda: {
        "hotpotqa": "musique/hotpotqa_test.json",
        "2wiki": "musique/2wiki_test.json",
        "musique": "musique/musique_test.json",
        "bamboogle": "musique/bamboogle_test.json"
    }) # type: ignore
    max_train_samples: Optional[int] = None
    
@dataclass
class Config:
    """总配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # 输出和日志
    output_dir: str = "experiments/outputs"
    logging_dir: str = "experiments/logs"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    
    # 硬件配置
    device: str = "cuda"
    num_gpus: int = 1
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """从YAML文件加载配置"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """保存配置到YAML文件"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)