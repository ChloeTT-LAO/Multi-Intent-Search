"""
价值函数模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import AutoModel, AutoTokenizer

class ValueFunction(nn.Module):
    """基础价值函数"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """前向传播"""
        raise NotImplementedError

    def estimate_value(self, *args, **kwargs) -> float:
        """估计状态价值"""
        raise NotImplementedError

class LinearValueFunction(ValueFunction):
    """线性价值函数"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config.get('input_dim', 768)
        self.hidden_dim = config.get('hidden_dim', 256)

        # 简单的线性层
        self.value_head = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.value_head(state_features)

    def estimate_value(self, state_features: torch.Tensor) -> float:
        """估计状态价值"""
        with torch.no_grad():
            value = self.forward(state_features)
            return value.item()

class TransformerValueFunction(ValueFunction):
    """基于Transformer的价值函数"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_name = config.get('value_model_name', 'distilbert-base-uncased')
        self.max_length = config.get('max_length', 512)

        # 加载预训练模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.backbone = AutoModel.from_pretrained(self.model_name)

            # 冻结backbone参数（可选）
            if config.get('freeze_backbone', False):
                for param in self.backbone.parameters():
                    param.requires_grad = False

            # 价值头
            hidden_size = self.backbone.config.hidden_size
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)
            )

        except Exception as e:
            print(f"Failed to load transformer model: {e}")
            # 回退到简单模型
            self.tokenizer = None
            self.backbone = None
            self.simple_value = LinearValueFunction(config)

    def encode_state(self, state_text: str) -> torch.Tensor:
        """编码状态文本"""
        if self.tokenizer is None or self.backbone is None:
            # 使用简单编码
            return torch.randn(1, self.config.get('input_dim', 768))

        inputs = self.tokenizer(
            state_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True
        )

        outputs = self.backbone(**inputs)
        # 使用[CLS] token的表示或平均池化
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state.mean(dim=1)

    def forward(self, state_text: Union[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        if self.backbone is None:
            # 使用简单模型
            if isinstance(state_text, str):
                features = torch.randn(1, self.config.get('input_dim', 768))
            else:
                features = state_text
            return self.simple_value(features)

        if isinstance(state_text, str):
            state_features = self.encode_state(state_text)
        else:
            state_features = state_text

        return self.value_head(state_features)

    def estimate_value(self, state_text: str) -> float:
        """估计状态价值"""
        with torch.no_grad():
            value = self.forward(state_text)
            return value.item()

class ContextAwareValueFunction(ValueFunction):
    """上下文感知的价值函数"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config.get('state_dim', 768)
        self.context_dim = config.get('context_dim', 256)
        self.hidden_dim = config.get('hidden_dim', 512)

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
               state_features: torch.Tensor,
               context_features: torch.Tensor = None) -> torch.Tensor:
        """前向传播"""
        # 编码状态
        state_encoded = self.state_encoder(state_features)

        if context_features is not None:
            # 编码上下文
            context_encoded = self.context_encoder(context_features)

            # 应用注意力机制
            attended_state, _ = self.attention(
                state_encoded.unsqueeze(0),  # 添加序列维度
                context_encoded.unsqueeze(0),
                context_encoded.unsqueeze(0)
            )

            final_features = attended_state.squeeze(0)
        else:
            final_features = state_encoded

        return self.value_head(final_features)

    def estimate_value(self,
                      state_features: torch.Tensor,
                      context_features: torch.Tensor = None) -> float:
        """估计状态价值"""
        with torch.no_grad():
            value = self.forward(state_features, context_features)
            return value.item()

class HierarchicalValueFunction(ValueFunction):
    """分层价值函数"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config.get('input_dim', 768)
        self.num_levels = config.get('num_levels', 3)
        self.hidden_dim = config.get('hidden_dim', 256)

        # 为每个层级创建价值函数
        self.level_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, 1)
            ) for _ in range(self.num_levels)
        ])

        # 级联权重
        self.level_weights = nn.Parameter(torch.ones(self.num_levels) / self.num_levels)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for level_func in self.level_functions:
            for module in level_func.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, state_features: torch.Tensor, level: int = -1) -> torch.Tensor:
        """前向传播"""
        if level >= 0 and level < self.num_levels:
            # 使用特定层级的价值函数
            return self.level_functions[level](state_features)
        else:
            # 使用所有层级的加权组合
            level_values = []
            for level_func in self.level_functions:
                value = level_func(state_features)
                level_values.append(value)

            level_values = torch.stack(level_values, dim=-1)  # [batch, 1, num_levels]
            weights = F.softmax(self.level_weights, dim=0)

            weighted_value = (level_values * weights).sum(dim=-1)
            return weighted_value

    def estimate_value(self, state_features: torch.Tensor, level: int = -1) -> float:
        """估计状态价值"""
        with torch.no_grad():
            value = self.forward(state_features, level)
            return value.item()

    def get_level_values(self, state_features: torch.Tensor) -> List[float]:
        """获取所有层级的价值估计"""
        with torch.no_grad():
            values = []
            for level_func in self.level_functions:
                value = level_func(state_features)
                values.append(value.item())
            return values

class EnsembleValueFunction(ValueFunction):
    """集成价值函数"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_models = config.get('num_models', 3)
        self.model_type = config.get('base_model_type', 'linear')

        # 创建多个基础模型
        self.models = nn.ModuleList()
        for i in range(self.num_models):
            if self.model_type == 'linear':
                model = LinearValueFunction(config)
            elif self.model_type == 'transformer':
                model = TransformerValueFunction(config)
            else:
                model = LinearValueFunction(config)

            self.models.append(model)

        # 集成权重
        self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """前向传播"""
        model_outputs = []

        for model in self.models:
            output = model(*args, **kwargs)
            model_outputs.append(output)

        # 加权平均
        stacked_outputs = torch.stack(model_outputs, dim=-1)
        weights = F.softmax(self.ensemble_weights, dim=0)

        ensemble_output = (stacked_outputs * weights).sum(dim=-1)
        return ensemble_output

    def estimate_value(self, *args, **kwargs) -> float:
        """估计状态价值"""
        with torch.no_grad():
            value = self.forward(*args, **kwargs)
            return value.item()

    def get_individual_estimates(self, *args, **kwargs) -> List[float]:
        """获取各个模型的独立估计"""
        with torch.no_grad():
            estimates = []
            for model in self.models:
                value = model(*args, **kwargs)
                estimates.append(value.item())
            return estimates

class AdaptiveValueFunction(ValueFunction):
    """自适应价值函数"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config.get('input_dim', 768)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.adaptation_rate = config.get('adaptation_rate', 0.01)

        # 主要价值网络
        self.main_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1)
        )

        # 自适应模块
        self.adaptation_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.Tanh()  # 输出调整因子
        )

        # 运行统计
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('num_updates', torch.tensor(0))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def update_statistics(self, values: torch.Tensor):
        """更新运行统计"""
        if self.training:
            batch_mean = values.mean()
            batch_var = values.var()

            # 更新运行平均
            if self.num_updates == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                momentum = min(self.adaptation_rate, 1.0 / (self.num_updates + 1))
                self.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)

            self.num_updates.add_(1)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 主网络输出
        main_output = self.main_network(state_features)

        # 自适应调整
        adaptation_factor = self.adaptation_network(state_features)
        adapted_output = main_output + adaptation_factor

        # 更新统计信息
        self.update_statistics(adapted_output)

        return adapted_output

    def estimate_value(self, state_features: torch.Tensor) -> float:
        """估计状态价值"""
        with torch.no_grad():
            value = self.forward(state_features)
            return value.item()

    def get_normalized_value(self, state_features: torch.Tensor) -> float:
        """获取标准化的价值估计"""
        with torch.no_grad():
            raw_value = self.forward(state_features)

            # 使用运行统计进行标准化
            if self.num_updates > 0:
                normalized_value = (raw_value - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
                return normalized_value.item()
            else:
                return raw_value.item()

def create_value_function(config: Dict[str, Any], model_type: str = 'linear') -> ValueFunction:
    """创建价值函数"""

    if model_type == 'linear':
        return LinearValueFunction(config)
    elif model_type == 'transformer':
        return TransformerValueFunction(config)
    elif model_type == 'context_aware':
        return ContextAwareValueFunction(config)
    elif model_type == 'hierarchical':
        return HierarchicalValueFunction(config)
    elif model_type == 'ensemble':
        return EnsembleValueFunction(config)
    elif model_type == 'adaptive':
        return AdaptiveValueFunction(config)
    else:
        raise ValueError(f"Unknown value function type: {model_type}")

# 价值函数工厂
class ValueFunctionFactory:
    """价值函数工厂"""

    _functions = {
        'linear': LinearValueFunction,
        'transformer': TransformerValueFunction,
        'context_aware': ContextAwareValueFunction,
        'hierarchical': HierarchicalValueFunction,
        'ensemble': EnsembleValueFunction,
        'adaptive': AdaptiveValueFunction
    }

    @classmethod
    def create(cls, model_type: str, config: Dict[str, Any]) -> ValueFunction:
        """创建价值函数"""
        if model_type not in cls._functions:
            raise ValueError(f"Unknown model type: {model_type}")

        return cls._functions[model_type](config)

    @classmethod
    def register(cls, name: str, function_class: type):
        """注册新的价值函数类"""
        cls._functions[name] = function_class

    @classmethod
    def list_functions(cls) -> List[str]:
        """列出所有可用的价值函数类型"""
        return list(cls._functions.keys())