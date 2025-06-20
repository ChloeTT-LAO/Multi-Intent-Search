"""
StepSearch多模态扩展模块
"""

from .multimodal_model import (
    MultimodalStepSearchModel,
    MultimodalStepSearchModelWithValueHead,
    create_multimodal_step_search_model
)

from .multimodal_search import (
    MultimodalSearchEngine,
    ImageTextSearchEngine,
    CLIPSearchEngine,
    create_multimodal_search_engine
)

from .multimodal_data import (
    MultimodalDataset,
    MultimodalEvaluationDataset,
    load_multimodal_train_dataset,
    load_multimodal_eval_dataset,
    MultimodalDataPipeline
)

from .multimodal_trainer import (
    MultimodalStePPOTrainer,
    MultimodalTrajectoryData,
    MultimodalTrajectoryStep
)

from .multimodal_reward import (
    MultimodalRewardCalculator,
    CrossModalRewardCalculator
)

__all__ = [
    # 多模态模型
    'MultimodalStepSearchModel',
    'MultimodalStepSearchModelWithValueHead',
    'create_multimodal_step_search_model',

    # 多模态搜索
    'MultimodalSearchEngine',
    'ImageTextSearchEngine',
    'CLIPSearchEngine',
    'create_multimodal_search_engine',

    # 多模态数据
    'MultimodalDataset',
    'MultimodalEvaluationDataset',
    'load_multimodal_train_dataset',
    'load_multimodal_eval_dataset',
    'MultimodalDataPipeline',

    # 多模态训练
    'MultimodalStePPOTrainer',
    'MultimodalTrajectoryData',
    'MultimodalTrajectoryStep',

    # 多模态奖励
    'MultimodalRewardCalculator',
    'CrossModalRewardCalculator'
]