"""
StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization

This package provides a complete framework for training and using StepSearch models.
"""

__version__ = "1.0.0"
__author__ = "StepSearch Team"
__email__ = "stepsearch@example.com"

# Import main components for easy access
from .models.step_search_model import StepSearchModel, StepSearchModelWithValueHead, create_step_search_model
from .training.steppo_trainer import StePPOTrainer
from .training.reward_calculator import StepSearchRewardCalculator
from .search.search_engine import SearchEngine, TFIDFSearchEngine, MockSearchEngine, create_search_engine
from .evaluation.evaluator import StepSearchEvaluator, create_evaluator
from .data.dataset import StepSearchDataset, EvaluationDataset

# Define what gets imported with "from stepsearch import *"
__all__ = [
    # Models
    'StepSearchModel',
    'StepSearchModelWithValueHead',
    'create_step_search_model',

    # Training
    'StePPOTrainer',
    'StepSearchRewardCalculator',

    # Search
    'SearchEngine',
    'TFIDFSearchEngine',
    'MockSearchEngine',
    'create_search_engine',

    # Evaluation
    'StepSearchEvaluator',
    'create_evaluator',

    # Data
    'StepSearchDataset',
    'EvaluationDataset',
]

# Version info
VERSION = __version__