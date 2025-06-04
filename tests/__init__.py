"""
StepSearch测试模块
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 测试配置
TEST_CONFIG = {
    'model': {
        'name': 'distilbert-base-uncased',  # 使用小模型进行测试
        'max_length': 256
    },
    'training': {
        'batch_size': 2,
        'learning_rate': 1e-3,
        'max_steps': 10
    },
    'reward': {
        'gamma_key': 0.1,
        'redundancy_threshold': 0.8,
        'max_search_steps': 3
    },
    'search': {
        'engine_type': 'mock',
        'top_k': 3
    }
}


# 测试工具函数
def skip_if_no_cuda():
    """如果没有CUDA则跳过测试"""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def skip_if_no_internet():
    """如果没有网络连接则跳过测试"""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except OSError:
        pytest.skip("No internet connection")


def skip_if_no_openai_key():
    """如果没有OpenAI API key则跳过测试"""
    import os
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip("OpenAI API key not available")


# 测试装饰器
def requires_cuda(func):
    """需要CUDA的测试装饰器"""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )(func)


def requires_internet(func):
    """需要网络的测试装饰器"""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        has_internet = True
    except OSError:
        has_internet = False

    return pytest.mark.skipif(
        not has_internet,
        reason="No internet connection"
    )(func)


def requires_openai_key(func):
    """需要OpenAI API key的测试装饰器"""
    import os
    return pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY'),
        reason="OpenAI API key not available"
    )(func)


# 测试夹具
@pytest.fixture
def test_config():
    """测试配置夹具"""
    return TEST_CONFIG.copy()


@pytest.fixture
def sample_question():
    """示例问题夹具"""
    return "What is the capital of France?"


@pytest.fixture
def sample_answer():
    """示例答案夹具"""
    return "Paris"


@pytest.fixture
def sample_search_data():
    """示例搜索数据夹具"""
    return {
        'question': 'What is the capital of France?',
        'answer': 'Paris',
        'subquestions': [
            {
                'sub_question': 'What is the capital of France?',
                'search_queries': ['France capital', 'capital of France', 'Paris France']
            }
        ],
        'reference_keywords': [['France capital', 'capital of France', 'Paris France']],
        'golden_docs': ['Paris is the capital and largest city of France.']
    }


@pytest.fixture
def mock_knowledge_base():
    """模拟知识库夹具"""
    return [
        "Paris is the capital and largest city of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is located in Paris.",
        "London is the capital of the United Kingdom.",
        "Tokyo is the capital of Japan."
    ]


__all__ = [
    'TEST_CONFIG',
    'skip_if_no_cuda',
    'skip_if_no_internet',
    'skip_if_no_openai_key',
    'requires_cuda',
    'requires_internet',
    'requires_openai_key'
]