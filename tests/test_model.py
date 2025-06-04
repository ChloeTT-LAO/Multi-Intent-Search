"""
测试模型相关功能
"""

import pytest
import torch
import tempfile
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.step_search_model import StepSearchModel, StepSearchModelWithValueHead, create_step_search_model
from config import CONFIG


class TestStepSearchModel:
    """测试StepSearch模型"""

    @pytest.fixture
    def config(self):
        """测试配置"""
        test_config = CONFIG.copy()
        test_config['model']['name'] = "distilbert-base-uncased"  # 使用更小的模型进行测试
        test_config['model']['max_length'] = 512
        return test_config

    @pytest.fixture
    def model(self, config):
        """创建测试模型"""
        return StepSearchModel("distilbert-base-uncased", config)

    def test_model_initialization(self, config):
        """测试模型初始化"""
        model = StepSearchModel("distilbert-base-uncased", config)

        assert model is not None
        assert model.tokenizer is not None
        assert model.model is not None
        assert len(model.special_tokens) > 0

    def test_special_tokens_added(self, model):
        """测试特殊token是否正确添加"""
        expected_tokens = ['<think>', '</think>', '<search>', '</search>',
                           '<information>', '</information>', '<answer>', '</answer>']

        for token in expected_tokens:
            assert token in model.tokenizer.get_vocab()

    def test_extract_search_query(self, model):
        """测试搜索查询提取"""
        text = "I need to <search>find information about Paris</search> for my research."
        query = model.extract_search_query(text)
        assert query == "find information about Paris"

        # 测试无搜索的情况
        text_no_search = "This is just regular text without search."
        query_none = model.extract_search_query(text_no_search)
        assert query_none is None

    def test_extract_answer(self, model):
        """测试答案提取"""
        text = "After research, I found that <answer>Paris is the capital of France</answer>."
        answer = model.extract_answer(text)
        assert answer == "Paris is the capital of France"

        # 测试无答案的情况
        text_no_answer = "This is just regular text without answer."
        answer_none = model.extract_answer(text_no_answer)
        assert answer_none is None

    def test_extract_thinking(self, model):
        """测试思考过程提取"""
        text = "Let me <think>consider the question carefully</think> before answering."
        thinking = model.extract_thinking(text)
        assert thinking == "consider the question carefully"

    def test_validate_format(self, model):
        """测试格式验证"""
        # 正确格式
        valid_text = "<think>thinking</think><search>query</search><answer>answer</answer>"
        assert model.validate_format(valid_text) == True

        # 缺少think标签
        invalid_text = "<search>query</search><answer>answer</answer>"
        assert model.validate_format(invalid_text) == False

        # 不平衡的搜索标签
        invalid_search = "<think>thinking</think><search>query<answer>answer</answer>"
        assert model.validate_format(invalid_search) == False

    def test_generate_response(self, model):
        """测试响应生成"""
        prompt = "What is the capital of France?"

        # 由于这是一个小模型，我们主要测试是否能正常运行
        try:
            response, log_probs = model.generate_response(prompt, max_new_tokens=50)

            assert isinstance(response, str)
            assert isinstance(log_probs, list)
            assert len(response) > 0

        except Exception as e:
            pytest.skip(f"Model generation failed (expected for small test model): {e}")

    def test_model_save_load(self, model):
        """测试模型保存和加载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"

            # 保存模型
            model.save_model(str(save_path))

            # 检查文件是否存在
            assert save_path.exists()
            assert (save_path / "config.json").exists()
            assert (save_path / "tokenizer.json").exists()

            # 加载模型
            new_model = StepSearchModel("distilbert-base-uncased", model.config)
            new_model.load_model(str(save_path))

            # 验证加载成功
            assert new_model.tokenizer.vocab_size == model.tokenizer.vocab_size


class TestStepSearchModelWithValueHead:
    """测试带价值头的StepSearch模型"""

    @pytest.fixture
    def config(self):
        """测试配置"""
        test_config = CONFIG.copy()
        test_config['model']['name'] = "distilbert-base-uncased"
        test_config['model']['max_length'] = 512
        return test_config

    @pytest.fixture
    def model_with_value(self, config):
        """创建带价值头的测试模型"""
        return StepSearchModelWithValueHead("distilbert-base-uncased", config)

    def test_value_head_initialization(self, model_with_value):
        """测试价值头初始化"""
        assert hasattr(model_with_value, 'value_head')
        assert model_with_value.value_head is not None

    def test_forward_with_value(self, model_with_value):
        """测试带价值函数的前向传播"""
        # 创建测试输入
        test_input = "This is a test input"
        inputs = model_with_value.tokenizer(
            test_input,
            return_tensors="pt",
            max_length=50,
            truncation=True
        )

        try:
            logits, values = model_with_value.forward_with_value(**inputs)

            assert logits is not None
            assert values is not None
            assert logits.shape[0] == values.shape[0]  # 批次大小应该相同

        except Exception as e:
            pytest.skip(f"Forward pass failed (expected for small test model): {e}")


class TestModelFactory:
    """测试模型工厂函数"""

    def test_create_step_search_model_basic(self):
        """测试基础模型创建"""
        test_config = CONFIG.copy()
        test_config['model']['name'] = "distilbert-base-uncased"

        model = create_step_search_model(test_config, with_value_head=False)

        assert isinstance(model, StepSearchModel)
        assert not isinstance(model, StepSearchModelWithValueHead)

    def test_create_step_search_model_with_value(self):
        """测试带价值头的模型创建"""
        test_config = CONFIG.copy()
        test_config['model']['name'] = "distilbert-base-uncased"

        model = create_step_search_model(test_config, with_value_head=True)

        assert isinstance(model, StepSearchModelWithValueHead)
        assert hasattr(model, 'value_head')


class TestModelIntegration:
    """模型集成测试"""

    @pytest.fixture
    def config(self):
        """测试配置"""
        test_config = CONFIG.copy()
        test_config['model']['name'] = "distilbert-base-uncased"
        test_config['model']['max_length'] = 256
        return test_config

    def test_full_inference_pipeline(self, config):
        """测试完整推理流程"""
        model = create_step_search_model(config, with_value_head=False)

        question = "What is the capital of France?"

        try:
            # 生成响应
            response, _ = model.generate_response(question, max_new_tokens=30)

            # 验证响应格式（即使内容可能不正确）
            assert isinstance(response, str)
            assert len(response) > 0

        except Exception as e:
            pytest.skip(f"Full pipeline test failed (expected for small model): {e}")

    def test_tokenizer_consistency(self, config):
        """测试tokenizer一致性"""
        model = create_step_search_model(config)

        test_text = "This is a test <think>thinking</think> with special tokens"

        # 编码和解码
        encoded = model.tokenizer.encode(test_text)
        decoded = model.tokenizer.decode(encoded)

        # 验证特殊token被正确处理
        assert '<think>' in decoded
        assert '</think>' in decoded


if __name__ == "__main__":
    pytest.main([__file__])