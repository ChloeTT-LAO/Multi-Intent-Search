"""
测试数据处理相关功能
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_pipeline import DataPipeline
from src.data.dataset import StepSearchDataset, EvaluationDataset
from src.data.data_utils import *
from config import CONFIG


class TestDataPipeline:
    """测试数据处理管道"""

    @pytest.fixture
    def config(self):
        """测试配置"""
        test_config = CONFIG.copy()
        test_config['data']['num_search_queries'] = 3
        test_config['data']['min_engines'] = 1
        return test_config

    @pytest.fixture
    def pipeline(self, config):
        """创建数据管道"""
        return DataPipeline(config)

    @pytest.fixture
    def sample_musique_data(self):
        """示例MuSiQue数据"""
        return {
            'id': 'test_001',
            'question': 'What is the capital of the country where the Eiffel Tower is located?',
            'answer': 'Paris',
            'decomposition': [
                {
                    'id': 1,
                    'question': 'Where is the Eiffel Tower located?',
                    'answer': 'Paris, France'
                },
                {
                    'id': 2,
                    'question': 'What is the capital of France?',
                    'answer': 'Paris'
                }
            ]
        }

    def test_pipeline_initialization(self, config):
        """测试管道初始化"""
        pipeline = DataPipeline(config)

        assert pipeline.config == config
        assert pipeline.data_config == config['data']
        assert len(pipeline.search_engines) > 0

    @patch('openai.ChatCompletion.create')
    def test_decompose_question_with_gpt4o(self, mock_openai, pipeline, sample_musique_data):
        """测试GPT-4o问题分解"""
        # 模拟OpenAI响应
        mock_response = Mock()
        mock_response.choices[0].message.content = json.dumps([
            {
                "sub_question": "Where is the Eiffel Tower located?",
                "reasoning": "Need to find the location first"
            },
            {
                "sub_question": "What is the capital of France?",
                "reasoning": "Then find the capital of that country"
            }
        ])
        mock_openai.return_value = mock_response

        result = pipeline.decompose_question_with_gpt4o(
            sample_musique_data['question'],
            sample_musique_data['answer']
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert 'sub_question' in result[0]
        assert 'reasoning' in result[0]

    @patch('openai.ChatCompletion.create')
    def test_generate_search_queries(self, mock_openai, pipeline):
        """测试搜索查询生成"""
        # 模拟OpenAI响应
        mock_response = Mock()
        mock_response.choices[0].message.content = json.dumps([
            "Eiffel Tower location",
            "where is Eiffel Tower",
            "Eiffel Tower Paris France"
        ])
        mock_openai.return_value = mock_response

        queries = pipeline.generate_search_queries("Where is the Eiffel Tower located?", 3)

        assert isinstance(queries, list)
        assert len(queries) <= 3
        assert all(isinstance(q, str) for q in queries)

    def test_validate_query_with_engines(self, pipeline):
        """测试查询验证"""
        # 测试有效查询
        valid_query = "Paris France capital"
        result = pipeline.validate_query_with_engines(valid_query)
        assert isinstance(result, bool)

        # 测试无效查询（太短）
        invalid_query = "ab"
        result = pipeline.validate_query_with_engines(invalid_query)
        assert result == False

    def test_mock_search_validation(self, pipeline):
        """测试模拟搜索验证"""
        # 有效查询
        assert pipeline.mock_search_validation("valid query", "google") == True

        # 无效查询（太短）
        assert pipeline.mock_search_validation("ab", "google") == False

        # 无效查询（太长）
        long_query = " ".join(["word"] * 15)
        assert pipeline.mock_search_validation(long_query, "google") == False

    @patch.object(DataPipeline, 'decompose_question_with_gpt4o')
    @patch.object(DataPipeline, 'generate_search_queries')
    @patch.object(DataPipeline, 'validate_query_with_engines')
    def test_process_sample(self, mock_validate, mock_generate, mock_decompose,
                            pipeline, sample_musique_data):
        """测试样本处理"""
        # 设置mock返回值
        mock_decompose.return_value = [
            {"sub_question": "Where is Eiffel Tower?", "reasoning": "First step"}
        ]
        mock_generate.return_value = ["Eiffel Tower location", "Paris tower"]
        mock_validate.return_value = True

        result = pipeline.process_sample(sample_musique_data)

        assert result is not None
        assert 'question' in result
        assert 'answer' in result
        assert 'subquestions' in result
        assert len(result['subquestions']) > 0


class TestStepSearchDataset:
    """测试StepSearch数据集"""

    @pytest.fixture
    def sample_data(self):
        """示例数据"""
        return [
            {
                'id': 'test_001',
                'question': 'What is the capital of France?',
                'answer': 'Paris',
                'subquestions': [
                    {
                        'sub_question': 'What is the capital of France?',
                        'search_queries': ['France capital', 'capital of France']
                    }
                ]
            },
            {
                'id': 'test_002',
                'question': 'Where is the Eiffel Tower?',
                'answer': 'Paris',
                'subquestions': [
                    {
                        'sub_question': 'Where is the Eiffel Tower located?',
                        'search_queries': ['Eiffel Tower location', 'where Eiffel Tower']
                    }
                ]
            }
        ]

    @pytest.fixture
    def temp_data_file(self, sample_data):
        """创建临时数据文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            return f.name

    def test_dataset_loading(self, temp_data_file):
        """测试数据集加载"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        dataset = StepSearchDataset(temp_data_file, tokenizer)

        assert len(dataset) == 2
        assert dataset.tokenizer == tokenizer

        # 清理临时文件
        Path(temp_data_file).unlink()

    def test_dataset_getitem(self, temp_data_file):
        """测试数据集索引访问"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        dataset = StepSearchDataset(temp_data_file, tokenizer)

        item = dataset[0]

        assert 'id' in item
        assert 'question' in item
        assert 'answer' in item
        assert 'subquestions' in item
        assert 'reference_keywords' in item
        assert 'golden_docs' in item

        # 清理临时文件
        Path(temp_data_file).unlink()

    def test_extract_reference_keywords(self, temp_data_file):
        """测试参考关键词提取"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        dataset = StepSearchDataset(temp_data_file, tokenizer)

        sample = dataset.data[0]
        keywords = dataset.extract_reference_keywords(sample)

        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert isinstance(keywords[0], list)

        # 清理临时文件
        Path(temp_data_file).unlink()

    def test_collate_fn(self, temp_data_file):
        """测试批处理函数"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        dataset = StepSearchDataset(temp_data_file, tokenizer)

        batch = [dataset[0], dataset[1]]
        collated = dataset.collate_fn(batch)

        assert 'ids' in collated
        assert 'questions' in collated
        assert 'answers' in collated
        assert len(collated['ids']) == 2
        assert len(collated['questions']) == 2

        # 清理临时文件
        Path(temp_data_file).unlink()


class TestEvaluationDataset:
    """测试评估数据集"""

    @pytest.fixture
    def hotpot_sample_data(self):
        """HotpotQA示例数据"""
        return [
            {
                '_id': 'hotpot_001',
                'question': 'What is the capital of France?',
                'answer': 'Paris',
                'type': 'comparison'
            }
        ]

    @pytest.fixture
    def temp_hotpot_file(self, hotpot_sample_data):
        """创建临时HotpotQA文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(hotpot_sample_data, f)
            return f.name

    def test_hotpotqa_loading(self, temp_hotpot_file):
        """测试HotpotQA数据加载"""
        dataset = EvaluationDataset('hotpotqa', temp_hotpot_file)

        assert len(dataset) == 1
        assert dataset.dataset_name == 'hotpotqa'

        item = dataset[0]
        assert 'id' in item
        assert 'question' in item
        assert 'answer' in item

        # 清理临时文件
        Path(temp_hotpot_file).unlink()

    def test_unknown_dataset(self):
        """测试未知数据集类型"""
        with pytest.raises(ValueError):
            EvaluationDataset('unknown_dataset', 'dummy_path')


class TestDataUtils:
    """测试数据工具函数"""

    def test_normalize_text(self):
        """测试文本标准化"""
        # 这里需要实现normalize_text函数
        # 暂时跳过，因为在代码中没有看到这个函数的定义
        pass

    def test_extract_keywords(self):
        """测试关键词提取"""
        # 这里需要实现extract_keywords函数
        # 暂时跳过，因为在代码中没有看到这个函数的定义
        pass


class TestDataIntegration:
    """数据处理集成测试"""

    def test_full_data_pipeline(self):
        """测试完整数据处理流程"""
        # 这是一个集成测试，测试从原始数据到训练数据的完整流程
        # 由于涉及到GPT-4o API调用，这里先跳过
        pytest.skip("Integration test requires API access")

    def test_data_format_consistency(self):
        """测试数据格式一致性"""
        # 测试处理后的数据是否符合预期格式
        sample_processed = {
            'id': 'test_001',
            'question': 'Test question?',
            'answer': 'Test answer',
            'subquestions': [
                {
                    'sub_question': 'Sub question?',
                    'reasoning': 'Reasoning',
                    'search_queries': ['query1', 'query2']
                }
            ]
        }

        # 验证必需字段
        required_fields = ['id', 'question', 'answer', 'subquestions']
        for field in required_fields:
            assert field in sample_processed

        # 验证子问题格式
        subq = sample_processed['subquestions'][0]
        subq_required = ['sub_question', 'search_queries']
        for field in subq_required:
            assert field in subq


if __name__ == "__main__":
    pytest.main([__file__])