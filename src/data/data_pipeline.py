"""
数据处理管道 - 基于MuSiQue构建搜索轨迹数据
"""

import json
import openai
import time
import os
import requests
import gdown
import zipfile
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from config import CONFIG
import re


class DataPipeline:
    """StepSearch数据处理管道"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']
        self.search_engines = ['google', 'bing', 'wiki']

        # 设置API keys
        self.setup_api_keys()

    def setup_api_keys(self):
        """设置API密钥"""
        # OpenAI API key
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

        # DeepSeek API key
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', "sk-anbsoppiznxtiuhzxdibxuvpnhsxoabbsderulnnzsfduyrq")
        self.deepseek_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.deepseek_headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }

    def load_local_musique_dataset(self, data_dir: Path) -> Optional[DatasetDict]:
        """从本地目录加载MuSiQue数据集"""

        try:
            datasets = {}

            # 查找数据文件
            data_files = {}
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.jsonl'):
                        file_path = os.path.join(root, file)
                        if 'train' in file:
                            data_files['train'] = file_path
                        elif 'dev' in file:
                            data_files['validation'] = file_path
                        elif 'test' in file:
                            data_files['test'] = file_path

            print(f"Found data files: {data_files}")

            # 加载每个分割
            for split, file_path in data_files.items():
                print(f"Loading {split} from {file_path}")

                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))

                datasets[split] = Dataset.from_list(data)
                print(f"Loaded {len(data)} examples for {split}")

            if datasets:
                return DatasetDict(datasets)
            else:
                print("No data files found")
                return None

        except Exception as e:
            print(f"Failed to load local dataset: {e}")
            return None

    def load_musique_data(self, raw_data_dir: str = "./data/raw") -> Dict[str, Any]:
        """加载MuSiQue数据集"""
        print("Loading MuSiQue dataset...")
        raw_data_dir = Path(raw_data_dir)

        # 方法1: 优先从本地加载（prepare_data.py已下载的数据）

        # 检查Google Drive下载的数据（JSONL格式）
        gdrive_extracted_dir = raw_data_dir / "musique"
        if gdrive_extracted_dir.exists():
            print("Loading from local Google Drive download...")
            dataset = self.load_local_musique_dataset(gdrive_extracted_dir)
            if dataset:
                return dataset

        # # 方法2: 最后尝试从Google Drive下载
        # print("Hub failed, attempting download from Google Drive...")
        # dataset = self.download_musique_from_gdrive(raw_data_dir)
        # if dataset:
        #     return dataset

        raise Exception("无法加载MuSiQue数据集，请检查网络连接或手动下载")

    def decompose_question_with_gpt4o(self, question: str, answer: str) -> List[Dict[str, str]]:
        """使用GPT-4o分解问题为子问题"""
        if not self.openai_api_key:
            print("No OpenAI API key available, falling back to simple decomposition")
            return [{"sub_question": question, "reasoning": "Single step question"}]

        prompt = f"""
        Please decompose the following multi-hop question into a series of sub-questions that need to be answered sequentially to reach the final answer.

        Question: {question}
        Final Answer: {answer}

        Format your response as a JSON list where each item has:
        - "sub_question": the sub-question text
        - "reasoning": brief explanation of why this sub-question is needed

        Example format:
        [
            {{"sub_question": "What is the first sub-question?", "reasoning": "This establishes the basic information needed"}},
            {{"sub_question": "What is the second sub-question?", "reasoning": "This builds on the first answer to get closer to the final answer"}}
        ]
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )

            result = response.choices[0].message.content
            # 解析JSON响应
            subquestions = json.loads(result)
            return subquestions

        except Exception as e:
            print(f"Error in GPT-4o decomposition: {e}")
            # 回退到简单分解
            return [{"sub_question": question, "reasoning": "Single step question"}]

    def generate_search_queries(self, sub_question: str, num_queries: int = 5) -> List[str]:
        """为子问题生成搜索查询"""
        if not self.openai_api_key:
            print("No OpenAI API key available, using simple query generation")
            return [sub_question, sub_question.replace("?", "").strip()]

        prompt = f"""
        Generate {num_queries} different search queries that would help answer this question. 
        Make the queries diverse and specific.

        Question: {sub_question}

        Format as a JSON list of strings:
        ["query1", "query2", "query3", ...]
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=300
            )

            result = response.choices[0].message.content
            queries = json.loads(result)
            return queries[:num_queries]

        except Exception as e:
            print(f"Error generating search queries: {e}")
            # 回退到基础查询生成
            return [sub_question, sub_question.replace("?", "").strip()]

    def decompose_question_with_deepseek(self, question: str, answer: str) -> List[Dict[str, str]]:
        """使用DeepSeek-R1分解问题为子问题"""
        prompt = f"""
        Please decompose the following multi-hop question into a series of sub-questions that need to be answered sequentially to reach the final answer.

        Question: {question}
        Final Answer: {answer}

        Format your response as a JSON list where each item has:
        - "sub_question": the sub-question text
        - "reasoning": brief explanation of why this sub-question is needed

        Example format:
        [
            {{"sub_question": "What is the first sub-question?", "reasoning": "This establishes the basic information needed"}},
            {{"sub_question": "What is the second sub-question?", "reasoning": "This builds on the first answer to get closer to the final answer"}}
        ]
        """

        payload = {
            "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "max_tokens": 1000,
            "thinking_budget": 4096,
            "min_p": 0.05,
            "stop": None,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }

        try:
            response = requests.post(self.deepseek_url, json=payload, headers=self.deepseek_headers)

            # 打印详细错误信息
            if response.status_code != 200:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()

            result_data = response.json()
            result = result_data['choices'][0]['message']['content']

            # 尝试解析JSON响应
            try:
                subquestions = json.loads(result)
                return subquestions
            except json.JSONDecodeError:
                # 如果不是标准JSON，尝试提取JSON部分
                import re
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    subquestions = json.loads(json_match.group())
                    return subquestions
                else:
                    print(f"Could not parse JSON from response: {result}")
                    return [{"sub_question": question, "reasoning": "Single step question"}]

        except requests.exceptions.RequestException as e:
            print(f"Error in API request: {e}")
            return [{"sub_question": question, "reasoning": "Single step question"}]
        except Exception as e:
            print(f"Error in DeepSeek decomposition: {e}")
            return [{"sub_question": question, "reasoning": "Single step question"}]

    def generate_search_queries_with_deepseek(self, sub_question: str, num_queries: int = 5) -> List[str]:
        """为子问题生成搜索查询"""
        prompt = f"""
        Generate {num_queries} different search queries that would help answer this question. 
        Make the queries diverse and specific.

        Question: {sub_question}

        Format as a JSON list of strings:
        ["query1", "query2", "query3", ...]
        """

        payload = {
            "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "max_tokens": 300,
            "thinking_budget": 4096,
            "min_p": 0.05,
            "stop": None,
            "temperature": 0.8,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }

        try:
            response = requests.post(self.deepseek_url, json=payload, headers=self.deepseek_headers)

            if response.status_code != 200:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()

            result_data = response.json()
            result = result_data['choices'][0]['message']['content']

            try:
                queries = json.loads(result)
                return queries[:num_queries]
            except json.JSONDecodeError:
                # 尝试提取JSON部分
                import re
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    queries = json.loads(json_match.group())
                    cleaned_queries = self._clean_and_validate_queries(queries, num_queries)
                    return cleaned_queries
                else:
                    print(f"Could not parse JSON from response: {result}")
                    return [sub_question, sub_question.replace("?", "").strip()]

        except requests.exceptions.RequestException as e:
            print(f"Error in API request: {e}")
            return [sub_question, sub_question.replace("?", "").strip()]
        except Exception as e:
            print(f"Error generating search queries: {e}")
            return [sub_question, sub_question.replace("?", "").strip()]

    def _clean_and_validate_queries(self, queries: Any, num_queries: int) -> List[str]:
        """清理和验证查询列表，确保返回字符串列表"""
        cleaned_queries = []

        # 确保queries是列表
        if not isinstance(queries, list):
            print(f"Warning: Expected list, got {type(queries)}: {queries}")
            return self._fallback_queries("fallback")

        for item in queries:
            # 处理嵌套列表的情况
            if isinstance(item, list):
                # 如果是列表，尝试连接成字符串
                if all(isinstance(x, str) for x in item):
                    query_str = " ".join(item).strip()
                    if query_str:
                        cleaned_queries.append(query_str)
                else:
                    print(f"Warning: Nested list contains non-string items: {item}")
            elif isinstance(item, str):
                # 如果是字符串，直接添加
                query_str = item.strip()
                if query_str:
                    cleaned_queries.append(query_str)
            elif isinstance(item, dict):
                # 如果是字典，尝试提取文本值
                text_values = []
                for value in item.values():
                    if isinstance(value, str):
                        text_values.append(value.strip())
                if text_values:
                    query_str = " ".join(text_values).strip()
                    if query_str:
                        cleaned_queries.append(query_str)
                else:
                    print(f"Warning: Dict contains no string values: {item}")
            else:
                print(f"Warning: Unexpected item type {type(item)}: {item}")

        # 确保返回所需数量的查询
        if len(cleaned_queries) < num_queries:
            return cleaned_queries[:len(cleaned_queries)]

        return cleaned_queries[:num_queries]

    def _fallback_queries(self, sub_question: str) -> List[str]:
        """生成后备查询列表"""
        base_query = sub_question.replace("?", "").strip()
        return [
            sub_question,
            base_query,
            f"{base_query} information",
            f"{base_query} details",
            f"{base_query} facts"
        ]

    def validate_query_with_engines(self, query: str) -> bool:
        """验证查询在多个搜索引擎中是否有效"""
        # 这里是模拟验证，实际需要调用真实搜索引擎API
        valid_engines = 0
        if not isinstance(query, str):
            return False

        for engine in self.search_engines:
            # 模拟搜索引擎调用
            if self.mock_search_validation(query, engine):
                valid_engines += 1

        # 需要在至少一半的搜索引擎中有效
        return valid_engines >= len(self.search_engines) // 2 + 1

    def mock_search_validation(self, query: str, engine: str) -> bool:
        """模拟搜索引擎验证（实际实现需要真实API）"""
        # 简单的启发式验证
        if len(query.strip()) < 3:
            return False
        if query.count(' ') > 10:  # 太长的查询可能无效
            return False
        return True

    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理单个样本"""
        question = sample['question']
        answer = sample['answer']

        # 步骤1: 分解问题
        # 优先使用DeepSeek，如果失败则尝试GPT-4o
        print("Start to decompose question...")
        if self.deepseek_api_key:
            subquestions = self.decompose_question_with_deepseek(question, answer)
        elif self.openai_api_key:
            subquestions = self.decompose_question_with_gpt4o(question, answer)
        else:
            # 回退到简单分解
            subquestions = [{"sub_question": question, "reasoning": "Single step question"}]
        print("Subquestions: ", subquestions)


        # 步骤2: 为每个子问题生成搜索查询
        print("Start to develop queries...")
        search_trajectories = []
        for subq_data in subquestions:
            subq = subq_data['sub_question']

            # 优先使用DeepSeek，如果失败则尝试GPT-4o
            if self.deepseek_api_key:
                queries = self.generate_search_queries_with_deepseek(subq, self.data_config['num_search_queries'])
            elif self.openai_api_key:
                queries = self.generate_search_queries(subq, self.data_config['num_search_queries'])
            else:
                # 回退到简单查询生成
                queries = [subq, subq.replace("?", "").strip()]
            print("queries: ", queries)

            # 步骤3: 验证查询
            print("Start to evaluate queries...")
            valid_queries = [q for q in queries if self.validate_query_with_engines(q)]

            if valid_queries:
                search_trajectories.append({
                    'sub_question': subq,
                    'reasoning': subq_data['reasoning'],
                    'search_queries': valid_queries
                })

        if not search_trajectories:
            return None

        return {
            'id': sample.get('id', ''),
            'question': question,
            'answer': answer,
            'subquestions': search_trajectories,
            'original_sample': sample
        }

    def process_dataset(self, dataset, split: str = 'train', max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """处理整个数据集"""
        processed_data = []
        samples = dataset[split]

        if max_samples:
            samples = samples.select(range(min(max_samples, len(samples))))

        print(f"Processing {len(samples)} samples from {split} split...")

        for i, sample in enumerate(tqdm(samples, desc=f"Processing {split}")):

            processed_sample = self.process_sample(sample)
            if processed_sample:
                processed_data.append(processed_sample)

            # 添加延迟以避免API限制
            if i % 10 == 0:
                time.sleep(1)

        print(f"Successfully processed {len(processed_data)} samples")
        return processed_data

    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str):
        """保存处理后的数据"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved processed data to {output_path}")

    def run_pipeline(self, max_samples_train: int = 1000, max_samples_dev: int = 200, raw_data_dir: str = "./data/raw"):
        """运行完整的数据处理管道"""
        # 加载原始数据
        dataset = self.load_musique_data(raw_data_dir)

        # 处理训练集
        train_data = self.process_dataset(dataset, 'train', max_samples_train)
        train_output = Path(self.data_config['output_path']) / 'train_processed.json'
        self.save_processed_data(train_data, train_output)

        # 处理验证集
        dev_data = self.process_dataset(dataset, 'validation', max_samples_dev)
        dev_output = Path(self.data_config['output_path']) / 'dev_processed.json'
        self.save_processed_data(dev_data, dev_output)

        return {
            'train_data': train_data,
            'dev_data': dev_data,
            'train_path': train_output,
            'dev_path': dev_output
        }


def main():
    """主函数"""
    pipeline = DataPipeline(CONFIG)
    result = pipeline.run_pipeline(max_samples_train=1000, max_samples_dev=200)
    print("Data pipeline completed!")
    print(f"Train samples: {len(result['train_data'])}")
    print(f"Dev samples: {len(result['dev_data'])}")


if __name__ == "__main__":
    main()