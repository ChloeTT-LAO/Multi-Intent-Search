"""
数据处理管道 - 基于MuSiQue构建搜索轨迹数据
"""

import json
import openai
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datasets import load_dataset
from config import CONFIG
import os


class DataPipeline:
    """StepSearch数据处理管道"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']
        self.search_engines = ['google', 'bing', 'wiki']

        # 设置OpenAI API (如果使用GPT-4o)
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def load_musique_data(self) -> Dict[str, Any]:
        """加载MuSiQue数据集"""
        print("Loading MuSiQue dataset...")
        dataset = load_dataset("tau/musique", "musique_v1.0")
        return dataset

    def decompose_question_with_gpt4o(self, question: str, answer: str) -> List[Dict[str, str]]:
        """使用GPT-4o分解问题为子问题"""
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

    def validate_query_with_engines(self, query: str) -> bool:
        """验证查询在多个搜索引擎中是否有效"""
        # 这里是模拟验证，实际需要调用真实搜索引擎API
        valid_engines = 0

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
        subquestions = self.decompose_question_with_gpt4o(question, answer)

        # 步骤2: 为每个子问题生成搜索查询
        search_trajectories = []
        for subq_data in subquestions:
            subq = subq_data['sub_question']
            queries = self.generate_search_queries(subq, self.data_config['num_search_queries'])

            # 步骤3: 验证查询
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

        for i, sample in enumerate(samples):
            if i % 100 == 0:
                print(f"Processed {i}/{len(samples)} samples")

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

    def run_pipeline(self, max_samples_train: int = 1000, max_samples_dev: int = 200):
        """运行完整的数据处理管道"""
        # 加载原始数据
        dataset = self.load_musique_data()

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