"""
数据集类定义
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from pathlib import Path


class StepSearchDataset(Dataset):
    """StepSearch训练数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        return {
            'id': sample.get('id', str(idx)),
            'question': sample['question'],
            'answer': sample['answer'],
            'subquestions': sample['subquestions'],
            'reference_keywords': self.extract_reference_keywords(sample),
            'golden_docs': self.extract_golden_docs(sample)
        }

    def extract_reference_keywords(self, sample: Dict[str, Any]) -> List[List[str]]:
        """提取参考关键词"""
        keywords = []
        for subq_data in sample['subquestions']:
            keywords.append(subq_data['search_queries'])
        return keywords

    def extract_golden_docs(self, sample: Dict[str, Any]) -> List[str]:
        """提取黄金标准文档（这里需要根据实际情况实现）"""
        # 这是一个简化版本，实际可能需要从原始MuSiQue数据中提取
        golden_docs = []
        if 'golden_docs' in sample:
            golden_docs = sample['golden_docs']
        else:
            # 回退方案：使用子问题作为伪黄金文档
            for subq_data in sample['subquestions']:
                golden_docs.append(subq_data['sub_question'])
        return golden_docs

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批处理函数"""
        return {
            'ids': [item['id'] for item in batch],
            'questions': [item['question'] for item in batch],
            'answers': [item['answer'] for item in batch],
            'subquestions': [item['subquestions'] for item in batch],
            'reference_keywords': [item['reference_keywords'] for item in batch],
            'golden_docs': [item['golden_docs'] for item in batch]
        }


class EvaluationDataset(Dataset):
    """评估数据集"""

    def __init__(self, dataset_name: str, data_path: str):
        self.dataset_name = dataset_name
        self.data_path = data_path

        # 加载不同格式的评估数据
        if dataset_name == 'hotpotqa':
            self.data = self.load_hotpotqa(data_path)
        elif dataset_name == '2wiki':
            self.data = self.load_2wiki(data_path)
        elif dataset_name == 'musique':
            self.data = self.load_musique(data_path)
        elif dataset_name == 'bamboogle':
            self.data = self.load_bamboogle(data_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def load_hotpotqa(self, data_path: str) -> List[Dict[str, Any]]:
        """加载HotpotQA数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_data.append({
                'id': item['_id'],
                'question': item['question'],
                'answer': item['answer'],
                'type': item.get('type', 'multi'),
                'level': item.get('level', 'hard')
            })
        return processed_data

    def load_2wiki(self, data_path: str) -> List[Dict[str, Any]]:
        """加载2WikiMultiHopQA数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_data.append({
                'id': item['_id'],
                'question': item['question'],
                'answer': item['answer']
            })
        return processed_data

    def load_musique(self, data_path: str) -> List[Dict[str, Any]]:
        """加载MuSiQue数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_data.append({
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer']
            })
        return processed_data

    def load_bamboogle(self, data_path: str) -> List[Dict[str, Any]]:
        """加载Bamboogle数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_data.append({
                'id': item.get('id', str(len(processed_data))),
                'question': item['input'],
                'answer': item['target']
            })
        return processed_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def load_train_dataset(data_path: str, tokenizer, config: Dict[str, Any]) -> StepSearchDataset:
    """加载训练数据集"""
    return StepSearchDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )


def load_eval_dataset(dataset_name: str, data_path: str) -> EvaluationDataset:
    """加载评估数据集"""
    return EvaluationDataset(dataset_name, data_path)