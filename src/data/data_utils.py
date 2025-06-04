"""
数据处理工具函数
"""

import json
import re
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL格式文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存为JSONL格式文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_json(file_path: str) -> Any:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def clean_text(text: str) -> str:
    """清理文本"""
    if not text:
        return ""

    # 移除多余的空格和换行
    text = re.sub(r'\s+', ' ', text)

    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 移除特殊字符
    text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\(\)]', ' ', text)

    return text.strip()


def extract_entities_simple(text: str) -> Dict[str, List[str]]:
    """简单的实体提取"""
    entities = {
        'dates': [],
        'numbers': [],
        'proper_nouns': []
    }

    # 提取日期
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    ]

    for pattern in date_patterns:
        entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))

    # 提取数字
    entities['numbers'] = re.findall(r'\b\d+(?:\.\d+)?\b', text)

    # 提取专有名词（大写字母开头的单词）
    entities['proper_nouns'] = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

    return entities


def split_into_sentences(text: str) -> List[str]:
    """将文本分割为句子"""
    if not text:
        return []

    # 简单的句子分割
    sentences = re.split(r'[.!?]+', text)

    # 清理并过滤空句子
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 5:  # 过滤太短的句子
            cleaned_sentences.append(sentence)

    return cleaned_sentences


def extract_keywords_simple(text: str, top_k: int = 10) -> List[str]:
    """简单的关键词提取"""
    if not text:
        return []

    # 移除停用词
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'among', 'within', 'without', 'against', 'across'
    }

    # 提取单词
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # 过滤停用词并计数
    word_counts = defaultdict(int)
    for word in words:
        if word not in stopwords:
            word_counts[word] += 1

    # 返回前top_k个关键词
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:top_k]]


def generate_search_variations(query: str) -> List[str]:
    """生成搜索查询的变体"""
    if not query:
        return []

    variations = [query]

    # 移除标点符号的版本
    no_punct = re.sub(r'[^\w\s]', ' ', query)
    if no_punct != query:
        variations.append(no_punct.strip())

    # 提取关键词版本
    keywords = extract_keywords_simple(query, 5)
    if keywords:
        variations.append(' '.join(keywords))

    # 提取专有名词版本
    entities = extract_entities_simple(query)
    if entities['proper_nouns']:
        variations.append(' '.join(entities['proper_nouns'][:3]))

    # 去重
    unique_variations = []
    seen = set()
    for var in variations:
        var = var.strip()
        if var and var not in seen:
            unique_variations.append(var)
            seen.add(var)

    return unique_variations[:5]  # 最多返回5个变体


def validate_data_format(data: List[Dict[str, Any]], required_keys: List[str]) -> Tuple[bool, List[str]]:
    """验证数据格式"""
    errors = []

    if not isinstance(data, list):
        errors.append("Data should be a list")
        return False, errors

    if len(data) == 0:
        errors.append("Data is empty")
        return False, errors

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not a dictionary")
            continue

        for key in required_keys:
            if key not in item:
                errors.append(f"Item {i} missing required key: {key}")

    return len(errors) == 0, errors


def sample_data(data: List[Any], n: int, random_seed: int = 42) -> List[Any]:
    """随机采样数据"""
    if len(data) <= n:
        return data

    random.seed(random_seed)
    return random.sample(data, n)


def balance_dataset(data: List[Dict[str, Any]], label_key: str, max_per_class: int = None) -> List[Dict[str, Any]]:
    """平衡数据集"""
    if not data:
        return data

    # 按标签分组
    grouped = defaultdict(list)
    for item in data:
        if label_key in item:
            grouped[item[label_key]].append(item)

    # 平衡采样
    balanced_data = []
    for label, items in grouped.items():
        if max_per_class and len(items) > max_per_class:
            items = sample_data(items, max_per_class)
        balanced_data.extend(items)

    # 随机打乱
    random.shuffle(balanced_data)
    return balanced_data


def filter_by_length(data: List[Dict[str, Any]], text_key: str,
                     min_length: int = 10, max_length: int = 1000) -> List[Dict[str, Any]]:
    """根据文本长度过滤数据"""
    filtered_data = []

    for item in data:
        if text_key in item:
            text = item[text_key]
            if isinstance(text, str) and min_length <= len(text) <= max_length:
                filtered_data.append(item)

    return filtered_data


def merge_datasets(datasets: List[List[Dict[str, Any]]], weights: List[float] = None) -> List[Dict[str, Any]]:
    """合并多个数据集"""
    if not datasets:
        return []

    if weights is None:
        weights = [1.0] * len(datasets)

    if len(weights) != len(datasets):
        raise ValueError("Number of weights must match number of datasets")

    merged_data = []

    for dataset, weight in zip(datasets, weights):
        # 根据权重采样
        n_samples = int(len(dataset) * weight)
        if n_samples > 0:
            sampled = sample_data(dataset, n_samples)
            merged_data.extend(sampled)

    # 随机打乱
    random.shuffle(merged_data)
    return merged_data


def create_data_splits(data: List[Dict[str, Any]],
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1,
                       random_seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """创建数据分割"""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    random.seed(random_seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    n_total = len(shuffled_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': shuffled_data[:n_train],
        'validation': shuffled_data[n_train:n_train + n_val],
        'test': shuffled_data[n_train + n_val:]
    }

    return splits


def compute_data_statistics(data: List[Dict[str, Any]], text_keys: List[str] = None) -> Dict[str, Any]:
    """计算数据统计信息"""
    if not data:
        return {}

    if text_keys is None:
        text_keys = ['question', 'answer', 'text']

    stats = {
        'total_samples': len(data),
        'text_statistics': {}
    }

    for key in text_keys:
        if any(key in item for item in data):
            texts = [item.get(key, '') for item in data if key in item]
            text_lengths = [len(text) for text in texts if isinstance(text, str)]

            if text_lengths:
                stats['text_statistics'][key] = {
                    'count': len(text_lengths),
                    'avg_length': np.mean(text_lengths),
                    'min_length': min(text_lengths),
                    'max_length': max(text_lengths),
                    'median_length': np.median(text_lengths)
                }

    return stats


def normalize_text_for_matching(text: str) -> str:
    """标准化文本用于匹配"""
    if not text:
        return ""

    # 转换为小写
    text = text.lower()

    # 移除标点符号
    text = re.sub(r'[^\w\s]', ' ', text)

    # 标准化空格
    text = ' '.join(text.split())

    return text


def compute_text_similarity_simple(text1: str, text2: str) -> float:
    """计算简单的文本相似度"""
    if not text1 or not text2:
        return 0.0

    # 标准化文本
    norm_text1 = normalize_text_for_matching(text1)
    norm_text2 = normalize_text_for_matching(text2)

    # 分词
    words1 = set(norm_text1.split())
    words2 = set(norm_text2.split())

    if not words1 or not words2:
        return 0.0

    # 计算Jaccard相似度
    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def deduplicate_data(data: List[Dict[str, Any]],
                     text_key: str,
                     similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
    """去重数据"""
    if not data:
        return data

    unique_data = []
    seen_texts = []

    for item in data:
        if text_key not in item:
            unique_data.append(item)
            continue

        current_text = item[text_key]
        is_duplicate = False

        # 检查是否与已有文本相似
        for seen_text in seen_texts:
            similarity = compute_text_similarity_simple(current_text, seen_text)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_data.append(item)
            seen_texts.append(current_text)

    return unique_data


class DataProcessor:
    """数据处理器类"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_text_length = self.config.get('min_text_length', 10)
        self.max_text_length = self.config.get('max_text_length', 1000)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.9)

    def process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理原始数据"""
        processed_data = []

        for item in raw_data:
            processed_item = self.process_single_item(item)
            if processed_item:
                processed_data.append(processed_item)

        return processed_data

    def process_single_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理单个数据项"""
        processed_item = item.copy()

        # 清理文本字段
        text_fields = ['question', 'answer', 'text', 'context']
        for field in text_fields:
            if field in processed_item:
                processed_item[field] = clean_text(processed_item[field])

        # 验证数据质量
        if not self.validate_item(processed_item):
            return None

        return processed_item

    def validate_item(self, item: Dict[str, Any]) -> bool:
        """验证单个数据项"""
        # 检查必需字段
        required_fields = self.config.get('required_fields', ['question', 'answer'])
        for field in required_fields:
            if field not in item or not item[field]:
                return False

        # 检查文本长度
        for field in ['question', 'answer']:
            if field in item:
                text = item[field]
                if not (self.min_text_length <= len(text) <= self.max_text_length):
                    return False

        return True

    def create_training_examples(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """创建训练样本"""
        training_examples = []

        for item in data:
            # 为每个数据项创建训练样本
            example = {
                'id': item.get('id', ''),
                'question': item['question'],
                'answer': item['answer'],
                'context': item.get('context', ''),
                'metadata': {
                    'source': item.get('source', 'unknown'),
                    'difficulty': item.get('difficulty', 'medium')
                }
            }

            training_examples.append(example)

        return training_examples