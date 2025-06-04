"""
评估指标计算
"""

import re
import string
from typing import List, Dict, Any
from collections import Counter

def normalize_answer(answer: str) -> str:
    """标准化答案文本"""
    if not answer:
        return ""

    # 转小写
    answer = answer.lower()

    # 移除标点符号
    answer = answer.translate(str.maketrans('', '', string.punctuation))

    # 移除多余空格
    answer = ' '.join(answer.split())

    return answer

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """计算精确匹配分数"""
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)

    return 1.0 if pred_normalized == gt_normalized else 0.0

def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """计算F1分数"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # 计算token级别的重叠
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    # 计算交集
    common_tokens = pred_counter & gt_counter
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    # 计算精确率和召回率
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall)

    return f1

def compute_em_f1_scores(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """批量计算EM和F1分数"""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")

    em_scores = []
    f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        # 处理多个可能的答案（如果ground truth是列表）
        if isinstance(gt, list):
            # 对每个可能答案计算分数，取最高分
            em_score = max(compute_exact_match(pred, answer) for answer in gt)
            f1_score = max(compute_f1_score(pred, answer) for answer in gt)
        else:
            em_score = compute_exact_match(pred, gt)
            f1_score = compute_f1_score(pred, gt)

        em_scores.append(em_score)
        f1_scores.append(f1_score)

    return {
        'exact_match': sum(em_scores) / len(em_scores),
        'f1_score': sum(f1_scores) / len(f1_scores),
        'num_samples': len(predictions)
    }

def compute_answer_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """计算详细的答案指标"""
    em_f1_scores = compute_em_f1_scores(predictions, ground_truths)

    # 计算额外指标
    exact_matches = [compute_exact_match(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    f1_scores = [compute_f1_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]

    # 统计信息
    num_exact_matches = sum(exact_matches)
    num_high_f1 = sum(1 for f1 in f1_scores if f1 >= 0.8)
    num_medium_f1 = sum(1 for f1 in f1_scores if 0.5 <= f1 < 0.8)
    num_low_f1 = sum(1 for f1 in f1_scores if 0.1 <= f1 < 0.5)
    num_zero_f1 = sum(1 for f1 in f1_scores if f1 < 0.1)

    # 计算答案长度统计
    pred_lengths = [len(pred.split()) for pred in predictions]
    gt_lengths = [len(gt.split()) if isinstance(gt, str) else len(gt[0].split())
                  for gt in ground_truths]

    metrics = {
        **em_f1_scores,
        'exact_match_count': num_exact_matches,
        'high_f1_count': num_high_f1,  # F1 >= 0.8
        'medium_f1_count': num_medium_f1,  # 0.5 <= F1 < 0.8
        'low_f1_count': num_low_f1,  # 0.1 <= F1 < 0.5
        'zero_f1_count': num_zero_f1,  # F1 < 0.1
        'avg_prediction_length': sum(pred_lengths) / len(pred_lengths),
        'avg_ground_truth_length': sum(gt_lengths) / len(gt_lengths),
        'prediction_length_std': (sum((l - sum(pred_lengths)/len(pred_lengths))**2 for l in pred_lengths) / len(pred_lengths))**0.5,
        'individual_em_scores': exact_matches,
        'individual_f1_scores': f1_scores
    }

    return metrics

def compute_retrieval_metrics(retrieved_docs_list: List[List[str]],
                            golden_docs_list: List[List[str]]) -> Dict[str, float]:
    """计算检索指标"""
    if len(retrieved_docs_list) != len(golden_docs_list):
        raise ValueError("Retrieved docs and golden docs must have the same length")

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for retrieved_docs, golden_docs in zip(retrieved_docs_list, golden_docs_list):
        if not golden_docs:
            continue

        # 计算文档级别的重叠
        retrieved_set = set(retrieved_docs)
        golden_set = set(golden_docs)

        intersection = retrieved_set & golden_set

        # 精确率：检索到的相关文档 / 总检索文档
        precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0

        # 召回率：检索到的相关文档 / 总相关文档
        recall = len(intersection) / len(golden_set) if golden_set else 0.0

        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        'retrieval_precision': sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        'retrieval_recall': sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        'retrieval_f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    }

def compute_search_efficiency_metrics(metadata_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算搜索效率指标"""
    search_steps = [m.get('num_search_steps', 0) for m in metadata_list]
    search_queries = [len(m.get('search_queries', [])) for m in metadata_list]

    # 基本统计
    avg_search_steps = sum(search_steps) / len(search_steps) if search_steps else 0.0
    avg_search_queries = sum(search_queries) / len(search_queries) if search_queries else 0.0

    # 搜索成功率（进行了至少一次搜索的样本比例）
    search_success_rate = sum(1 for steps in search_steps if steps > 0) / len(search_steps) if search_steps else 0.0

    # 搜索步数分布
    zero_search = sum(1 for steps in search_steps if steps == 0)
    one_search = sum(1 for steps in search_steps if steps == 1)
    multiple_search = sum(1 for steps in search_steps if steps > 1)

    return {
        'avg_search_steps': avg_search_steps,
        'avg_search_queries': avg_search_queries,
        'search_success_rate': search_success_rate,
        'zero_search_ratio': zero_search / len(search_steps) if search_steps else 0.0,
        'one_search_ratio': one_search / len(search_steps) if search_steps else 0.0,
        'multiple_search_ratio': multiple_search / len(search_steps) if search_steps else 0.0,
        'max_search_steps': max(search_steps) if search_steps else 0,
        'min_search_steps': min(search_steps) if search_steps else 0
    }

def compute_reasoning_quality_metrics(metadata_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算推理质量指标"""
    reasoning_steps = [len(m.get('reasoning_steps', [])) for m in metadata_list]

    # 平均推理步数
    avg_reasoning_steps = sum(reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0.0

    # 推理一致性（这里简化为检查是否有推理步骤）
    reasoning_consistency = sum(1 for steps in reasoning_steps if steps > 0) / len(reasoning_steps) if reasoning_steps else 0.0

    return {
        'avg_reasoning_steps': avg_reasoning_steps,
        'reasoning_consistency': reasoning_consistency,
        'max_reasoning_steps': max(reasoning_steps) if reasoning_steps else 0,
        'min_reasoning_steps': min(reasoning_steps) if reasoning_steps else 0
    }

def compute_comprehensive_metrics(predictions: List[str],
                                ground_truths: List[str],
                                metadata_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """计算综合指标"""
    # 基础答案指标
    answer_metrics = compute_answer_metrics(predictions, ground_truths)

    metrics = {
        'answer_metrics': answer_metrics
    }

    if metadata_list:
        # 搜索效率指标
        search_metrics = compute_search_efficiency_metrics(metadata_list)
        metrics['search_metrics'] = search_metrics

        # 推理质量指标
        reasoning_metrics = compute_reasoning_quality_metrics(metadata_list)
        metrics['reasoning_metrics'] = reasoning_metrics

        # 检索指标（如果有检索文档信息）
        retrieved_docs_list = [m.get('retrieved_docs', []) for m in metadata_list]
        golden_docs_list = [m.get('golden_docs', []) for m in metadata_list]

        if any(retrieved_docs_list) and any(golden_docs_list):
            retrieval_metrics = compute_retrieval_metrics(retrieved_docs_list, golden_docs_list)
            metrics['retrieval_metrics'] = retrieval_metrics

    return metrics

def print_metrics_summary(metrics: Dict[str, Any], dataset_name: str = "Dataset"):
    """打印指标摘要"""
    print(f"\n=== {dataset_name} Evaluation Results ===")

    if 'answer_metrics' in metrics:
        am = metrics['answer_metrics']
        print(f"Answer Metrics:")
        print(f"  Exact Match: {am['exact_match']:.3f} ({am['exact_match_count']}/{am['num_samples']})")
        print(f"  F1 Score: {am['f1_score']:.3f}")
        print(f"  High F1 (≥0.8): {am['high_f1_count']}/{am['num_samples']}")
        print(f"  Medium F1 (0.5-0.8): {am['medium_f1_count']}/{am['num_samples']}")
        print(f"  Low F1 (<0.5): {am['low_f1_count'] + am['zero_f1_count']}/{am['num_samples']}")

    if 'search_metrics' in metrics:
        sm = metrics['search_metrics']
        print(f"\nSearch Metrics:")
        print(f"  Avg Search Steps: {sm['avg_search_steps']:.1f}")
        print(f"  Search Success Rate: {sm['search_success_rate']:.3f}")
        print(f"  Multiple Search Rate: {sm['multiple_search_ratio']:.3f}")

    if 'reasoning_metrics' in metrics:
        rm = metrics['reasoning_metrics']
        print(f"\nReasoning Metrics:")
        print(f"  Avg Reasoning Steps: {rm['avg_reasoning_steps']:.1f}")
        print(f"  Reasoning Consistency: {rm['reasoning_consistency']:.3f}")

    if 'retrieval_metrics' in metrics:
        ret = metrics['retrieval_metrics']
        print(f"\nRetrieval Metrics:")
        print(f"  Retrieval Precision: {ret['retrieval_precision']:.3f}")
        print(f"  Retrieval Recall: {ret['retrieval_recall']:.3f}")
        print(f"  Retrieval F1: {ret['retrieval_f1']:.3f}")

    print("=" * 50)