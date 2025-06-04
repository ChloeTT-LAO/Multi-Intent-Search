"""
评估器实现
"""

import json
import torch
from typing import Dict, List, Any, Tuple
from pathlib import Path
from tqdm import tqdm

from ..models.step_search_model import StepSearchModel
from ..data.dataset import EvaluationDataset
from .metrics import compute_em_f1_scores, compute_answer_metrics


class StepSearchEvaluator:
    """StepSearch评估器"""

    def __init__(self, model: StepSearchModel, search_engine, config: Dict[str, Any]):
        self.model = model
        self.search_engine = search_engine
        self.config = config
        self.max_search_steps = config['reward']['max_search_steps']

        # 设置模型为评估模式
        self.model.eval()

    def create_evaluation_prompt(self, question: str) -> str:
        """创建评估提示"""
        prompt = f"""You are a research assistant. Answer the question by searching for information step by step.

Question: {question}

Use the following format:
<think>your reasoning process</think>
<search>search keywords</search>
(information will be provided)
<think>continue reasoning based on the information</think>
<answer>final answer</answer>

Begin your response:
"""
        return prompt

    def extract_answer_from_response(self, response: str) -> str:
        """从响应中提取最终答案"""
        import re

        # 寻找最后一个answer标签
        answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', response, re.DOTALL))
        if answer_matches:
            return answer_matches[-1].group(1).strip()

        # 如果没有找到answer标签，尝试从最后部分提取
        lines = response.strip().split('\n')
        for line in reversed(lines):
            if line.strip() and not line.startswith('<'):
                return line.strip()

        return ""

    def generate_answer(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """为给定问题生成答案"""
        prompt = self.create_evaluation_prompt(question)
        current_state = prompt

        search_queries = []
        retrieved_docs = []
        reasoning_steps = []
        step_count = 0

        while step_count < self.max_search_steps:
            # 生成响应
            response, _ = self.model.generate_response(
                current_state,
                max_new_tokens=256,
                temperature=0.1,  # 较低温度以获得更一致的结果
                do_sample=False  # 使用贪婪解码
            )

            # 解析响应
            import re

            # 提取思考过程
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if think_match:
                reasoning_steps.append(think_match.group(1).strip())

            # 提取搜索查询
            search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
            if search_match:
                query = search_match.group(1).strip()
                search_queries.append(query)

                # 执行搜索
                docs = self.search_engine.search(query, top_k=3)
                retrieved_docs.extend(docs)

                # 更新状态
                info_text = f"\n<information>{' '.join(docs)}</information>\n"
                current_state += response + info_text
            else:
                current_state += response

            # 检查是否有答案
            if '<answer>' in response:
                break

            step_count += 1

        # 提取最终答案
        final_answer = self.extract_answer_from_response(current_state)

        # 收集元数据
        metadata = {
            'search_queries': search_queries,
            'retrieved_docs': retrieved_docs,
            'reasoning_steps': reasoning_steps,
            'num_search_steps': len(search_queries),
            'full_response': current_state
        }

        return final_answer, metadata

    def evaluate_dataset(self, dataset: EvaluationDataset,
                         max_samples: int = None) -> Dict[str, Any]:
        """评估整个数据集"""
        predictions = []
        ground_truths = []
        metadata_list = []

        # 确定评估样本数量
        num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

        print(f"Evaluating {num_samples} samples from {dataset.dataset_name}")

        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Evaluating"):
                sample = dataset[i]
                question = sample['question']
                gt_answer = sample['answer']

                # 生成预测
                try:
                    pred_answer, metadata = self.generate_answer(question)
                    predictions.append(pred_answer)
                    ground_truths.append(gt_answer)
                    metadata_list.append(metadata)

                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    predictions.append("")
                    ground_truths.append(gt_answer)
                    metadata_list.append({})

        # 计算指标
        metrics = compute_em_f1_scores(predictions, ground_truths)

        # 添加额外的统计信息
        avg_search_steps = sum(m.get('num_search_steps', 0) for m in metadata_list) / len(metadata_list)
        metrics['avg_search_steps'] = avg_search_steps

        # 计算搜索成功率（至少有一次搜索的样本比例）
        search_success_rate = sum(1 for m in metadata_list if m.get('num_search_steps', 0) > 0) / len(metadata_list)
        metrics['search_success_rate'] = search_success_rate

        return {
            'metrics': metrics,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'metadata': metadata_list,
            'dataset_name': dataset.dataset_name
        }

    def evaluate_multiple_datasets(self, dataset_configs: List[Dict[str, str]],
                                   max_samples_per_dataset: int = None) -> Dict[str, Any]:
        """评估多个数据集"""
        results = {}

        for config in dataset_configs:
            dataset_name = config['name']
            data_path = config['path']

            print(f"\n=== Evaluating {dataset_name} ===")

            try:
                # 加载数据集
                dataset = EvaluationDataset(dataset_name, data_path)

                # 评估
                result = self.evaluate_dataset(dataset, max_samples_per_dataset)
                results[dataset_name] = result

                # 打印结果
                metrics = result['metrics']
                print(f"Results for {dataset_name}:")
                print(f"  Exact Match: {metrics['exact_match']:.3f}")
                print(f"  F1 Score: {metrics['f1_score']:.3f}")
                print(f"  Average Search Steps: {metrics['avg_search_steps']:.1f}")
                print(f"  Search Success Rate: {metrics['search_success_rate']:.3f}")

            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}

        return results

    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 准备保存的数据（排除不可序列化的内容）
        save_data = {}
        for dataset_name, result in results.items():
            if 'error' in result:
                save_data[dataset_name] = result
            else:
                save_data[dataset_name] = {
                    'metrics': result['metrics'],
                    'dataset_name': result['dataset_name'],
                    'num_samples': len(result['predictions'])
                }

                # 保存详细的预测结果到单独文件
                detail_path = output_path.parent / f"{dataset_name}_detailed.json"
                detailed_data = {
                    'predictions': result['predictions'],
                    'ground_truths': result['ground_truths'],
                    'metadata': result['metadata']
                }

                with open(detail_path, 'w', encoding='utf-8') as f:
                    json.dump(detailed_data, f, ensure_ascii=False, indent=2)

        # 保存汇总结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"Evaluation results saved to {output_path}")

    def compare_with_baselines(self, results: Dict[str, Any],
                               baseline_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """与基线结果比较"""
        if baseline_results is None:
            # 默认基线结果（来自论文）
            baseline_results = {
                'hotpotqa': {'exact_match': 0.272, 'f1_score': 0.361},
                '2wiki': {'exact_match': 0.248, 'f1_score': 0.296},
                'musique': {'exact_match': 0.081, 'f1_score': 0.146},
                'bamboogle': {'exact_match': 0.176, 'f1_score': 0.270}
            }

        comparison = {}

        for dataset_name, result in results.items():
            if 'error' in result or dataset_name not in baseline_results:
                continue

            our_metrics = result['metrics']
            baseline_metrics = baseline_results[dataset_name]

            comparison[dataset_name] = {
                'our_em': our_metrics['exact_match'],
                'baseline_em': baseline_metrics['exact_match'],
                'em_improvement': our_metrics['exact_match'] - baseline_metrics['exact_match'],
                'our_f1': our_metrics['f1_score'],
                'baseline_f1': baseline_metrics['f1_score'],
                'f1_improvement': our_metrics['f1_score'] - baseline_metrics['f1_score']
            }

        return comparison

    def print_comparison_table(self, comparison: Dict[str, Any]):
        """打印比较表格"""
        print("\n" + "=" * 80)
        print("COMPARISON WITH BASELINES")
        print("=" * 80)
        print(f"{'Dataset':<12} {'Our EM':<8} {'Base EM':<8} {'EM Δ':<8} {'Our F1':<8} {'Base F1':<8} {'F1 Δ':<8}")
        print("-" * 80)

        for dataset_name, comp in comparison.items():
            print(f"{dataset_name:<12} "
                  f"{comp['our_em']:<8.3f} "
                  f"{comp['baseline_em']:<8.3f} "
                  f"{comp['em_improvement']:+<8.3f} "
                  f"{comp['our_f1']:<8.3f} "
                  f"{comp['baseline_f1']:<8.3f} "
                  f"{comp['f1_improvement']:+<8.3f}")

        print("=" * 80)


def create_evaluator(model: StepSearchModel, search_engine, config: Dict[str, Any]) -> StepSearchEvaluator:
    """创建评估器"""
    return StepSearchEvaluator(model, search_engine, config)