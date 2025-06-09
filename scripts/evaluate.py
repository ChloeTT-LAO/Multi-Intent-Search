#!/usr/bin/env python3
"""
StepSearch评估脚本
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config import CONFIG
from src.models.step_search_model import create_step_search_model
from src.search.search_engine import create_search_engine
from src.evaluation.evaluator import create_evaluator
from src.utils.logging_utils import setup_logger


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Evaluate StepSearch model")

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['hotpotqa', '2wiki', 'musique', 'bamboogle'],
                        help='Datasets to evaluate on')
    parser.add_argument('--musique-dir', type=str, required=True,
                        help='Directory containing evaluation datasets')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per dataset')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Evaluation batch size')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save detailed predictions')
    parser.add_argument('--compare-baselines', action='store_true',
                        help='Compare with baseline results')

    return parser.parse_args()


def setup_evaluation_environment(args):
    """设置评估环境"""
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger('evaluate', output_dir / 'evaluate.log')

    return logger, output_dir


def load_model_and_components(model_path: str, logger):
    """加载模型和相关组件"""
    logger.info(f"Loading model from {model_path}")

    # 创建模型
    model = create_step_search_model(CONFIG, with_value_head=False)

    # 加载模型权重
    if Path(model_path).is_dir():
        # 从目录加载
        model.load_model(model_path)
    else:
        # 从检查点加载
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    logger.info("Model loaded successfully")

    # 创建搜索引擎
    logger.info("Creating search engine...")
    search_engine = create_search_engine(CONFIG)

    return model, search_engine


def prepare_dataset_configs(datasets: list, data_dir: str) -> list:
    """准备数据集配置"""
    dataset_configs = []
    data_dir = Path(data_dir)

    # 数据集文件映射
    dataset_files = {
        'hotpotqa': 'hotpot_dev_distractor_v1.json',
        '2wiki': '2wiki_dev.json',
        'musique': 'musique_ans_v1.0_dev.jsonl',
        'bamboogle': 'bamboogle_dev.json'
    }

    for dataset_name in datasets:
        if dataset_name in dataset_files:
            file_path = data_dir / dataset_files[dataset_name]

            # 如果文件不存在，尝试其他可能的文件名
            if not file_path.exists():
                possible_files = [
                    f"{dataset_name}_dev.json",
                    f"{dataset_name}_test.json",
                    f"{dataset_name}.json",
                    f"{dataset_name}_eval.json"
                ]

                for possible_file in possible_files:
                    alt_path = data_dir / possible_file
                    if alt_path.exists():
                        file_path = alt_path
                        break

            if file_path.exists():
                dataset_configs.append({
                    'name': dataset_name,
                    'path': str(file_path)
                })
            else:
                print(f"Warning: Dataset file not found for {dataset_name}")
        else:
            print(f"Warning: Unknown dataset {dataset_name}")

    return dataset_configs


def save_detailed_results(results: dict, output_dir: Path, save_predictions: bool):
    """保存详细结果"""
    # 保存汇总结果
    summary_path = output_dir / 'evaluation_summary.json'
    summary_data = {}

    for dataset_name, result in results.items():
        if 'error' in result:
            summary_data[dataset_name] = result
        else:
            summary_data[dataset_name] = {
                'metrics': result['metrics'],
                'num_samples': len(result['predictions'])
            }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"Summary saved to {summary_path}")

    # 保存详细预测结果
    if save_predictions:
        for dataset_name, result in results.items():
            if 'error' not in result:
                detail_path = output_dir / f'{dataset_name}_detailed.json'

                detailed_data = {
                    'predictions': result['predictions'],
                    'ground_truths': result['ground_truths'],
                    'metadata': result['metadata'],
                    'metrics': result['metrics']
                }

                with open(detail_path, 'w', encoding='utf-8') as f:
                    json.dump(detailed_data, f, ensure_ascii=False, indent=2)

                print(f"Detailed results for {dataset_name} saved to {detail_path}")


def print_results_table(results: dict):
    """打印结果表格"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Dataset':<15} {'EM':<8} {'F1':<8} {'Samples':<8} {'Avg Steps':<10} {'Success Rate':<12}")
    print("-" * 80)

    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"{dataset_name:<15} {'ERROR':<8} {'ERROR':<8} {'-':<8} {'-':<10} {'-':<12}")
        else:
            metrics = result['metrics']
            em = metrics['exact_match']
            f1 = metrics['f1_score']
            samples = len(result['predictions'])
            avg_steps = metrics.get('avg_search_steps', 0)
            success_rate = metrics.get('search_success_rate', 0)

            print(f"{dataset_name:<15} {em:<8.3f} {f1:<8.3f} {samples:<8} {avg_steps:<10.1f} {success_rate:<12.3f}")

    print("=" * 80)


def compare_with_baselines(results: dict, logger):
    """与基线结果比较"""
    # 论文中的基线结果 (Search-R1)
    baseline_results = {
        'hotpotqa': {'exact_match': 0.272, 'f1_score': 0.361},
        '2wiki': {'exact_match': 0.248, 'f1_score': 0.296},
        'musique': {'exact_match': 0.081, 'f1_score': 0.146},
        'bamboogle': {'exact_match': 0.176, 'f1_score': 0.270}
    }

    print("\n" + "=" * 80)
    print("COMPARISON WITH SEARCH-R1 BASELINE")
    print("=" * 80)
    print(f"{'Dataset':<12} {'Our EM':<8} {'Base EM':<8} {'EM Δ':<8} {'Our F1':<8} {'Base F1':<8} {'F1 Δ':<8}")
    print("-" * 80)

    improvements = {}

    for dataset_name, result in results.items():
        if 'error' in result or dataset_name not in baseline_results:
            continue

        our_metrics = result['metrics']
        baseline_metrics = baseline_results[dataset_name]

        our_em = our_metrics['exact_match']
        our_f1 = our_metrics['f1_score']
        base_em = baseline_metrics['exact_match']
        base_f1 = baseline_metrics['f1_score']

        em_improvement = our_em - base_em
        f1_improvement = our_f1 - base_f1

        improvements[dataset_name] = {
            'em_improvement': em_improvement,
            'f1_improvement': f1_improvement
        }

        print(f"{dataset_name:<12} "
              f"{our_em:<8.3f} "
              f"{base_em:<8.3f} "
              f"{em_improvement:+<8.3f} "
              f"{our_f1:<8.3f} "
              f"{base_f1:<8.3f} "
              f"{f1_improvement:+<8.3f}")

    # 计算平均改进
    if improvements:
        avg_em_improvement = sum(imp['em_improvement'] for imp in improvements.values()) / len(improvements)
        avg_f1_improvement = sum(imp['f1_improvement'] for imp in improvements.values()) / len(improvements)

        print("-" * 80)
        print(
            f"{'Average':<12} {'':<8} {'':<8} {avg_em_improvement:+<8.3f} {'':<8} {'':<8} {avg_f1_improvement:+<8.3f}")

    print("=" * 80)

    logger.info("Baseline comparison completed")
    return improvements


def main():
    """主函数"""
    args = parse_arguments()

    # 设置评估环境
    logger, output_dir = setup_evaluation_environment(args)
    logger.info("Starting StepSearch evaluation...")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # 加载模型和组件
        model, search_engine = load_model_and_components(args.model_path, logger)

        # 创建评估器
        logger.info("Creating evaluator...")
        evaluator = create_evaluator(model, search_engine, CONFIG)

        # 准备数据集配置
        dataset_configs = prepare_dataset_configs(args.datasets, args.data_dir)
        logger.info(f"Evaluating on {len(dataset_configs)} datasets: {[c['name'] for c in dataset_configs]}")

        # 执行评估
        logger.info("Starting evaluation...")
        results = evaluator.evaluate_multiple_datasets(
            dataset_configs,
            max_samples_per_dataset=args.max_samples
        )

        # 打印结果表格
        print_results_table(results)

        # 与基线比较
        if args.compare_baselines:
            improvements = compare_with_baselines(results, logger)

        # 保存结果
        logger.info("Saving results...")
        save_detailed_results(results, output_dir, args.save_predictions)

        # 保存比较结果
        if args.compare_baselines:
            comparison_path = output_dir / 'baseline_comparison.json'
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(improvements, f, ensure_ascii=False, indent=2)
            logger.info(f"Baseline comparison saved to {comparison_path}")

        logger.info("Evaluation completed successfully!")

        # 计算总体统计
        valid_results = [r for r in results.values() if 'error' not in r]
        if valid_results:
            avg_em = sum(r['metrics']['exact_match'] for r in valid_results) / len(valid_results)
            avg_f1 = sum(r['metrics']['f1_score'] for r in valid_results) / len(valid_results)
            total_samples = sum(len(r['predictions']) for r in valid_results)

            print(f"\nOverall Statistics:")
            print(f"  Average EM: {avg_em:.3f}")
            print(f"  Average F1: {avg_f1:.3f}")
            print(f"  Total Samples: {total_samples}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise

    logger.info("Evaluation script completed!")


if __name__ == "__main__":
    main()