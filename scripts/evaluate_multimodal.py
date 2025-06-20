# scripts/evaluate_multimodal.py
# !/usr/bin/env python3
import json
import torch
from pathlib import Path
from PIL import Image

from src.multimodal.multimodal_model import create_multimodal_step_search_model
from src.multimodal.multimodal_search import create_multimodal_search_engine
from src.evaluation.evaluator import StepSearchEvaluator
from src.evaluation.metrics import compute_em_f1_scores


class MultimodalEvaluator(StepSearchEvaluator):
    """多模态评估器"""

    def generate_answer(self, question: str, image_path: str = None):
        """生成多模态答案"""
        # 加载图像
        image = None
        if image_path and Path(image_path).exists():
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")

        # 创建多模态提示
        if image is not None:
            prompt = f"""You are a multimodal research assistant. Answer the question by analyzing the image and searching for information.

Question: {question}

Use the following format:
<think>your reasoning</think>
<image_analysis>describe the image</image_analysis>
<search>text search keywords</search>
<image_search>image-related search keywords</image_search>
<answer>final answer</answer>

Begin:"""
        else:
            prompt = f"""Answer the question by searching for information.

Question: {question}

Use the following format:
<think>your reasoning</think>
<search>search keywords</search>
<answer>final answer</answer>

Begin:"""

        # 生成响应
        response, _ = self.model.generate_multimodal_response(
            prompt, image,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False
        )

        # 提取答案
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        return ""


def evaluate_multimodal_model():
    """评估多模态模型"""

    # 加载模型
    model = create_multimodal_step_search_model(CONFIG, with_value_head=False)
    model.load_model('./checkpoints/multimodal/final_model')

    # 创建搜索引擎
    search_engine = create_multimodal_search_engine(CONFIG)

    # 创建评估器
    evaluator = MultimodalEvaluator(model, search_engine, CONFIG)

    # 加载测试数据
    with open('./data/multimodal/test_multimodal.json', 'r') as f:
        test_data = json.load(f)

    # 评估
    predictions = []
    ground_truths = []

    for item in test_data[:100]:  # 评估前100个样本
        pred_answer = evaluator.generate_answer(
            item['question'],
            item.get('image')
        )

        predictions.append(pred_answer)
        ground_truths.append(item['answer'])

        print(f"Q: {item['question']}")
        print(f"Pred: {pred_answer}")
        print(f"GT: {item['answer']}")
        print("-" * 50)

    # 计算指标
    metrics = compute_em_f1_scores(predictions, ground_truths)

    print(f"Evaluation Results:")
    print(f"Exact Match: {metrics['exact_match']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")

    return metrics


if __name__ == "__main__":
    evaluate_multimodal_model()