#!/usr/bin/env python3
"""
StepSearch推理脚本 - 用于交互式问答
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
from src.utils.logging_utils import setup_logger


class StepSearchInference:
    """StepSearch推理类"""

    def __init__(self, model_path: str, config: dict):
        self.config = config
        self.max_search_steps = config['reward']['max_search_steps']

        # 加载模型
        print("Loading model...")
        self.model = create_step_search_model(config, with_value_head=False)

        if Path(model_path).is_dir():
            self.model.load_model(model_path)
        else:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        print("Model loaded successfully!")

        # 创建搜索引擎
        print("Initializing search engine...")
        self.search_engine = create_search_engine(config)
        print("Search engine ready!")

    def create_prompt(self, question: str) -> str:
        """创建推理提示"""
        prompt = f"""You are a helpful research assistant. Answer the question by searching for information step by step.

Question: {question}

Instructions:
- Use <think>...</think> to show your reasoning
- Use <search>...</search> to search for specific information
- Use <answer>...</answer> to provide your final answer
- Search for information when you need it, but try to be efficient

Begin your response:
"""
        return prompt

    def answer_question(self, question: str, verbose: bool = True) -> dict:
        """回答问题并返回详细信息"""
        if verbose:
            print(f"\nQuestion: {question}")
            print("=" * 60)

        prompt = self.create_prompt(question)
        current_state = prompt

        search_queries = []
        retrieved_docs = []
        reasoning_steps = []
        step_count = 0
        response_history = []

        while step_count < self.max_search_steps:
            # 生成响应
            response, _ = self.model.generate_response(
                current_state,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False
            )

            response_history.append(response)

            if verbose:
                print(f"\nStep {step_count + 1}:")
                print(response)

            # 解析响应
            import re

            # 提取思考过程
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                reasoning_steps.append(reasoning)
                if verbose:
                    print(f"[Reasoning]: {reasoning}")

            # 提取搜索查询
            search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
            if search_match:
                query = search_match.group(1).strip()
                search_queries.append(query)

                if verbose:
                    print(f"[Searching]: {query}")

                # 执行搜索
                docs = self.search_engine.search(query, top_k=3)
                retrieved_docs.extend(docs)

                if verbose:
                    print(f"[Retrieved {len(docs)} documents]")
                    for i, doc in enumerate(docs):
                        print(f"  {i + 1}. {doc[:100]}...")

                # 更新状态
                info_text = f"\n<information>{' '.join(docs)}</information>\n"
                current_state += response + info_text
            else:
                current_state += response + "\n"

            # 检查是否有最终答案
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
                if verbose:
                    print(f"\n[Final Answer]: {final_answer}")
                break

            step_count += 1

        # 如果没有找到明确的答案，尝试从最后的响应中提取
        if 'final_answer' not in locals():
            # 从整个对话中寻找最后一个answer标签
            all_text = current_state
            answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', all_text, re.DOTALL))
            if answer_matches:
                final_answer = answer_matches[-1].group(1).strip()
            else:
                final_answer = "Sorry, I couldn't find a definitive answer."

        result = {
            'question': question,
            'final_answer': final_answer,
            'search_queries': search_queries,
            'retrieved_docs': retrieved_docs,
            'reasoning_steps': reasoning_steps,
            'num_search_steps': len(search_queries),
            'full_conversation': current_state,
            'response_history': response_history
        }

        if verbose:
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            print(f"Search Steps: {len(search_queries)}")
            print("=" * 60)

        return result


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="StepSearch Interactive Inference")

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch', 'single'],
                        default='interactive', help='Inference mode')
    parser.add_argument('--question', type=str, default=None,
                        help='Single question for single mode')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Input file for batch mode')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file for batch mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def interactive_mode(inferencer: StepSearchInference):
    """交互模式"""
    print("\n" + "=" * 60)
    print("StepSearch Interactive Mode")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60)

    while True:
        try:
            question = input("\nEnter your question: ").strip()

            if question.lower() in ['quit', 'exit', '']:
                print("Goodbye!")
                break

            # 回答问题
            result = inferencer.answer_question(question, verbose=True)

            # 询问是否保存结果
            save_choice = input("\nSave this result? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = f"result_{len(question.split()[:3])}_words.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"Result saved to {filename}")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def single_mode(inferencer: StepSearchInference, question: str, verbose: bool = True):
    """单问题模式"""
    result = inferencer.answer_question(question, verbose=verbose)
    return result


def batch_mode(inferencer: StepSearchInference, input_file: str, output_file: str, verbose: bool = False):
    """批处理模式"""
    print(f"Processing batch file: {input_file}")

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                questions = [item['question'] if isinstance(item, dict) else str(item) for item in data]
            else:
                questions = [data['question']] if 'question' in data else [str(data)]
        else:
            # 假设是纯文本文件，每行一个问题
            questions = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(questions)} questions...")

    results = []
    for i, question in enumerate(questions):
        print(f"\nProcessing question {i + 1}/{len(questions)}: {question[:50]}...")

        try:
            result = inferencer.answer_question(question, verbose=verbose)
            results.append(result)

            if verbose:
                print(f"Answer: {result['final_answer']}")

        except Exception as e:
            print(f"Error processing question {i + 1}: {e}")
            results.append({
                'question': question,
                'error': str(e),
                'final_answer': 'Error occurred'
            })

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nBatch processing completed! Results saved to {output_file}")

    # 打印统计信息
    successful = sum(1 for r in results if 'error' not in r)
    print(f"Successfully processed: {successful}/{len(questions)} questions")


def main():
    """主函数"""
    args = parse_arguments()

    try:
        # 创建推理器
        inferencer = StepSearchInference(args.model_path, CONFIG)

        if args.mode == 'interactive':
            interactive_mode(inferencer)

        elif args.mode == 'single':
            if not args.question:
                args.question = input("Enter your question: ").strip()

            result = single_mode(inferencer, args.question, args.verbose)

            # 打印结果
            print(f"\nQuestion: {result['question']}")
            print(f"Answer: {result['final_answer']}")
            print(f"Search Steps: {result['num_search_steps']}")

            # 保存结果（如果指定输出文件）
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"Result saved to {args.output_file}")

        elif args.mode == 'batch':
            if not args.input_file:
                raise ValueError("Input file required for batch mode")
            if not args.output_file:
                args.output_file = 'batch_results.json'

            batch_mode(inferencer, args.input_file, args.output_file, args.verbose)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()