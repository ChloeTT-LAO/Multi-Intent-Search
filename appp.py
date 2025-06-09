#!/usr/bin/env python3
"""
测试DeepSeek API调用
"""

import requests
import json
import os


def main():
    """测试DeepSeek API调用"""

    # API配置
    api_key = "sk-anbsoppiznxtiuhzxdibxuvpnhsxoabbsderulnnzsfduyrq"
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 测试1: 最简单的请求
    print("=== 测试1: 基础API调用 ===")

    simple_payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=simple_payload, headers=headers)
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ 基础API调用成功!")
            print(f"回复: {result['choices'][0]['message']['content']}")
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False


    # 测试3: 问题分解任务
    print("\n=== 测试3: 问题分解任务 ===")

    decomposition_payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "messages": [
            {
                "role": "user",
                "content": """
Please decompose the following multi-hop question into a series of sub-questions.

Question: What is the population of the capital city of France?
Final Answer: Paris has a population of approximately 2.1 million.

Format your response as a JSON list where each item has:
- "sub_question": the sub-question text  
- "reasoning": brief explanation

Example format:
[
    {"sub_question": "What is the capital of France?", "reasoning": "Need to identify the capital first"}
]
"""
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=decomposition_payload, headers=headers)
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print("✅ 问题分解任务成功!")
            print(f"原始回复: {content}")

            # 尝试解析JSON
            try:
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group())
                    print(f"✅ JSON解析成功: {parsed_json}")
                else:
                    print("⚠️ 未找到JSON格式的回复")
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON解析失败: {e}")

        else:
            print(f"❌ 问题分解任务失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"❌ 问题分解任务异常: {e}")

    return True


def api_limits():
    """测试API限制和参数"""
    print("\n=== 测试4: API参数兼容性 ===")

    api_key = "sk-anbsoppiznxtiuhzxdibxuvpnhsxoabbsderulnnzsfduyrq"
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 测试各种参数组合
    params_to_test = [
        {
            "name": "基础参数",
            "params": {
                "model": "Qwen/QwQ-32B",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "temperature": 0.7
            }
        },
        {
            "name": "添加top_p和top_k",
            "params": {
                "model": "Qwen/QwQ-32B",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50
            }
        },
        {
            "name": "添加frequency_penalty",
            "params": {
                "model": "Qwen/QwQ-32B",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "temperature": 0.7,
                "frequency_penalty": 0.5
            }
        },
        {
            "name": "添加不支持的参数",
            "params": {
                "model": "Qwen/QwQ-32B",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "temperature": 0.7,
                "enable_thinking": True,  # 可能不支持
                "thinking_budget": 4096,  # 可能不支持
                "min_p": 0.05  # 可能不支持
            }
        }
    ]

    for test_case in params_to_test:
        print(f"\n测试: {test_case['name']}")
        try:
            response = requests.post(url, json=test_case['params'], headers=headers)
            if response.status_code == 200:
                print(f"✅ {test_case['name']} 成功")
            else:
                print(f"❌ {test_case['name']} 失败: {response.status_code}")
                print(f"错误: {response.text}")
        except Exception as e:
            print(f"❌ {test_case['name']} 异常: {e}")


if __name__ == "__main__":
    print("开始测试DeepSeek API...")

    if main():
        api_limits()

    print("\n测试完成!")