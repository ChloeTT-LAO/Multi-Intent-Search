# scripts/interactive_multimodal.py
# !/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image

from src.multimodal.multimodal_model import create_multimodal_step_search_model
from src.multimodal.multimodal_search import create_multimodal_search_engine


def interactive_multimodal_qa():
    """交互式多模态问答"""

    # 加载模型
    print("Loading multimodal model...")
    model = create_multimodal_step_search_model(CONFIG, with_value_head=False)
    model.load_model('./checkpoints/multimodal/final_model')

    # 创建搜索引擎
    print("Loading search engine...")
    search_engine = create_multimodal_search_engine(CONFIG)

    print("Multimodal StepSearch Interactive Mode")
    print("Commands: 'quit' to exit, 'image <path>' to load image")
    print("=" * 50)

    current_image = None

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() == 'quit':
                break

            if user_input.startswith('image '):
                image_path = user_input[6:].strip()
                try:
                    current_image = Image.open(image_path).convert('RGB')
                    print(f"Image loaded: {image_path}")
                    continue
                except Exception as e:
                    print(f"Failed to load image: {e}")
                    continue

            # 生成回答
            response, _ = model.generate_multimodal_response(
                user_input, current_image,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )

            print(f"Assistant: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")


if __name__ == "__main__":
    interactive_multimodal_qa()