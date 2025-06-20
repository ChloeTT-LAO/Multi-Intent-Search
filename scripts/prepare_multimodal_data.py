# scripts/prepare_multimodal_data.py
import json
import requests
from pathlib import Path
from PIL import Image
import base64
import io


def download_image(url: str, save_path: str) -> bool:
    """下载图像"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False


def image_to_base64(image_path: str) -> str:
    """将图像转换为base64编码"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def create_multimodal_sample(text_data: dict, image_info: dict = None) -> dict:
    """创建多模态样本"""
    sample = {
        'id': text_data['id'],
        'question': text_data['question'],
        'answer': text_data['answer'],
        'subquestions': []
    }

    # 处理子问题
    for subq in text_data.get('subquestions', []):
        multimodal_subq = {
            'sub_question': subq['sub_question'],
            'reasoning': subq['reasoning'],
            'search_queries': subq['search_queries'],
            'image_search_queries': []
        }

        # 为每个子问题生成图像搜索查询
        if image_info:
            multimodal_subq['image_search_queries'] = generate_image_search_queries(
                subq['sub_question'], image_info
            )

        sample['subquestions'].append(multimodal_subq)

    # 添加图像信息
    if image_info:
        sample['image'] = image_info['path']
        sample['image_caption'] = image_info.get('caption', '')
        sample['image_metadata'] = image_info.get('metadata', {})

    return sample


def generate_image_search_queries(question: str, image_info: dict) -> list:
    """为子问题生成图像相关搜索查询"""
    queries = []

    # 基于图像标题/描述
    if 'caption' in image_info:
        queries.append(image_info['caption'])

    # 基于检测到的对象
    if 'objects' in image_info:
        for obj in image_info['objects'][:3]:
            queries.append(obj)

    # 结合问题和图像关键词
    question_words = question.lower().split()
    if 'keywords' in image_info:
        common_words = set(question_words) & set(image_info['keywords'])
        if common_words:
            queries.append(' '.join(common_words))

    return queries[:3]  # 最多返回3个查询