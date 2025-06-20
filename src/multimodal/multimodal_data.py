"""
多模态数据处理
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import base64
import io


class MultimodalDataset(Dataset):
    """多模态StepSearch训练数据集"""

    def __init__(self, data_path: str, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} multimodal samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        return {
            'id': sample.get('id', str(idx)),
            'question': sample['question'],
            'image': sample.get('image', None),
            'answer': sample['answer'],
            'subquestions': sample['subquestions'],
            'reference_keywords': self.extract_reference_keywords(sample),
            'golden_docs': self.extract_golden_docs(sample),
            'image_info': sample.get('image_info', {})
        }

    def extract_reference_keywords(self, sample: Dict[str, Any]) -> List[List[str]]:
        """提取参考关键词"""
        keywords = []
        for subq_data in sample['subquestions']:
            subq_keywords = []
            # 文本搜索关键词
            if 'search_queries' in subq_data:
                subq_keywords.extend(subq_data['search_queries'])
            # 图像搜索关键词
            if 'image_search_queries' in subq_data:
                subq_keywords.extend(subq_data['image_search_queries'])
            keywords.append(subq_keywords)
        return keywords

    def extract_golden_docs(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取黄金标准文档（包含文本和图像）"""
        golden_docs = []
        if 'golden_docs' in sample:
            golden_docs = sample['golden_docs']
        else:
            # 回退方案：使用子问题作为伪黄金文档
            for subq_data in sample['subquestions']:
                doc = {
                    'text': subq_data['sub_question'],
                    'image': None
                }
                golden_docs.append(doc)
        return golden_docs

    def process_image(self, image_input: Union[str, Dict[str, Any], None]) -> Optional[Image.Image]:
        """处理图像输入"""
        if image_input is None:
            return None

        if isinstance(image_input, str):
            if image_input.startswith('data:image'):
                # Base64编码的图片
                header, data = image_input.split(',', 1)
                image_data = base64.b64decode(data)
                return Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                # 文件路径
                return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, dict):
            # 图像信息字典
            if 'path' in image_input:
                return Image.open(image_input['path']).convert('RGB')
            elif 'data' in image_input:
                image_data = base64.b64decode(image_input['data'])
                return Image.open(io.BytesIO(image_data)).convert('RGB')

        return None

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批处理函数"""
        processed_batch = {
            'ids': [item['id'] for item in batch],
            'questions': [item['question'] for item in batch],
            'images': [self.process_image(item['image']) for item in batch],
            'answers': [item['answer'] for item in batch],
            'subquestions': [item['subquestions'] for item in batch],
            'reference_keywords': [item['reference_keywords'] for item in batch],
            'golden_docs': [item['golden_docs'] for item in batch],
            'image_info': [item['image_info'] for item in batch]
        }

        return processed_batch


class MultimodalEvaluationDataset(Dataset):
    """多模态评估数据集"""

    def __init__(self, dataset_name: str, data_path: str):
        self.dataset_name = dataset_name
        self.data_path = data_path

        # 加载多模态评估数据
        if dataset_name == 'mmqa':
            self.data = self.load_mmqa(data_path)
        elif dataset_name == 'visual_reasoning':
            self.data = self.load_visual_reasoning(data_path)
        elif dataset_name == 'multimodal_hotpotqa':
            self.data = self.load_multimodal_hotpotqa(data_path)
        else:
            # 通用多模态数据加载
            self.data = self.load_generic_multimodal(data_path)

    def load_mmqa(self, data_path: str) -> List[Dict[str, Any]]:
        """加载MMQA数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_data.append({
                'id': item.get('id', ''),
                'question': item['question'],
                'image': item.get('image', None),
                'answer': item['answer'],
                'metadata': item.get('metadata', {})
            })
        return processed_data

    def load_visual_reasoning(self, data_path: str) -> List[Dict[str, Any]]:
        """加载视觉推理数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_data.append({
                'id': item.get('id', ''),
                'question': item['question'],
                'image': item['image_path'],
                'answer': item['answer'],
                'reasoning_steps': item.get('reasoning_steps', [])
            })
        return processed_data

    def load_multimodal_hotpotqa(self, data_path: str) -> List[Dict[str, Any]]:
        """加载多模态HotpotQA数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_data.append({
                'id': item['_id'],
                'question': item['question'],
                'image': item.get('image', None),
                'answer': item['answer'],
                'supporting_facts': item.get('supporting_facts', [])
            })
        return processed_data

    def load_generic_multimodal(self, data_path: str) -> List[Dict[str, Any]]:
        """加载通用多模态数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_data.append({
                'id': item.get('id', ''),
                'question': item.get('question', ''),
                'image': item.get('image', None),
                'answer': item.get('answer', ''),
                'context': item.get('context', ''),
                'image_caption': item.get('image_caption', '')
            })
        return processed_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class MultimodalDataPipeline:
    """多模态数据处理管道"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']

    def create_multimodal_sample(self, text_sample: Dict[str, Any],
                                 image_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """将文本样本转换为多模态样本"""
        multimodal_sample = text_sample.copy()

        # 添加图像信息
        if image_info:
            multimodal_sample['image'] = image_info.get('image', None)
            multimodal_sample['image_info'] = image_info

            # 为子问题添加图像搜索查询
            for i, subq in enumerate(multimodal_sample.get('subquestions', [])):
                # 生成图像搜索查询
                image_search_queries = self.generate_image_search_queries(
                    subq['sub_question'],
                    image_info
                )
                subq['image_search_queries'] = image_search_queries

        return multimodal_sample

    def generate_image_search_queries(self, sub_question: str,
                                      image_info: Dict[str, Any]) -> List[str]:
        """为子问题生成图像搜索查询"""
        queries = []

        # 基于图像caption的查询
        if 'caption' in image_info:
            caption = image_info['caption']
            queries.append(caption)

        # 基于图像中检测到的对象的查询
        if 'objects' in image_info:
            objects = image_info['objects']
            for obj in objects[:3]:  # 取前3个对象
                queries.append(obj)

        # 基于子问题和图像信息的组合查询
        question_keywords = sub_question.lower().split()
        if 'keywords' in image_info:
            image_keywords = image_info['keywords']
            # 寻找交集
            common_keywords = set(question_keywords) & set(image_keywords)
            if common_keywords:
                queries.append(' '.join(common_keywords))

        # 去重并限制数量
        unique_queries = list(set(queries))
        return unique_queries[:5]

    def process_multimodal_dataset(self, text_data_path: str,
                                   image_data_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """处理多模态数据集"""
        # 加载文本数据
        with open(text_data_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)

        # 加载图像数据
        image_data = {}
        if image_data_path and Path(image_data_path).exists():
            with open(image_data_path, 'r', encoding='utf-8') as f:
                image_list = json.load(f)
                # 将图像数据转换为字典
                for img_info in image_list:
                    question_id = img_info.get('question_id', img_info.get('id', ''))
                    image_data[question_id] = img_info

        # 处理数据
        multimodal_data = []
        for sample in text_data:
            sample_id = sample.get('id', '')

            # 查找对应的图像信息
            image_info = image_data.get(sample_id, None)

            # 创建多模态样本
            multimodal_sample = self.create_multimodal_sample(sample, image_info)
            multimodal_data.append(multimodal_sample)

        return multimodal_data

    def save_multimodal_data(self, data: List[Dict[str, Any]], output_path: str):
        """保存多模态数据"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved multimodal data to {output_path}")

    def run_multimodal_pipeline(self, text_data_path: str,
                                image_data_path: Optional[str] = None,
                                output_path: str = None) -> List[Dict[str, Any]]:
        """运行多模态数据处理管道"""
        print("Processing multimodal dataset...")

        # 处理数据
        multimodal_data = self.process_multimodal_dataset(text_data_path, image_data_path)

        # 保存数据
        if output_path:
            self.save_multimodal_data(multimodal_data, output_path)

        print(f"Processed {len(multimodal_data)} multimodal samples")
        return multimodal_data


def load_multimodal_train_dataset(data_path: str, processor, config: Dict[str, Any]) -> MultimodalDataset:
    """加载多模态训练数据集"""
    return MultimodalDataset(
        data_path=data_path,
        processor=processor,
        max_length=config['model']['max_length']
    )


def load_multimodal_eval_dataset(dataset_name: str, data_path: str) -> MultimodalEvaluationDataset:
    """加载多模态评估数据集"""
    return MultimodalEvaluationDataset(dataset_name, data_path)