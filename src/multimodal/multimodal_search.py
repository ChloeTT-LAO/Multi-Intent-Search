"""
改进版多模态搜索引擎实现
"""

import base64
import io
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class MultimodalSearchEngine(ABC):
    """多模态搜索引擎抽象基类"""

    @abstractmethod
    def search_multimodal(self, text_query: str = None, image_query: Union[str, Image.Image] = None,
                          top_k: int = 3) -> List[Dict[str, Any]]:
        """多模态搜索"""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档到索引"""
        pass

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """兼容原有接口的文本搜索"""
        results = self.search_multimodal(text_query=query, top_k=top_k)
        return [result.get('text', '') for result in results]


class CachedMultimodalSearchEngine(MultimodalSearchEngine):
    """带缓存的多模态搜索引擎基类"""

    def __init__(self, cache_dir: str = "./cache/multimodal_search"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存设置
        self.query_cache = {}
        self.image_embedding_cache = {}
        self.text_embedding_cache = {}
        self.max_cache_size = 10000
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    def _get_cache_key(self, text_query: str = None, image_query: Union[str, Image.Image] = None) -> str:
        """生成缓存键"""
        key_parts = []
        if text_query:
            key_parts.append(f"text:{hash(text_query)}")
        if image_query:
            if isinstance(image_query, str):
                key_parts.append(f"image_path:{hash(image_query)}")
            else:
                # 对PIL图像生成哈希
                img_bytes = io.BytesIO()
                image_query.save(img_bytes, format='PNG')
                key_parts.append(f"image_data:{hash(img_bytes.getvalue())}")
        return "_".join(key_parts)

    def _manage_cache_size(self, cache_dict: dict):
        """管理缓存大小"""
        if len(cache_dict) > self.max_cache_size:
            # 删除最旧的25%缓存项
            items_to_remove = len(cache_dict) // 4
            keys_to_remove = list(cache_dict.keys())[:items_to_remove]
            for key in keys_to_remove:
                del cache_dict[key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hit_count + self.cache_miss_count
        hit_rate = self.cache_hit_count / total_requests if total_requests > 0 else 0

        return {
            'cache_hit_count': self.cache_hit_count,
            'cache_miss_count': self.cache_miss_count,
            'hit_rate': hit_rate,
            'query_cache_size': len(self.query_cache),
            'image_embedding_cache_size': len(self.image_embedding_cache),
            'text_embedding_cache_size': len(self.text_embedding_cache)
        }


class AdvancedImageTextSearchEngine(CachedMultimodalSearchEngine):
    """高级图像-文本搜索引擎"""

    def __init__(self, documents: List[Dict[str, Any]] = None,
                 cache_dir: str = "./cache/multimodal_search",
                 model_name: str = "ViT-B/32",
                 device: str = "auto"):
        super().__init__(cache_dir)

        self.documents = documents or []
        self.model_name = model_name
        self.device = self._setup_device(device)

        # 初始化模型
        self.clip_model = None
        self.clip_preprocess = None
        self.text_search_engine = None

        # 预计算的embeddings
        self.image_embeddings = []
        self.text_embeddings = []

        # 异步支持
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 初始化组件
        self._setup_models()
        if self.documents:
            self._process_documents()

    def _setup_device(self, device: str) -> str:
        """设置设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _setup_models(self):
        """设置模型"""
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load(self.model_name, device=self.device)
            logger.info(f"CLIP model loaded: {self.model_name} on {self.device}")
        except ImportError:
            logger.warning("CLIP not available, falling back to text-only search")
            self._setup_text_only_fallback()
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self._setup_text_only_fallback()

    def _setup_text_only_fallback(self):
        """设置纯文本搜索回退"""
        from ..search.search_engine import TFIDFSearchEngine
        texts = [doc.get('text', '') for doc in self.documents if doc.get('text')]
        if texts:
            self.text_search_engine = TFIDFSearchEngine(texts)

    def _process_documents(self):
        """处理文档，预计算embeddings"""
        logger.info(f"Processing {len(self.documents)} documents...")

        if self.clip_model is None:
            return

        # 批处理计算embeddings
        batch_size = 32

        # 处理文本embeddings
        texts = []
        for doc in self.documents:
            text = doc.get('text', '')
            if text:
                texts.append(text)
            else:
                texts.append("")  # 占位符

        if texts:
            self.text_embeddings = self._compute_text_embeddings_batch(texts, batch_size)

        # 处理图像embeddings
        images = []
        for doc in self.documents:
            image = doc.get('image')
            if image:
                processed_image = self._process_image(image)
                images.append(processed_image)
            else:
                images.append(None)

        if any(img is not None for img in images):
            self.image_embeddings = self._compute_image_embeddings_batch(images, batch_size)

        logger.info("Document processing completed")

    def _compute_text_embeddings_batch(self, texts: List[str], batch_size: int) -> List[np.ndarray]:
        """批量计算文本embeddings"""
        embeddings = []

        try:
            import clip

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # 过滤空文本
                valid_texts = [text if text else "empty" for text in batch_texts]

                tokens = clip.tokenize(valid_texts, truncate=True).to(self.device)

                with torch.no_grad():
                    batch_embeddings = self.clip_model.encode_text(tokens)
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                    embeddings.extend(batch_embeddings.cpu().numpy())

        except Exception as e:
            logger.error(f"Error computing text embeddings: {e}")
            # 返回零向量作为回退
            embeddings = [np.zeros(512) for _ in texts]

        return embeddings

    def _compute_image_embeddings_batch(self, images: List[Optional[Image.Image]],
                                        batch_size: int) -> List[Optional[np.ndarray]]:
        """批量计算图像embeddings"""
        embeddings = []

        try:
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_tensors = []
                batch_indices = []

                # 准备有效图像
                for j, img in enumerate(batch_images):
                    if img is not None:
                        try:
                            tensor = self.clip_preprocess(img).unsqueeze(0)
                            batch_tensors.append(tensor)
                            batch_indices.append(i + j)
                        except Exception as e:
                            logger.warning(f"Failed to process image {i + j}: {e}")

                # 计算embeddings
                if batch_tensors:
                    batch_tensor = torch.cat(batch_tensors).to(self.device)

                    with torch.no_grad():
                        batch_embeddings = self.clip_model.encode_image(batch_tensor)
                        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                        batch_embeddings = batch_embeddings.cpu().numpy()

                    # 分配embeddings到正确位置
                    for idx, embedding in zip(batch_indices, batch_embeddings):
                        while len(embeddings) <= idx:
                            embeddings.append(None)
                        embeddings[idx] = embedding

                # 填充空位置
                while len(embeddings) < i + len(batch_images):
                    embeddings.append(None)

        except Exception as e:
            logger.error(f"Error computing image embeddings: {e}")
            embeddings = [None for _ in images]

        return embeddings

    def _process_image(self, image_input: Union[str, Image.Image, dict]) -> Optional[Image.Image]:
        """处理图像输入"""
        try:
            if image_input is None:
                return None

            if isinstance(image_input, str):
                if image_input.startswith('data:image'):
                    # Base64编码
                    header, data = image_input.split(',', 1)
                    image_data = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_data))
                elif image_input.startswith('http'):
                    # URL (这里需要添加下载逻辑)
                    logger.warning(f"URL image loading not implemented: {image_input}")
                    return None
                else:
                    # 文件路径
                    image = Image.open(image_input)
            elif isinstance(image_input, dict):
                # 图像信息字典
                if 'path' in image_input:
                    image = Image.open(image_input['path'])
                elif 'data' in image_input:
                    image_data = base64.b64decode(image_input['data'])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    return None
            else:
                image = image_input

            return image.convert('RGB') if image else None

        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return None

    def _encode_query(self, text_query: str = None,
                      image_query: Union[str, Image.Image] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """编码查询"""
        text_embedding = None
        image_embedding = None

        # 检查缓存
        cache_key = self._get_cache_key(text_query, image_query)
        if cache_key in self.query_cache:
            self.cache_hit_count += 1
            return self.query_cache[cache_key]

        self.cache_miss_count += 1

        if self.clip_model is None:
            return text_embedding, image_embedding

        try:
            import clip

            # 编码文本查询
            if text_query:
                text_key = f"text:{hash(text_query)}"
                if text_key in self.text_embedding_cache:
                    text_embedding = self.text_embedding_cache[text_key]
                else:
                    tokens = clip.tokenize([text_query], truncate=True).to(self.device)
                    with torch.no_grad():
                        text_embedding = self.clip_model.encode_text(tokens)
                        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                        text_embedding = text_embedding.cpu().numpy()[0]

                    self.text_embedding_cache[text_key] = text_embedding
                    self._manage_cache_size(self.text_embedding_cache)

            # 编码图像查询
            if image_query:
                processed_image = self._process_image(image_query)
                if processed_image:
                    img_key = f"image:{hash(str(image_query))}"
                    if img_key in self.image_embedding_cache:
                        image_embedding = self.image_embedding_cache[img_key]
                    else:
                        image_tensor = self.clip_preprocess(processed_image).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            image_embedding = self.clip_model.encode_image(image_tensor)
                            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                            image_embedding = image_embedding.cpu().numpy()[0]

                        self.image_embedding_cache[img_key] = image_embedding
                        self._manage_cache_size(self.image_embedding_cache)

        except Exception as e:
            logger.error(f"Error encoding query: {e}")

        # 缓存结果
        result = (text_embedding, image_embedding)
        self.query_cache[cache_key] = result
        self._manage_cache_size(self.query_cache)

        return result

    def search_multimodal(self, text_query: str = None, image_query: Union[str, Image.Image] = None,
                          top_k: int = 3, fusion_strategy: str = "weighted_sum") -> List[Dict[str, Any]]:
        """多模态搜索"""
        if not text_query and not image_query:
            return []

        if self.clip_model is None:
            # 回退到纯文本搜索
            if text_query and self.text_search_engine:
                texts = self.text_search_engine.search(text_query, top_k)
                return [{'text': text, 'image': None, 'score': 1.0, 'doc_index': i}
                        for i, text in enumerate(texts)]
            return []

        # 编码查询
        text_embedding, image_embedding = self._encode_query(text_query, image_query)

        # 计算相似度
        similarities = self._compute_similarities(text_embedding, image_embedding, fusion_strategy)

        # 排序并返回top_k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # 最小相关度阈值
                result = {
                    'text': self.documents[idx].get('text', ''),
                    'image': self.documents[idx].get('image', None),
                    'score': float(similarities[idx]),
                    'doc_index': idx,
                    'metadata': self.documents[idx].get('metadata', {})
                }
                results.append(result)

        return results

    def _compute_similarities(self, text_embedding: Optional[np.ndarray],
                              image_embedding: Optional[np.ndarray],
                              fusion_strategy: str = "weighted_sum") -> np.ndarray:
        """计算相似度"""
        num_docs = len(self.documents)
        similarities = np.zeros(num_docs)

        if fusion_strategy == "weighted_sum":
            text_weight = 0.6 if text_embedding is not None else 0.0
            image_weight = 0.4 if image_embedding is not None else 0.0

            # 重新标准化权重
            total_weight = text_weight + image_weight
            if total_weight > 0:
                text_weight /= total_weight
                image_weight /= total_weight

        elif fusion_strategy == "max":
            text_weight = 1.0 if text_embedding is not None else 0.0
            image_weight = 1.0 if image_embedding is not None else 0.0

        else:  # average
            text_weight = 0.5
            image_weight = 0.5

        # 计算文本相似度
        if text_embedding is not None and self.text_embeddings:
            text_sims = np.array([
                np.dot(text_embedding, doc_emb) if doc_emb is not None else 0.0
                for doc_emb in self.text_embeddings
            ])

            if fusion_strategy == "max":
                similarities = np.maximum(similarities, text_sims)
            else:
                similarities += text_weight * text_sims

        # 计算图像相似度
        if image_embedding is not None and self.image_embeddings:
            image_sims = np.array([
                np.dot(image_embedding, doc_emb) if doc_emb is not None else 0.0
                for doc_emb in self.image_embeddings
            ])

            if fusion_strategy == "max":
                similarities = np.maximum(similarities, image_sims)
            else:
                similarities += image_weight * image_sims

        return similarities

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档并更新索引"""
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents...")

        self.documents.extend(documents)

        # 重新计算embeddings（增量更新）
        if self.clip_model is not None:
            # 为新文档计算embeddings
            new_texts = [doc.get('text', '') for doc in documents]
            new_images = [doc.get('image') for doc in documents]

            if new_texts:
                new_text_embeddings = self._compute_text_embeddings_batch(new_texts, 32)
                self.text_embeddings.extend(new_text_embeddings)

            if any(img is not None for img in new_images):
                processed_images = [self._process_image(img) for img in new_images]
                new_image_embeddings = self._compute_image_embeddings_batch(processed_images, 32)
                self.image_embeddings.extend(new_image_embeddings)

        # 清空相关缓存
        self.query_cache.clear()

        logger.info("Documents added successfully")

    async def search_multimodal_async(self, text_query: str = None,
                                      image_query: Union[str, Image.Image] = None,
                                      top_k: int = 3) -> List[Dict[str, Any]]:
        """异步多模态搜索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.search_multimodal, text_query, image_query, top_k
        )

    def save_index(self, save_path: str):
        """保存索引"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存文档
        with open(save_path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2, default=str)

        # 保存embeddings
        if self.text_embeddings:
            np.save(save_path / "text_embeddings.npy", np.array(self.text_embeddings))

        if self.image_embeddings:
            # 处理None值
            valid_embeddings = []
            valid_indices = []
            for i, emb in enumerate(self.image_embeddings):
                if emb is not None:
                    valid_embeddings.append(emb)
                    valid_indices.append(i)

            if valid_embeddings:
                np.save(save_path / "image_embeddings.npy", np.array(valid_embeddings))
                np.save(save_path / "image_indices.npy", np.array(valid_indices))

        # 保存配置
        config = {
            'model_name': self.model_name,
            'device': self.device,
            'num_documents': len(self.documents)
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Index saved to {save_path}")

    def load_index(self, load_path: str):
        """加载索引"""
        load_path = Path(load_path)

        # 加载文档
        with open(load_path / "documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        # 加载embeddings
        if (load_path / "text_embeddings.npy").exists():
            self.text_embeddings = list(np.load(load_path / "text_embeddings.npy"))

        if (load_path / "image_embeddings.npy").exists():
            valid_embeddings = np.load(load_path / "image_embeddings.npy")
            valid_indices = np.load(load_path / "image_indices.npy")

            # 重建完整的image_embeddings列表
            self.image_embeddings = [None] * len(self.documents)
            for emb, idx in zip(valid_embeddings, valid_indices):
                self.image_embeddings[idx] = emb

        logger.info(f"Index loaded from {load_path}, {len(self.documents)} documents")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'num_documents': len(self.documents),
            'num_text_embeddings': len([e for e in self.text_embeddings if e is not None]),
            'num_image_embeddings': len([e for e in self.image_embeddings if e is not None]),
            'model_name': self.model_name,
            'device': self.device,
        }
        stats.update(self.get_cache_stats())
        return stats

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def create_multimodal_search_engine(config: Dict[str, Any]) -> MultimodalSearchEngine:
    """创建多模态搜索引擎工厂函数"""
    search_config = config['search']
    engine_type = search_config.get('engine_type', 'advanced_imagetext')

    if engine_type == 'advanced_imagetext':
        return AdvancedImageTextSearchEngine(
            cache_dir=search_config.get('cache_dir', './cache/multimodal_search'),
            model_name=search_config.get('clip_model', 'ViT-B/32'),
            device=search_config.get('device', 'auto')
        )
    else:
        raise ValueError(f"Unknown multimodal search engine type: {engine_type}")