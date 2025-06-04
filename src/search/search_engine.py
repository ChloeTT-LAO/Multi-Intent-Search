"""
搜索引擎接口和实现
"""

import json
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SearchEngine(ABC):
    """搜索引擎抽象基类"""

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """搜索相关文档"""
        pass

    @abstractmethod
    def add_documents(self, documents: List[str]) -> None:
        """添加文档到索引"""
        pass


class TFIDFSearchEngine(SearchEngine):
    """基于TF-IDF的搜索引擎"""

    def __init__(self, documents: List[str] = None, index_path: str = None):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=50000,
            ngram_range=(1, 2),
            lowercase=True
        )
        self.documents = []
        self.doc_vectors = None

        if documents:
            self.add_documents(documents)
        elif index_path and Path(index_path).exists():
            self.load_index(index_path)

    def add_documents(self, documents: List[str]) -> None:
        """添加文档到索引"""
        self.documents.extend(documents)
        print(f"Indexing {len(self.documents)} documents...")

        # 过滤空文档
        valid_docs = [doc for doc in self.documents if doc.strip()]

        if valid_docs:
            self.doc_vectors = self.vectorizer.fit_transform(valid_docs)
            self.documents = valid_docs
            print(f"Successfully indexed {len(self.documents)} documents")
        else:
            print("No valid documents to index")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """搜索相关文档"""
        if not query.strip() or self.doc_vectors is None:
            return []

        try:
            # 向量化查询
            query_vector = self.vectorizer.transform([query])

            # 计算相似度
            similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

            # 获取top_k相关文档
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # 过滤低相关度的结果
            relevant_docs = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # 最小相关度阈值
                    relevant_docs.append(self.documents[idx])

            return relevant_docs

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def save_index(self, save_path: str) -> None:
        """保存索引"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存文档
        with open(save_path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        # 保存向量化器
        with open(save_path / "vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)

        # 保存文档向量
        with open(save_path / "doc_vectors.pkl", 'wb') as f:
            pickle.dump(self.doc_vectors, f)

        print(f"Index saved to {save_path}")

    def load_index(self, load_path: str) -> None:
        """加载索引"""
        load_path = Path(load_path)

        # 加载文档
        with open(load_path / "documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        # 加载向量化器
        with open(load_path / "vectorizer.pkl", 'rb') as f:
            self.vectorizer = pickle.load(f)

        # 加载文档向量
        with open(load_path / "doc_vectors.pkl", 'rb') as f:
            self.doc_vectors = pickle.load(f)

        print(f"Index loaded from {load_path}, {len(self.documents)} documents")


class WikipediaSearchEngine(TFIDFSearchEngine):
    """Wikipedia搜索引擎"""

    def __init__(self, wiki_dump_path: str, index_path: str = None):
        self.wiki_dump_path = Path(wiki_dump_path)
        super().__init__(index_path=index_path)

        if not self.documents:  # 如果没有加载到索引，则构建
            self.build_index_from_wiki_dump()

    def build_index_from_wiki_dump(self) -> None:
        """从Wikipedia dump构建索引"""
        print(f"Building index from Wikipedia dump: {self.wiki_dump_path}")

        documents = []

        if self.wiki_dump_path.is_file():
            # 单个文件
            documents = self._parse_wiki_file(self.wiki_dump_path)
        elif self.wiki_dump_path.is_dir():
            # 目录中的多个文件
            for file_path in self.wiki_dump_path.rglob("*.json"):
                documents.extend(self._parse_wiki_file(file_path))

        if documents:
            self.add_documents(documents)
        else:
            print("No documents found in Wikipedia dump")

    def _parse_wiki_file(self, file_path: Path) -> List[str]:
        """解析Wikipedia文件"""
        documents = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        article = json.loads(line.strip())

                        # 提取标题和文本
                        title = article.get('title', '')
                        text = article.get('text', '')

                        if title and text:
                            # 合并标题和文本
                            full_text = f"{title}. {text}"
                            # 限制文档长度
                            if len(full_text) > 1000:
                                full_text = full_text[:1000] + "..."
                            documents.append(full_text)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

        return documents


class MockSearchEngine(SearchEngine):
    """模拟搜索引擎（用于测试）"""

    def __init__(self, knowledge_base: List[str] = None):
        if knowledge_base is None:
            # 默认知识库
            knowledge_base = [
                "Beijing is the capital of China and has a population of over 21 million people.",
                "Paris is the capital of France and is known for the Eiffel Tower.",
                "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
                "The Great Wall of China is a fortification built across northern China.",
                "Tokyo is the capital of Japan and one of the most populous cities in the world.",
                "London is the capital of the United Kingdom and England.",
                "The Thames is a river that flows through London.",
                "Mount Fuji is the highest mountain in Japan.",
                "The Louvre Museum is located in Paris and houses the Mona Lisa.",
                "The Forbidden City is a palace complex in Beijing, China."
            ]

        self.search_engine = TFIDFSearchEngine(knowledge_base)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """搜索相关文档"""
        return self.search_engine.search(query, top_k)

    def add_documents(self, documents: List[str]) -> None:
        """添加文档"""
        self.search_engine.add_documents(documents)


class HybridSearchEngine(SearchEngine):
    """混合搜索引擎（结合多个搜索源）"""

    def __init__(self, search_engines: List[SearchEngine], weights: List[float] = None):
        self.search_engines = search_engines
        self.weights = weights or [1.0] * len(search_engines)

        if len(self.weights) != len(self.search_engines):
            raise ValueError("Number of weights must match number of search engines")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """从多个搜索引擎搜索并合并结果"""
        all_results = []

        for engine, weight in zip(self.search_engines, self.weights):
            results = engine.search(query, top_k)
            # 为每个结果添加权重信息
            weighted_results = [(doc, weight) for doc in results]
            all_results.extend(weighted_results)

        # 去重并按权重排序
        unique_docs = {}
        for doc, weight in all_results:
            if doc in unique_docs:
                unique_docs[doc] += weight
            else:
                unique_docs[doc] = weight

        # 按权重排序并返回top_k
        sorted_docs = sorted(unique_docs.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:top_k]]

    def add_documents(self, documents: List[str]) -> None:
        """向所有搜索引擎添加文档"""
        for engine in self.search_engines:
            engine.add_documents(documents)


def create_search_engine(config: Dict[str, Any]) -> SearchEngine:
    """根据配置创建搜索引擎"""
    search_config = config['search']
    engine_type = search_config['engine_type']

    if engine_type == 'mock':
        return MockSearchEngine()

    elif engine_type == 'wiki':
        wiki_dump_path = search_config['wiki_dump_path']
        index_path = search_config.get('index_path', None)
        return WikipediaSearchEngine(wiki_dump_path, index_path)

    elif engine_type == 'tfidf':
        # 基础TF-IDF搜索引擎
        documents_path = search_config.get('documents_path', None)
        if documents_path and Path(documents_path).exists():
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            return TFIDFSearchEngine(documents)
        else:
            return TFIDFSearchEngine()

    elif engine_type == 'hybrid':
        # 混合搜索引擎
        engines = []
        weights = search_config.get('weights', [])

        for i, sub_config in enumerate(search_config['engines']):
            sub_engine = create_search_engine({'search': sub_config})
            engines.append(sub_engine)

        return HybridSearchEngine(engines, weights if weights else None)

    else:
        raise ValueError(f"Unknown search engine type: {engine_type}")


# 搜索引擎工具函数
def build_knowledge_base_from_datasets(datasets_paths: List[str]) -> List[str]:
    """从多个数据集构建知识库"""
    documents = []

    for dataset_path in datasets_paths:
        path = Path(dataset_path)

        if not path.exists():
            continue

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取文档文本
        for item in data:
            if isinstance(item, dict):
                # 提取各种可能的文本字段
                text_fields = ['text', 'passage', 'context', 'content', 'answer']
                for field in text_fields:
                    if field in item and item[field]:
                        documents.append(item[field])

    return documents