"""
模拟搜索引擎实现 - 用于测试和开发
"""

import random
import re
from typing import List, Dict, Any, Optional
from .search_engine import SearchEngine


class MockSearchEngine(SearchEngine):
    """模拟搜索引擎 - 基础版本"""

    def __init__(self, knowledge_base: List[str] = None, response_delay: float = 0.0):
        self.knowledge_base = knowledge_base or self.create_default_knowledge_base()
        self.response_delay = response_delay
        self.search_history = []

    def create_default_knowledge_base(self) -> List[str]:
        """创建默认知识库"""
        return [
            "Paris is the capital and most populous city of France. It is located in the north-central part of the country.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was built between 1887 and 1889.",
            "Beijing is the capital of China and one of the most populous cities in the world with over 21 million residents.",
            "The Great Wall of China is a series of fortifications built across the northern borders of ancient Chinese states.",
            "Tokyo is the capital of Japan and the most populous metropolitan area in the world.",
            "Mount Fuji is the highest mountain in Japan and is considered a sacred mountain in Japanese culture.",
            "London is the capital and largest city of England and the United Kingdom.",
            "The Thames is a river that flows through southern England, most notably through London.",
            "New York City is the most populous city in the United States, located in the state of New York.",
            "The Statue of Liberty is a neoclassical sculpture on Liberty Island in New York Harbor.",
            "Rome is the capital city of Italy and was the capital of the ancient Roman Empire.",
            "The Colosseum is an ancient amphitheatre in the centre of Rome, Italy.",
            "Berlin is the capital and largest city of Germany.",
            "The Berlin Wall was a barrier that divided Berlin from 1961 to 1989.",
            "Sydney is the largest city in Australia and is known for its Sydney Opera House.",
            "The Sydney Harbour Bridge is a steel through arch bridge across Sydney Harbour.",
            "Madrid is the capital and most populous city of Spain.",
            "The Prado Museum is the main Spanish national art museum, located in central Madrid.",
            "Amsterdam is the capital and most populous city of the Netherlands.",
            "The Anne Frank House is a historic house and biographical museum in Amsterdam.",
        ]

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """模拟搜索功能"""
        import time

        # 记录搜索历史
        self.search_history.append(query)

        # 模拟延迟
        if self.response_delay > 0:
            time.sleep(self.response_delay)

        if not query.strip():
            return []

        # 简单的关键词匹配搜索
        query_words = set(query.lower().split())
        scored_docs = []

        for doc in self.knowledge_base:
            doc_words = set(doc.lower().split())
            # 计算匹配度
            overlap = len(query_words & doc_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored_docs.append((score, doc))

        # 排序并返回top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:top_k]]

    def add_documents(self, documents: List[str]) -> None:
        """添加文档到知识库"""
        self.knowledge_base.extend(documents)

    def get_search_history(self) -> List[str]:
        """获取搜索历史"""
        return self.search_history.copy()

    def clear_history(self):
        """清空搜索历史"""
        self.search_history.clear()

    def statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'knowledge_base_size': len(self.knowledge_base),
            'search_count': len(self.search_history),
            'unique_queries': len(set(self.search_history))
        }


class AdvancedMockSearchEngine(MockSearchEngine):
    """高级模拟搜索引擎"""

    def __init__(self, knowledge_base: List[str] = None,
                 response_delay: float = 0.0,
                 failure_rate: float = 0.0,
                 relevance_noise: float = 0.1):
        super().__init__(knowledge_base, response_delay)
        self.failure_rate = failure_rate  # 搜索失败率
        self.relevance_noise = relevance_noise  # 相关性噪音
        self.entity_knowledge = self.build_entity_knowledge()

    def build_entity_knowledge(self) -> Dict[str, List[str]]:
        """构建实体知识图谱"""
        entities = {
            'capitals': {
                'paris': ['france', 'eiffel tower', 'seine', 'louvre'],
                'beijing': ['china', 'great wall', 'forbidden city'],
                'tokyo': ['japan', 'mount fuji', 'shibuya'],
                'london': ['england', 'uk', 'thames', 'big ben'],
                'rome': ['italy', 'colosseum', 'vatican'],
                'berlin': ['germany', 'brandenburg gate'],
                'madrid': ['spain', 'prado museum'],
                'amsterdam': ['netherlands', 'anne frank']
            },
            'landmarks': {
                'eiffel tower': ['paris', 'france', 'iron', 'tower'],
                'great wall': ['china', 'beijing', 'fortification'],
                'colosseum': ['rome', 'italy', 'amphitheatre'],
                'statue of liberty': ['new york', 'usa', 'liberty island']
            },
            'countries': {
                'france': ['paris', 'eiffel tower', 'europe'],
                'china': ['beijing', 'great wall', 'asia'],
                'japan': ['tokyo', 'mount fuji', 'asia'],
                'italy': ['rome', 'colosseum', 'europe']
            }
        }

        # 扁平化实体知识
        flat_knowledge = {}
        for category, entity_dict in entities.items():
            for entity, related in entity_dict.items():
                flat_knowledge[entity] = related

        return flat_knowledge

    def expand_query(self, query: str) -> List[str]:
        """查询扩展"""
        expanded_queries = [query]
        query_lower = query.lower()

        # 基于实体知识扩展
        for entity, related_terms in self.entity_knowledge.items():
            if entity in query_lower:
                # 添加相关术语
                for term in related_terms[:2]:  # 最多添加2个相关术语
                    expanded_query = query + " " + term
                    expanded_queries.append(expanded_query)

        return expanded_queries

    def search_with_expansion(self, query: str, top_k: int = 3) -> List[str]:
        """带查询扩展的搜索"""
        # 扩展查询
        expanded_queries = self.expand_query(query)

        all_results = []
        for exp_query in expanded_queries:
            results = super().search(exp_query, top_k)
            all_results.extend(results)

        # 去重并重新排序
        unique_results = []
        seen = set()
        for result in all_results:
            if result not in seen:
                unique_results.append(result)
                seen.add(result)

        return unique_results[:top_k]

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """搜索功能（带故障模拟）"""
        # 模拟搜索失败
        if random.random() < self.failure_rate:
            return []

        # 使用扩展搜索
        results = self.search_with_expansion(query, top_k)

        # 添加相关性噪音
        if self.relevance_noise > 0 and results:
            # 随机重排一些结果
            num_to_shuffle = max(1, int(len(results) * self.relevance_noise))
            if num_to_shuffle < len(results):
                # 保持最相关的结果，打乱其他的
                stable_results = results[:-num_to_shuffle]
                noisy_results = results[-num_to_shuffle:]
                random.shuffle(noisy_results)
                results = stable_results + noisy_results

        return results


class DomainMockSearchEngine(MockSearchEngine):
    """领域特定模拟搜索引擎"""

    def __init__(self, domain: str = 'general', **kwargs):
        self.domain = domain
        knowledge_base = self.create_domain_knowledge_base(domain)
        super().__init__(knowledge_base, **kwargs)

    def create_domain_knowledge_base(self, domain: str) -> List[str]:
        """创建领域特定知识库"""
        if domain == 'science':
            return [
                "DNA is a molecule that carries genetic instructions in living organisms.",
                "Photosynthesis is the process by which plants convert sunlight into energy.",
                "The speed of light in vacuum is approximately 299,792,458 meters per second.",
                "Water has the chemical formula H2O, consisting of two hydrogen atoms and one oxygen atom.",
                "The periodic table organizes chemical elements by their atomic number.",
                "Evolution is the process by which species change over time through natural selection.",
                "Gravity is a fundamental force that attracts objects with mass toward each other.",
                "Cells are the basic units of life in all living organisms.",
                "The human body has 206 bones in the adult skeleton.",
                "Atoms consist of protons, neutrons, and electrons."
            ]
        elif domain == 'history':
            return [
                "World War II lasted from 1939 to 1945 and involved most of the world's nations.",
                "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th centuries.",
                "The Roman Empire was one of the largest empires in ancient history.",
                "The American Civil War was fought from 1861 to 1865.",
                "The Industrial Revolution began in Britain in the late 18th century.",
                "Ancient Egypt was known for its pyramids, pharaohs, and hieroglyphic writing.",
                "The Cold War was a period of geopolitical tension between the US and Soviet Union.",
                "The French Revolution began in 1789 and led to major political changes.",
                "Christopher Columbus reached the Americas in 1492.",
                "The Great Wall of China was built over many centuries to protect against invasions."
            ]
        elif domain == 'technology':
            return [
                "Artificial Intelligence refers to computer systems that can perform tasks requiring human intelligence.",
                "The Internet is a global network of interconnected computers.",
                "Machine Learning is a subset of AI that enables computers to learn from musique.",
                "Blockchain is a distributed ledger technology used in cryptocurrencies.",
                "Cloud computing provides on-demand access to computing resources over the internet.",
                "5G is the fifth generation of wireless communication technology.",
                "Virtual Reality creates immersive digital environments for users.",
                "Quantum computing uses quantum mechanical phenomena to process information.",
                "Cybersecurity protects computer systems and networks from digital attacks.",
                "The Internet of Things connects everyday devices to the internet."
            ]
        else:
            return self.create_default_knowledge_base()


class ConfigurableMockSearchEngine(MockSearchEngine):
    """可配置的模拟搜索引擎"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 从配置中提取参数
        knowledge_base = config.get('knowledge_base', None)
        response_delay = config.get('response_delay', 0.0)
        self.max_results = config.get('max_results', 10)
        self.min_relevance_score = config.get('min_relevance_score', 0.1)
        self.enable_fuzzy_matching = config.get('enable_fuzzy_matching', False)

        super().__init__(knowledge_base, response_delay)

        # 加载自定义知识库
        if 'knowledge_file' in config:
            self.load_knowledge_from_file(config['knowledge_file'])

    def load_knowledge_from_file(self, file_path: str):
        """从文件加载知识库"""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        self.knowledge_base.extend(data)
                    elif isinstance(data, dict) and 'documents' in data:
                        self.knowledge_base.extend(data['documents'])
                else:
                    # 纯文本文件，每行一个文档
                    for line in f:
                        line = line.strip()
                        if line:
                            self.knowledge_base.append(line)
        except Exception as e:
            print(f"Failed to load knowledge from {file_path}: {e}")

    def fuzzy_match(self, word1: str, word2: str) -> float:
        """模糊匹配分数"""
        if word1 == word2:
            return 1.0

        # 简单的编辑距离近似
        if len(word1) == 0 or len(word2) == 0:
            return 0.0

        # 检查是否一个词包含另一个
        if word1 in word2 or word2 in word1:
            return 0.8

        # 检查公共前缀
        common_prefix = 0
        for i in range(min(len(word1), len(word2))):
            if word1[i] == word2[i]:
                common_prefix += 1
            else:
                break

        if common_prefix >= 3:  # 至少3个字符的公共前缀
            return 0.6

        return 0.0

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """增强的搜索功能"""
        if not query.strip():
            return []

        # 记录搜索历史
        self.search_history.append(query)

        query_words = [word.lower() for word in query.split()]
        scored_docs = []

        for doc in self.knowledge_base:
            doc_words = [word.lower() for word in doc.split()]
            score = 0.0

            for q_word in query_words:
                best_match = 0.0
                for d_word in doc_words:
                    if self.enable_fuzzy_matching:
                        match_score = self.fuzzy_match(q_word, d_word)
                    else:
                        match_score = 1.0 if q_word == d_word else 0.0
                    best_match = max(best_match, match_score)
                score += best_match

            # 标准化分数
            if len(query_words) > 0:
                score = score / len(query_words)

            if score >= self.min_relevance_score:
                scored_docs.append((score, doc))

        # 排序并返回结果
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        top_k = min(top_k, self.max_results)

        return [doc for score, doc in scored_docs[:top_k]]


class InteractiveMockSearchEngine(MockSearchEngine):
    """交互式模拟搜索引擎"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_feedback = {}  # 存储用户反馈
        self.query_improvements = {}  # 存储查询改进建议

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """搜索并收集反馈"""
        results = super().search(query, top_k)

        # 检查是否有该查询的改进建议
        if query in self.query_improvements:
            print(f"Suggestion: Try searching for '{self.query_improvements[query]}'")

        return results

    def provide_feedback(self, query: str, result_index: int, is_relevant: bool):
        """提供结果相关性反馈"""
        if query not in self.user_feedback:
            self.user_feedback[query] = {}

        self.user_feedback[query][result_index] = is_relevant

    def suggest_query_improvement(self, original_query: str, improved_query: str):
        """添加查询改进建议"""
        self.query_improvements[original_query] = improved_query

    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈摘要"""
        total_feedback = 0
        positive_feedback = 0

        for query_feedback in self.user_feedback.values():
            for is_relevant in query_feedback.values():
                total_feedback += 1
                if is_relevant:
                    positive_feedback += 1

        relevance_rate = positive_feedback / total_feedback if total_feedback > 0 else 0.0

        return {
            'total_feedback': total_feedback,
            'positive_feedback': positive_feedback,
            'relevance_rate': relevance_rate,
            'queries_with_feedback': len(self.user_feedback)
        }


def create_mock_search_engine(config: Dict[str, Any]) -> MockSearchEngine:
    """创建模拟搜索引擎"""
    engine_type = config.get('type', 'basic')

    if engine_type == 'basic':
        return MockSearchEngine(
            knowledge_base=config.get('knowledge_base'),
            response_delay=config.get('response_delay', 0.0)
        )
    elif engine_type == 'advanced':
        return AdvancedMockSearchEngine(
            knowledge_base=config.get('knowledge_base'),
            response_delay=config.get('response_delay', 0.0),
            failure_rate=config.get('failure_rate', 0.0),
            relevance_noise=config.get('relevance_noise', 0.1)
        )
    elif engine_type == 'domain':
        return DomainMockSearchEngine(
            domain=config.get('domain', 'general'),
            response_delay=config.get('response_delay', 0.0)
        )
    elif engine_type == 'configurable':
        return ConfigurableMockSearchEngine(config)
    elif engine_type == 'interactive':
        return InteractiveMockSearchEngine(
            knowledge_base=config.get('knowledge_base'),
            response_delay=config.get('response_delay', 0.0)
        )
    else:
        raise ValueError(f"Unknown mock search engine type: {engine_type}")