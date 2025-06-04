"""
奖励模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoModel, AutoTokenizer


class RewardModel(nn.Module):
    """基础奖励模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """前向传播"""
        raise NotImplementedError

    def compute_reward(self, *args, **kwargs) -> float:
        """计算奖励"""
        raise NotImplementedError


class AnswerRewardModel(RewardModel):
    """答案奖励模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('reward_model_name', 'sentence-transformers/all-MiniLM-L6-v2')

        # 加载预训练模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        except:
            # 如果加载失败，使用简单的线性层
            self.tokenizer = None
            self.model = None
            self.simple_embedding = nn.Embedding(10000, 256)
            self.reward_head = nn.Linear(256, 1)

    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本"""
        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)
        else:
            # 简单编码
            tokens = text.split()[:100]  # 截断到100个token
            token_ids = [hash(token) % 10000 for token in tokens]
            if not token_ids:
                token_ids = [0]

            token_tensor = torch.tensor(token_ids)
            embeddings = self.simple_embedding(token_tensor)
            return embeddings.mean(dim=0, keepdim=True)

    def compute_similarity(self, pred_answer: str, gt_answer: str) -> float:
        """计算答案相似度"""
        pred_emb = self.encode_text(pred_answer)
        gt_emb = self.encode_text(gt_answer)

        # 计算余弦相似度
        similarity = F.cosine_similarity(pred_emb, gt_emb).item()
        return max(0.0, similarity)  # 确保非负

    def compute_reward(self, pred_answer: str, gt_answer: str, format_correct: bool = True) -> float:
        """计算答案奖励"""
        if not format_correct:
            return 0.0

        if not pred_answer or not gt_answer:
            return 0.0

        # 计算基于相似度的奖励
        similarity_reward = self.compute_similarity(pred_answer, gt_answer)

        # 计算基于精确匹配的奖励
        exact_match_reward = 1.0 if pred_answer.strip().lower() == gt_answer.strip().lower() else 0.0

        # 组合奖励
        total_reward = 0.7 * similarity_reward + 0.3 * exact_match_reward

        return total_reward


class SearchQualityRewardModel(RewardModel):
    """搜索质量奖励模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.relevance_threshold = config.get('relevance_threshold', 0.3)

    def compute_search_relevance(self, query: str, retrieved_docs: List[str],
                                 question: str) -> float:
        """计算搜索相关性"""
        if not query or not retrieved_docs:
            return 0.0

        # 简单的相关性评估：检查查询词在检索文档中的出现
        query_words = set(query.lower().split())
        question_words = set(question.lower().split())

        relevance_scores = []
        for doc in retrieved_docs:
            doc_words = set(doc.lower().split())

            # 计算查询词覆盖率
            query_coverage = len(query_words & doc_words) / len(query_words) if query_words else 0.0

            # 计算问题词覆盖率
            question_coverage = len(question_words & doc_words) / len(question_words) if question_words else 0.0

            # 组合分数
            score = 0.6 * query_coverage + 0.4 * question_coverage
            relevance_scores.append(score)

        return max(relevance_scores) if relevance_scores else 0.0

    def compute_reward(self, query: str, retrieved_docs: List[str],
                       question: str, expected_info: str = "") -> float:
        """计算搜索质量奖励"""
        # 基础相关性奖励
        relevance_reward = self.compute_search_relevance(query, retrieved_docs, question)

        # 如果有期望信息，计算信息覆盖度
        coverage_reward = 0.0
        if expected_info and retrieved_docs:
            expected_words = set(expected_info.lower().split())
            for doc in retrieved_docs:
                doc_words = set(doc.lower().split())
                coverage = len(expected_words & doc_words) / len(expected_words) if expected_words else 0.0
                coverage_reward = max(coverage_reward, coverage)

        # 组合奖励
        total_reward = 0.7 * relevance_reward + 0.3 * coverage_reward

        return total_reward


class ProcessRewardModel(RewardModel):
    """过程奖励模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.step_weight = config.get('step_weight', 0.1)
        self.consistency_weight = config.get('consistency_weight', 0.2)

    def evaluate_reasoning_step(self, step: str, context: str) -> float:
        """评估推理步骤质量"""
        if not step:
            return 0.0

        # 评估步骤的逻辑性
        logic_indicators = [
            'because', 'therefore', 'since', 'as a result',
            'consequently', 'thus', 'hence', 'so'
        ]

        logic_score = sum(1 for indicator in logic_indicators if indicator in step.lower())
        logic_score = min(logic_score / 3.0, 1.0)  # 标准化到[0,1]

        # 评估步骤的信息量
        info_score = min(len(step.split()) / 20.0, 1.0)  # 基于长度的信息量估计

        # 评估与上下文的一致性
        consistency_score = 0.5  # 简化评估
        if context:
            context_words = set(context.lower().split())
            step_words = set(step.lower().split())
            overlap = len(context_words & step_words)
            consistency_score = min(overlap / len(step_words) if step_words else 0.0, 1.0)

        # 组合分数
        total_score = 0.4 * logic_score + 0.3 * info_score + 0.3 * consistency_score

        return total_score

    def compute_reward(self, reasoning_steps: List[str], context: str = "") -> float:
        """计算过程奖励"""
        if not reasoning_steps:
            return 0.0

        step_scores = []
        for step in reasoning_steps:
            score = self.evaluate_reasoning_step(step, context)
            step_scores.append(score)

        # 计算平均步骤质量
        avg_step_quality = sum(step_scores) / len(step_scores)

        # 奖励连贯的多步推理
        coherence_bonus = 0.0
        if len(reasoning_steps) > 1:
            coherence_bonus = 0.1 * min(len(reasoning_steps) / 5.0, 1.0)

        total_reward = avg_step_quality + coherence_bonus

        return min(total_reward, 1.0)


class CompositeRewardModel(RewardModel):
    """组合奖励模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 初始化各个子奖励模型
        self.answer_model = AnswerRewardModel(config)
        self.search_model = SearchQualityRewardModel(config)
        self.process_model = ProcessRewardModel(config)

        # 奖励权重
        self.answer_weight = config.get('answer_weight', 0.5)
        self.search_weight = config.get('search_weight', 0.3)
        self.process_weight = config.get('process_weight', 0.2)

    def compute_composite_reward(self,
                                 pred_answer: str,
                                 gt_answer: str,
                                 search_queries: List[str],
                                 retrieved_docs: List[List[str]],
                                 reasoning_steps: List[str],
                                 question: str,
                                 format_correct: bool = True) -> Dict[str, float]:
        """计算组合奖励"""

        # 答案奖励
        answer_reward = self.answer_model.compute_reward(pred_answer, gt_answer, format_correct)

        # 搜索奖励
        search_reward = 0.0
        if search_queries and retrieved_docs:
            search_rewards = []
            for query, docs in zip(search_queries, retrieved_docs):
                reward = self.search_model.compute_reward(query, docs, question)
                search_rewards.append(reward)
            search_reward = sum(search_rewards) / len(search_rewards) if search_rewards else 0.0

        # 过程奖励
        process_reward = self.process_model.compute_reward(reasoning_steps)

        # 组合总奖励
        total_reward = (self.answer_weight * answer_reward +
                        self.search_weight * search_reward +
                        self.process_weight * process_reward)

        return {
            'answer_reward': answer_reward,
            'search_reward': search_reward,
            'process_reward': process_reward,
            'total_reward': total_reward
        }


class LearnableRewardModel(RewardModel):
    """可学习的奖励模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 特征提取器
        self.input_dim = config.get('input_dim', 768)
        self.hidden_dim = config.get('hidden_dim', 256)

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 奖励头
        self.reward_head = nn.Linear(self.hidden_dim, 1)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_features(self,
                         pred_answer: str,
                         gt_answer: str,
                         search_info: Dict[str, Any]) -> torch.Tensor:
        """提取特征"""
        # 这里需要实现特征提取逻辑
        # 简化版本：使用随机特征
        features = torch.randn(1, self.input_dim)
        return features

    def forward(self,
                pred_answer: str,
                gt_answer: str,
                search_info: Dict[str, Any]) -> torch.Tensor:
        """前向传播"""
        features = self.extract_features(pred_answer, gt_answer, search_info)
        hidden = self.feature_extractor(features)
        reward = self.reward_head(hidden)
        return torch.sigmoid(reward)  # 确保奖励在[0,1]范围内

    def compute_reward(self,
                       pred_answer: str,
                       gt_answer: str,
                       search_info: Dict[str, Any]) -> float:
        """计算奖励"""
        with torch.no_grad():
            reward_tensor = self.forward(pred_answer, gt_answer, search_info)
            return reward_tensor.item()

    def train_step(self,
                   batch_data: List[Dict[str, Any]],
                   optimizer: torch.optim.Optimizer) -> float:
        """训练步骤"""
        self.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for data in batch_data:
            pred_reward = self.forward(
                data['pred_answer'],
                data['gt_answer'],
                data['search_info']
            )

            # 使用真实奖励作为监督信号
            target_reward = torch.tensor([[data['target_reward']]], dtype=torch.float32)

            loss = F.mse_loss(pred_reward, target_reward)
            total_loss += loss.item()

            loss.backward()

        optimizer.step()

        return total_loss / len(batch_data)


def create_reward_model(config: Dict[str, Any], model_type: str = 'composite') -> RewardModel:
    """创建奖励模型"""

    if model_type == 'answer':
        return AnswerRewardModel(config)
    elif model_type == 'search':
        return SearchQualityRewardModel(config)
    elif model_type == 'process':
        return ProcessRewardModel(config)
    elif model_type == 'composite':
        return CompositeRewardModel(config)
    elif model_type == 'learnable':
        return LearnableRewardModel(config)
    else:
        raise ValueError(f"Unknown reward model type: {model_type}")


# 奖励模型工厂
class RewardModelFactory:
    """奖励模型工厂"""

    _models = {
        'answer': AnswerRewardModel,
        'search': SearchQualityRewardModel,
        'process': ProcessRewardModel,
        'composite': CompositeRewardModel,
        'learnable': LearnableRewardModel
    }

    @classmethod
    def create(cls, model_type: str, config: Dict[str, Any]) -> RewardModel:
        """创建奖励模型"""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")

        return cls._models[model_type](config)

    @classmethod
    def register(cls, name: str, model_class: type):
        """注册新的奖励模型类"""
        cls._models[name] = model_class

    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有可用的模型类型"""
        return list(cls._models.keys())