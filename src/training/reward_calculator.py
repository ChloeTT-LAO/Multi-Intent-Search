"""
StepSearch奖励计算器
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class StepSearchRewardCalculator:
    """StepSearch奖励计算器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_config = config['reward']
        self.gamma_key = self.reward_config['gamma_key']
        self.redundancy_threshold = self.reward_config['redundancy_threshold']

        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )

        # 存储每个episode的记忆状态
        self.episode_memories = defaultdict(lambda: defaultdict(lambda: np.array([])))

    def compute_f1_score(self, pred: str, gold: str) -> float:
        """计算F1分数"""
        if not pred or not gold:
            return 0.0

        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())

        if len(pred_tokens) == 0:
            return 0.0

        intersection = pred_tokens & gold_tokens
        if len(intersection) == 0:
            return 0.0

        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gold_tokens) if len(gold_tokens) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def compute_answer_reward(self, pred_answer: str, gt_answer: str,
                              format_correct: bool = True) -> float:
        """计算答案奖励"""
        if not format_correct:
            return 0.0

        return self.compute_f1_score(pred_answer, gt_answer)

    def compute_search_key_reward(self, search_queries: List[str],
                                  reference_keywords: List[List[str]]) -> float:
        """计算搜索关键词奖励"""
        if not search_queries or not reference_keywords:
            return 0.0

        total_score = 0.0
        num_subquestions = len(reference_keywords)

        for ref_subq_keywords in reference_keywords:
            max_score_for_subq = 0.0

            # 对于每个子问题的参考关键词
            for ref_keyword in ref_subq_keywords:
                max_score_for_keyword = 0.0

                # 找到与该关键词最匹配的查询
                for query in search_queries:
                    score = self.compute_f1_score(query, ref_keyword)
                    max_score_for_keyword = max(max_score_for_keyword, score)

                max_score_for_subq = max(max_score_for_subq, max_score_for_keyword)

            total_score += max_score_for_subq

        return total_score / num_subquestions if num_subquestions > 0 else 0.0

    def compute_global_reward(self, pred_answer: str, gt_answer: str,
                              search_queries: List[str], reference_keywords: List[List[str]],
                              format_correct: bool = True) -> Dict[str, float]:
        """计算全局奖励"""
        # 答案奖励
        r_answer = self.compute_answer_reward(pred_answer, gt_answer, format_correct)

        # 搜索关键词奖励
        r_key = self.compute_search_key_reward(search_queries, reference_keywords)

        # 总奖励
        r_total = r_answer + self.gamma_key * r_key

        return {
            'answer_reward': r_answer,
            'search_key_reward': r_key,
            'total_global_reward': r_total
        }

    def compute_document_similarity(self, doc1: str, doc2: str) -> float:
        """计算两个文档的相似度"""
        if not doc1 or not doc2:
            return 0.0

        try:
            # 使用TF-IDF计算余弦相似度
            docs = [doc1, doc2]
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            # 回退到简单的词汇重叠
            words1 = set(doc1.lower().split())
            words2 = set(doc2.lower().split())
            if len(words1) == 0 or len(words2) == 0:
                return 0.0
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if len(union) > 0 else 0.0

    def compute_information_gain(self, retrieved_docs: List[str], golden_docs: List[str],
                                 episode_id: str, step: int) -> float:
        """计算信息增益"""
        if not retrieved_docs or not golden_docs:
            return 0.0

        # 获取或初始化该episode的记忆
        if len(self.episode_memories[episode_id][step]) == 0:
            self.episode_memories[episode_id][step] = np.zeros(len(golden_docs))

        memory = self.episode_memories[episode_id][step]

        # 计算当前检索文档与黄金文档的相似度
        current_similarities = []
        for i, golden_doc in enumerate(golden_docs):
            max_sim = 0.0
            for retrieved_doc in retrieved_docs:
                sim = self.compute_document_similarity(retrieved_doc, golden_doc)
                max_sim = max(max_sim, sim)
            current_similarities.append(max_sim)

        # 计算信息增益
        total_gain = 0.0
        for i, current_sim in enumerate(current_similarities):
            # 增益 = max(0, 当前相似度 - 历史最大相似度)
            gain = max(0, current_sim - memory[i])
            total_gain += gain

            # 更新记忆
            memory[i] = max(memory[i], current_sim)

        # 更新episode记忆
        self.episode_memories[episode_id][step] = memory

        return total_gain / len(golden_docs)

    def compute_redundancy_penalty(self, retrieved_docs: List[str],
                                   history_docs: List[str]) -> float:
        """计算冗余惩罚"""
        if not retrieved_docs or not history_docs:
            return 0.0

        redundant_count = 0
        for retrieved_doc in retrieved_docs:
            for history_doc in history_docs:
                similarity = self.compute_document_similarity(retrieved_doc, history_doc)
                if similarity > self.redundancy_threshold:
                    redundant_count += 1
                    break  # 只要找到一个重复就跳出

        return redundant_count / len(retrieved_docs)

    def compute_step_reward(self, retrieved_docs: List[str], golden_docs: List[str],
                            history_docs: List[str], episode_id: str, step: int) -> Dict[str, float]:
        """计算步骤奖励"""
        # 信息增益
        information_gain = self.compute_information_gain(
            retrieved_docs, golden_docs, episode_id, step
        )

        # 冗余惩罚
        redundancy_penalty = self.compute_redundancy_penalty(retrieved_docs, history_docs)

        # 步骤奖励 = 信息增益 - 冗余惩罚
        step_reward = information_gain - redundancy_penalty

        return {
            'information_gain': information_gain,
            'redundancy_penalty': redundancy_penalty,
            'step_reward': step_reward
        }

    def compute_trajectory_rewards(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算整个轨迹的奖励"""
        episode_id = trajectory_data['episode_id']
        steps = trajectory_data['steps']
        final_answer = trajectory_data['final_answer']
        gt_answer = trajectory_data['gt_answer']
        reference_keywords = trajectory_data['reference_keywords']
        golden_docs = trajectory_data['golden_docs']
        format_correct = trajectory_data.get('format_correct', True)

        # 收集所有搜索查询
        all_search_queries = []
        for step_data in steps:
            if step_data.get('search_query'):
                all_search_queries.append(step_data['search_query'])

        # 计算全局奖励
        global_rewards = self.compute_global_reward(
            final_answer, gt_answer, all_search_queries,
            reference_keywords, format_correct
        )

        # 计算每一步的奖励
        step_rewards = []
        history_docs = []

        for i, step_data in enumerate(steps):
            retrieved_docs = step_data.get('retrieved_docs', [])

            if retrieved_docs:  # 只对有检索的步骤计算奖励
                step_reward_data = self.compute_step_reward(
                    retrieved_docs, golden_docs, history_docs, episode_id, i
                )
                step_rewards.append(step_reward_data)
                history_docs.extend(retrieved_docs)
            else:
                # 没有检索的步骤给0奖励
                step_rewards.append({
                    'information_gain': 0.0,
                    'redundancy_penalty': 0.0,
                    'step_reward': 0.0
                })

        return {
            'global_rewards': global_rewards,
            'step_rewards': step_rewards,
            'episode_id': episode_id
        }

    def reset_episode_memory(self, episode_id: str):
        """重置episode记忆"""
        if episode_id in self.episode_memories:
            del self.episode_memories[episode_id]

    def get_reward_statistics(self) -> Dict[str, float]:
        """获取奖励统计信息"""
        # 这里可以添加奖励分布的统计信息
        return {
            'num_episodes_processed': len(self.episode_memories),
            'avg_memory_size': np.mean([
                len(episode_memory)
                for episode_memory in self.episode_memories.values()
            ]) if self.episode_memories else 0.0
        }


class RewardNormalizer:
    """奖励标准化器"""

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    def update(self, rewards: List[float]):
        """更新运行统计"""
        if not rewards:
            return

        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)

        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

        self.count += len(rewards)

    def normalize(self, rewards: List[float]) -> List[float]:
        """标准化奖励"""
        if not rewards or self.running_var == 0:
            return rewards

        normalized = [(r - self.running_mean) / (np.sqrt(self.running_var) + 1e-8) for r in rewards]
        return normalized

    def __call__(self, rewards: List[float]) -> List[float]:
        """可调用接口"""
        self.update(rewards)
        return self.normalize(rewards)