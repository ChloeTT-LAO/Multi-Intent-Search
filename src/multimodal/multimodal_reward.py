"""
多模态奖励计算器
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from PIL import Image

from ..training.reward_calculator import StepSearchRewardCalculator


class MultimodalRewardCalculator(StepSearchRewardCalculator):
    """多模态奖励计算器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 多模态奖励权重
        self.image_analysis_weight = config['reward'].get('image_analysis_weight', 0.1)
        self.cross_modal_consistency_weight = config['reward'].get('cross_modal_consistency_weight', 0.1)

        # 初始化图像相似度计算
        self.setup_image_similarity()

    def setup_image_similarity(self):
        """设置图像相似度计算"""
        try:
            import clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.device = device
        except ImportError:
            print("CLIP not available, image similarity calculation disabled")
            self.clip_model = None

    def compute_image_text_consistency(self, image_analysis: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """计算图像分析与检索文档的一致性"""
        if not image_analysis or not retrieved_docs:
            return 0.0

        consistency_scores = []

        for doc in retrieved_docs:
            doc_text = doc.get('text', '')
            if doc_text:
                # 计算图像分析文本与文档文本的相似度
                similarity = self.compute_text_similarity(image_analysis, doc_text)
                consistency_scores.append(similarity)

        return max(consistency_scores) if consistency_scores else 0.0

    def compute_image_analysis_quality(self, image_analysis: str, question: str) -> float:
        """计算图像分析质量"""
        if not image_analysis:
            return 0.0

        quality_score = 0.0

        # 检查分析的详细程度
        analysis_words = image_analysis.split()
        if len(analysis_words) >= 10:  # 至少10个词
            quality_score += 0.3

        # 检查是否包含描述性词汇
        descriptive_words = {
            'color', 'shape', 'size', 'location', 'object', 'person', 'building',
            'shows', 'depicts', 'contains', 'visible', 'appears', 'seems'
        }

        analysis_lower = image_analysis.lower()
        found_descriptive = sum(1 for word in descriptive_words if word in analysis_lower)
        quality_score += min(found_descriptive / len(descriptive_words), 0.3)

        # 检查与问题的相关性
        question_words = set(question.lower().split())
        analysis_words_set = set(analysis_lower.split())
        relevance = len(question_words & analysis_words_set) / len(question_words) if question_words else 0
        quality_score += relevance * 0.4

        return min(quality_score, 1.0)

    def compute_multimodal_search_relevance(self, text_query: str, image_search_query: str,
                                            retrieved_docs: List[Dict[str, Any]], question: str) -> float:
        """计算多模态搜索相关性"""
        if not retrieved_docs:
            return 0.0

        relevance_scores = []

        for doc in retrieved_docs:
            doc_text = doc.get('text', '')
            doc_score = 0.0

            # 文本查询相关性
            if text_query and doc_text:
                text_relevance = self.compute_search_relevance(text_query, [doc_text], question)
                doc_score += 0.6 * text_relevance

            # 图像搜索查询相关性
            if image_search_query and doc_text:
                image_relevance = self.compute_search_relevance(image_search_query, [doc_text], question)
                doc_score += 0.4 * image_relevance

            relevance_scores.append(doc_score)

        return max(relevance_scores) if relevance_scores else 0.0

    def compute_multimodal_global_reward(self, pred_answer: str, gt_answer: str,
                                         search_queries: List[str], image_search_queries: List[str],
                                         reference_keywords: List[List[str]], image_analysis_steps: List[str],
                                         format_correct: bool = True, has_image: bool = False) -> Dict[str, float]:
        """计算多模态全局奖励"""

        # 基础答案奖励
        r_answer = self.compute_answer_reward(pred_answer, gt_answer, format_correct)

        # 文本搜索关键词奖励
        all_search_queries = search_queries + image_search_queries
        r_key = self.compute_search_key_reward(all_search_queries, reference_keywords)

        # 图像分析奖励
        r_image_analysis = 0.0
        if has_image and image_analysis_steps:
            # 对所有图像分析步骤的平均质量评分
            analysis_scores = [
                self.compute_image_analysis_quality(analysis, pred_answer)
                for analysis in image_analysis_steps
            ]
            r_image_analysis = np.mean(analysis_scores) if analysis_scores else 0.0

        # 组合总奖励
        r_total = (r_answer +
                   self.gamma_key * r_key +
                   self.image_analysis_weight * r_image_analysis)

        return {
            'answer_reward': r_answer,
            'search_key_reward': r_key,
            'image_analysis_reward': r_image_analysis,
            'total_global_reward': r_total
        }

    def compute_multimodal_step_reward(self, retrieved_docs: List[Dict[str, Any]],
                                       golden_docs: List[Dict[str, Any]],
                                       history_docs: List[Dict[str, Any]],
                                       episode_id: str, step: int,
                                       image_analysis: str = None,
                                       search_query: str = None,
                                       image_search_query: str = None) -> Dict[str, float]:
        """计算多模态步骤奖励"""

        # 提取文本用于基础计算
        retrieved_texts = [doc.get('text', '') for doc in retrieved_docs if doc.get('text')]
        golden_texts = [doc.get('text', '') for doc in golden_docs if doc.get('text')]
        history_texts = [doc.get('text', '') for doc in history_docs if doc.get('text')]

        # 基础步骤奖励
        base_step_rewards = self.compute_step_reward(
            retrieved_texts, golden_texts, history_texts, episode_id, step
        )

        # 跨模态一致性奖励
        cross_modal_reward = 0.0
        if image_analysis and retrieved_docs:
            cross_modal_reward = self.compute_image_text_consistency(image_analysis, retrieved_docs)

        # 多模态搜索效果奖励
        multimodal_search_reward = 0.0
        if search_query or image_search_query:
            # 这里简化为检查是否有效使用了多模态搜索
            if search_query and image_search_query:
                multimodal_search_reward = 0.1  # 使用了两种搜索方式的奖励

        # 组合步骤奖励
        enhanced_step_reward = (base_step_rewards['step_reward'] +
                                self.cross_modal_consistency_weight * cross_modal_reward +
                                multimodal_search_reward)

        return {
            'information_gain': base_step_rewards['information_gain'],
            'redundancy_penalty': base_step_rewards['redundancy_penalty'],
            'cross_modal_consistency': cross_modal_reward,
            'multimodal_search_bonus': multimodal_search_reward,
            'step_reward': enhanced_step_reward
        }

    def compute_multimodal_trajectory_rewards(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算多模态轨迹奖励"""
        episode_id = trajectory_data['episode_id']
        steps = trajectory_data['steps']
        final_answer = trajectory_data['final_answer']
        gt_answer = trajectory_data['gt_answer']
        reference_keywords = trajectory_data['reference_keywords']
        golden_docs = trajectory_data['golden_docs']
        format_correct = trajectory_data.get('format_correct', True)
        has_image = trajectory_data.get('has_image', False)
        image_analysis_steps = trajectory_data.get('image_analysis_steps', [])

        # 收集所有搜索查询
        search_queries = []
        image_search_queries = []
        for step_data in steps:
            if step_data.get('search_query'):
                search_queries.append(step_data['search_query'])
            if step_data.get('image_search_query'):
                image_search_queries.append(step_data['image_search_query'])

        # 计算全局奖励
        global_rewards = self.compute_multimodal_global_reward(
            final_answer, gt_answer, search_queries, image_search_queries,
            reference_keywords, image_analysis_steps, format_correct, has_image
        )

        # 计算每一步的奖励
        step_rewards = []
        history_docs = []

        for i, step_data in enumerate(steps):
            retrieved_docs = step_data.get('retrieved_docs', [])
            image_analysis = step_data.get('image_analysis', '')
            search_query = step_data.get('search_query', '')
            image_search_query = step_data.get('image_search_query', '')

            step_reward_data = self.compute_multimodal_step_reward(
                retrieved_docs, golden_docs, history_docs, episode_id, i,
                image_analysis, search_query, image_search_query
            )
            step_rewards.append(step_reward_data)

            # 更新历史文档
            history_docs.extend(retrieved_docs)

        return {
            'global_rewards': global_rewards,
            'step_rewards': step_rewards,
            'episode_id': episode_id
        }


class CrossModalRewardCalculator:
    """跨模态奖励计算器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_cross_modal_models()

    def setup_cross_modal_models(self):
        """设置跨模态模型"""
        try:
            import clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.device = device
        except ImportError:
            print("CLIP not available for cross-modal reward calculation")
            self.clip_model = None

    def compute_image_question_alignment(self, image: Image.Image, question: str) -> float:
        """计算图像与问题的对齐度"""
        if self.clip_model is None or image is None:
            return 0.0

        try:
            import clip

            # 编码图像
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            # 编码问题
            text_tokens = clip.tokenize([question]).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_tokens)

                # 标准化特征
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 计算相似度
                similarity = torch.cosine_similarity(image_features, text_features).item()

            return max(0.0, similarity)  # 确保非负

        except Exception as e:
            print(f"Error in image-question alignment: {e}")
            return 0.0

    def compute_visual_grounding_reward(self, image_analysis: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """计算视觉基础奖励"""
        if not image_analysis or not retrieved_docs:
            return 0.0

        # 检查图像分析是否有助于理解检索到的文档
        grounding_score = 0.0

        for doc in retrieved_docs:
            doc_text = doc.get('text', '')
            if doc_text:
                # 简单的关键词匹配方法
                analysis_words = set(image_analysis.lower().split())
                doc_words = set(doc_text.lower().split())

                # 计算重叠度
                overlap = len(analysis_words & doc_words)
                if overlap > 0:
                    grounding_score += overlap / len(analysis_words | doc_words)

        return min(grounding_score, 1.0)

    def compute_multimodal_coherence(self, text_search_query: str, image_search_query: str,
                                     image_analysis: str) -> float:
        """计算多模态连贯性"""
        if not all([text_search_query, image_search_query, image_analysis]):
            return 0.0

        # 检查文本搜索、图像搜索和图像分析之间的连贯性
        coherence_score = 0.0

        # 方法1：检查关键词重叠
        text_words = set(text_search_query.lower().split())
        image_words = set(image_search_query.lower().split())
        analysis_words = set(image_analysis.lower().split())

        # 计算各部分的重叠
        text_image_overlap = len(text_words & image_words) / len(text_words | image_words) if (
                    text_words | image_words) else 0
        text_analysis_overlap = len(text_words & analysis_words) / len(text_words | analysis_words) if (
                    text_words | analysis_words) else 0
        image_analysis_overlap = len(image_words & analysis_words) / len(image_words | analysis_words) if (
                    image_words | analysis_words) else 0

        coherence_score = (text_image_overlap + text_analysis_overlap + image_analysis_overlap) / 3

        return coherence_score

    