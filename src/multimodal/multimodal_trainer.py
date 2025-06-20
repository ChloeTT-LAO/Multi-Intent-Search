"""
多模态StePPO训练器实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from PIL import Image

from ..training.steppo_trainer import TrajectoryStep, TrajectoryData, GAECalculator
from .multimodal_model import MultimodalStepSearchModelWithValueHead
from .multimodal_reward import MultimodalRewardCalculator


@dataclass
class MultimodalTrajectoryStep:
    """多模态轨迹步骤数据"""
    state: str
    action: str
    log_prob: float
    value: float
    image: Optional[Image.Image] = None
    retrieved_docs: List[Dict[str, Any]] = None
    search_query: str = None
    image_search_query: str = None
    image_analysis: str = None
    reward: float = 0.0
    advantage: float = 0.0


@dataclass
class MultimodalTrajectoryData:
    """完整多模态轨迹数据"""
    episode_id: str
    steps: List[MultimodalTrajectoryStep]
    final_answer: str
    question: str
    image: Optional[Image.Image]
    gt_answer: str
    reference_keywords: List[List[str]]
    golden_docs: List[Dict[str, Any]]
    global_reward: float = 0.0
    format_correct: bool = True


class MultimodalStePPOTrainer:
    """多模态StePPO训练器"""

    def __init__(self, model: MultimodalStepSearchModelWithValueHead,
                 reward_calculator: MultimodalRewardCalculator,
                 search_engine, config: Dict[str, Any]):
        self.model = model
        self.reward_calculator = reward_calculator
        self.search_engine = search_engine
        self.config = config

        # 训练配置
        self.training_config = config['training']
        self.lr = self.training_config['learning_rate']
        self.value_lr = self.training_config['value_lr']
        self.clip_range = self.training_config['clip_range']
        self.kl_coef = self.training_config['kl_coef']
        self.max_search_steps = config['reward']['max_search_steps']

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        self.value_optimizer = torch.optim.AdamW(
            self.model.value_head.parameters(),
            lr=self.value_lr,
            weight_decay=0.01
        )

        # GAE计算器
        self.gae_calculator = GAECalculator()

        # 统计信息
        self.training_stats = defaultdict(list)

    def extract_multimodal_components(self, action: str) -> Dict[str, str]:
        """从动作中提取多模态组件"""
        components = {
            'thinking': None,
            'image_analysis': None,
            'search_query': None,
            'image_search_query': None,
            'answer': None
        }

        # 提取思考过程
        think_match = re.search(r'<think>(.*?)</think>', action, re.DOTALL)
        if think_match:
            components['thinking'] = think_match.group(1).strip()

        # 提取图像分析
        image_analysis_match = re.search(r'<image_analysis>(.*?)</image_analysis>', action, re.DOTALL)
        if image_analysis_match:
            components['image_analysis'] = image_analysis_match.group(1).strip()

        # 提取文本搜索查询
        search_match = re.search(r'<search>(.*?)</search>', action, re.DOTALL)
        if search_match:
            components['search_query'] = search_match.group(1).strip()

        # 提取图像搜索查询
        image_search_match = re.search(r'<image_search>(.*?)</image_search>', action, re.DOTALL)
        if image_search_match:
            components['image_search_query'] = image_search_match.group(1).strip()

        # 提取答案
        answer_match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
        if answer_match:
            components['answer'] = answer_match.group(1).strip()

        return components

    def validate_multimodal_format(self, action: str) -> bool:
        """验证多模态动作格式"""
        # 检查必需的标签
        has_think = '<think>' in action and '</think>' in action

        # 检查搜索标签是否成对出现
        search_tags = [
            ('<search>', '</search>'),
            ('<image_search>', '</image_search>'),
            ('<image_analysis>', '</image_analysis>'),
            ('<answer>', '</answer>')
        ]

        for open_tag, close_tag in search_tags:
            has_open = open_tag in action
            has_close = close_tag in action
            if has_open != has_close:
                return False

        return has_think

    def generate_multimodal_trajectory(self, question: str, image: Optional[Image.Image],
                                       reference_data: Dict[str, Any]) -> MultimodalTrajectoryData:
        """生成多模态轨迹"""
        episode_id = reference_data.get('id', 'unknown')
        trajectory_steps = []

        # 初始状态
        current_state = self.create_multimodal_prompt(question, image)
        step_count = 0
        final_answer = ""
        format_correct = True

        while step_count < self.max_search_steps:
            # 生成动作
            action, log_probs = self.model.generate_multimodal_response(
                current_state,
                image,
                max_new_tokens=512,
                temperature=1.0,
                do_sample=True
            )

            # 验证格式
            if not self.validate_multimodal_format(action):
                format_correct = False

            # 提取组件
            components = self.extract_multimodal_components(action)

            # 计算价值函数
            try:
                _, value = self.model.forward_with_value(current_state + action, image)
                value = value.item() if torch.is_tensor(value) else value
            except Exception as e:
                print(f"Error computing value: {e}")
                value = 0.0

            # 执行搜索（如果有）
            retrieved_docs = []

            # 文本搜索
            if components['search_query']:
                if hasattr(self.search_engine, 'search_multimodal'):
                    search_results = self.search_engine.search_multimodal(
                        text_query=components['search_query'],
                        top_k=3
                    )
                    retrieved_docs.extend(search_results)
                else:
                    text_results = self.search_engine.search(components['search_query'])
                    retrieved_docs.extend([{'text': text, 'image': None} for text in text_results])

            # 图像搜索
            if components['image_search_query'] and image is not None:
                if hasattr(self.search_engine, 'search_multimodal'):
                    image_search_results = self.search_engine.search_multimodal(
                        text_query=components['image_search_query'],
                        image_query=image,
                        top_k=3
                    )
                    retrieved_docs.extend(image_search_results)

            # 更新状态
            if retrieved_docs:
                info_texts = []
                for doc in retrieved_docs:
                    info_text = doc.get('text', '')
                    if doc.get('image') is not None:
                        info_text += " [Contains image]"
                    info_texts.append(info_text)

                info_section = f"\n<information>{' '.join(info_texts[:3])}</information>\n"
                current_state += action + info_section
            else:
                current_state += action + "\n"

            # 创建轨迹步骤
            step = MultimodalTrajectoryStep(
                state=current_state,
                action=action,
                log_prob=np.mean(log_probs) if log_probs else 0.0,
                value=value,
                image=image,
                retrieved_docs=retrieved_docs,
                search_query=components['search_query'],
                image_search_query=components['image_search_query'],
                image_analysis=components['image_analysis']
            )
            trajectory_steps.append(step)

            # 检查是否结束
            if components['answer']:
                final_answer = components['answer']
                break

            step_count += 1

        # 创建轨迹数据
        trajectory = MultimodalTrajectoryData(
            episode_id=episode_id,
            steps=trajectory_steps,
            final_answer=final_answer,
            question=question,
            image=image,
            gt_answer=reference_data['answer'],
            reference_keywords=reference_data['reference_keywords'],
            golden_docs=reference_data['golden_docs'],
            format_correct=format_correct
        )

        return trajectory

    def compute_multimodal_trajectory_rewards(self, trajectory: MultimodalTrajectoryData) -> MultimodalTrajectoryData:
        """计算多模态轨迹奖励"""
        # 准备数据
        trajectory_data = {
            'episode_id': trajectory.episode_id,
            'steps': [],
            'final_answer': trajectory.final_answer,
            'gt_answer': trajectory.gt_answer,
            'reference_keywords': trajectory.reference_keywords,
            'golden_docs': trajectory.golden_docs,
            'format_correct': trajectory.format_correct,
            'has_image': trajectory.image is not None,
            'image_analysis_steps': []
        }

        # 收集步骤数据
        for step in trajectory.steps:
            step_data = {
                'search_query': step.search_query,
                'image_search_query': step.image_search_query,
                'retrieved_docs': step.retrieved_docs or [],
                'image_analysis': step.image_analysis
            }
            trajectory_data['steps'].append(step_data)

            # 收集图像分析步骤
            if step.image_analysis:
                trajectory_data['image_analysis_steps'].append(step.image_analysis)

        # 计算奖励
        reward_results = self.reward_calculator.compute_multimodal_trajectory_rewards(trajectory_data)

        # 分配全局奖励
        trajectory.global_reward = reward_results['global_rewards']['total_global_reward']

        # 分配步骤奖励
        step_rewards = reward_results['step_rewards']
        for i, step in enumerate(trajectory.steps):
            if i < len(step_rewards):
                step.reward = step_rewards[i]['step_reward']
            else:
                step.reward = 0.0

        # 为最后一步添加全局奖励
        if trajectory.steps:
            trajectory.steps[-1].reward += trajectory.global_reward

        return trajectory

    def compute_multimodal_advantages(self, trajectory: MultimodalTrajectoryData) -> MultimodalTrajectoryData:
        """计算多模态优势函数"""
        rewards = [step.reward for step in trajectory.steps]
        values = [step.value for step in trajectory.steps]

        # 使用GAE计算优势
        advantages, returns = self.gae_calculator.compute_gae(rewards, values)

        # 标准化优势
        if len(advantages) > 1:
            mean_adv = np.mean(advantages)
            std_adv = np.std(advantages)
            if std_adv > 1e-8:
                advantages = [(adv - mean_adv) / (std_adv + 1e-8) for adv in advantages]

        # 更新轨迹
        for i, step in enumerate(trajectory.steps):
            step.advantage = advantages[i]

        return trajectory

    def compute_multimodal_ppo_loss(self, trajectories: List[MultimodalTrajectoryData]) -> Dict[str, torch.Tensor]:
        """计算多模态PPO损失"""
        policy_losses = []
        value_losses = []
        kl_divs = []

        for trajectory in trajectories:
            for step in trajectory.steps:
                try:
                    # 重新计算当前策略的log概率和价值
                    _, new_value = self.model.forward_with_value(step.state, step.image)

                    # 简化的log概率计算
                    new_response, new_log_probs = self.model.generate_multimodal_response(
                        step.state,
                        step.image,
                        max_new_tokens=len(step.action.split()),
                        temperature=0.1,
                        do_sample=False
                    )

                    if new_log_probs:
                        new_log_prob = torch.tensor(np.mean(new_log_probs), requires_grad=True)
                    else:
                        continue

                    # 计算比率
                    old_log_prob = torch.tensor(step.log_prob)
                    ratio = torch.exp(new_log_prob - old_log_prob)

                    # 计算优势
                    advantage = torch.tensor(step.advantage)

                    # PPO裁剪损失
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    policy_losses.append(policy_loss)

                    # 价值函数损失
                    target_value = torch.tensor(step.reward + step.advantage)
                    if torch.is_tensor(new_value):
                        if new_value.dim() > 0:
                            new_value = new_value.squeeze()
                        value_loss = F.mse_loss(new_value, target_value)
                    else:
                        value_loss = F.mse_loss(torch.tensor(new_value), target_value)
                    value_losses.append(value_loss)

                    # KL散度
                    kl_div = old_log_prob - new_log_prob
                    kl_divs.append(kl_div)

                except Exception as e:
                    print(f"Error in loss computation: {e}")
                    continue

        return {
            'policy_loss': torch.stack(policy_losses).mean() if policy_losses else torch.tensor(0.0),
            'value_loss': torch.stack(value_losses).mean() if value_losses else torch.tensor(0.0),
            'kl_div': torch.stack(kl_divs).mean() if kl_divs else torch.tensor(0.0)
        }

    def train_step(self, batch_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """执行一步多模态训练"""
        trajectories = []

        # 生成轨迹
        for data in batch_data:
            trajectory = self.generate_multimodal_trajectory(
                data['question'],
                data.get('image', None),
                data
            )
            trajectory = self.compute_multimodal_trajectory_rewards(trajectory)
            trajectory = self.compute_multimodal_advantages(trajectory)
            trajectories.append(trajectory)

        # 计算损失
        losses = self.compute_multimodal_ppo_loss(trajectories)

        # 总损失
        total_loss = (losses['policy_loss'] +
                      0.5 * losses['value_loss'] +
                      self.kl_coef * losses['kl_div'])

        # 反向传播
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        if total_loss.requires_grad:
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.value_optimizer.step()

        # 清理episode记忆
        for trajectory in trajectories:
            self.reward_calculator.reset_episode_memory(trajectory.episode_id)

        # 统计信息
        stats = {
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'policy_loss': losses['policy_loss'].item() if torch.is_tensor(losses['policy_loss']) else 0.0,
            'value_loss': losses['value_loss'].item() if torch.is_tensor(losses['value_loss']) else 0.0,
            'kl_div': losses['kl_div'].item() if torch.is_tensor(losses['kl_div']) else 0.0,
            'avg_reward': np.mean([traj.global_reward for traj in trajectories]),
            'avg_trajectory_length': np.mean([len(traj.steps) for traj in trajectories]),
            'avg_image_analysis_steps': np.mean([
                sum(1 for step in traj.steps if step.image_analysis) for traj in trajectories
            ]),
            'multimodal_search_rate': np.mean([
                sum(1 for step in traj.steps if step.image_search_query) / len(traj.steps)
                for traj in trajectories if len(traj.steps) > 0
            ])
        }

        # 更新统计
        for key, value in stats.items():
            self.training_stats[key].append(value)

        return stats

    def create_multimodal_prompt(self, question: str, image: Optional[Image.Image]) -> str:
        """创建多模态提示"""
        if image is None:
            # 纯文本提示
            prompt = f"""You are a research assistant. Answer the question by searching for information step by step.

Question: {question}

Use the following format:
<think>your reasoning process</think>
<search>search keywords</search>
(system will provide information)
<answer>final answer</answer>

Begin your response:
"""
        else:
            # 多模态提示
            prompt = f"""You are a multimodal research assistant. Answer the question by analyzing the image and searching for information step by step.

Question: {question}

Use the following format:
<think>your reasoning process</think>
<image_analysis>describe what you see in the image</image_analysis>
<search>text search keywords</search>
<image_search>image-related search keywords</image_search>
(system will provide information)
<answer>final answer</answer>

Begin your response:
"""
        return prompt

    def save_checkpoint(self, save_path: str, epoch: int):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': dict(self.training_stats),
            'config': self.config
        }
        torch.save(checkpoint, save_path)
        print(f"Multimodal checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(load_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = defaultdict(list, checkpoint['training_stats'])

        print(f"Multimodal checkpoint loaded from {load_path}")
        return checkpoint['epoch']