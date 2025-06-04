"""
StePPO训练器实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..models.step_search_model import StepSearchModelWithValueHead
from .reward_calculator import StepSearchRewardCalculator, RewardNormalizer


@dataclass
class TrajectoryStep:
    """轨迹步骤数据"""
    state: str
    action: str
    log_prob: float
    value: float
    retrieved_docs: List[str] = None
    search_query: str = None
    reward: float = 0.0
    advantage: float = 0.0


@dataclass
class TrajectoryData:
    """完整轨迹数据"""
    episode_id: str
    steps: List[TrajectoryStep]
    final_answer: str
    question: str
    gt_answer: str
    reference_keywords: List[List[str]]
    golden_docs: List[str]
    global_reward: float = 0.0
    format_correct: bool = True


class GAECalculator:
    """广义优势估计计算器"""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

    def compute_gae(self, rewards: List[float], values: List[float],
                    next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """计算GAE优势和返回值"""
        advantages = []
        returns = []

        # 反向计算优势
        gae = 0
        for i in reversed(range(len(rewards))):
            next_val = next_value if i == len(rewards) - 1 else values[i + 1]
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)

        # 计算返回值
        for i in range(len(rewards)):
            returns.append(advantages[i] + values[i])

        return advantages, returns


class StePPOTrainer:
    """StePPO训练器"""

    def __init__(self, model: StepSearchModelWithValueHead,
                 reward_calculator: StepSearchRewardCalculator,
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

        # 奖励标准化器
        self.reward_normalizer = RewardNormalizer()

        # 统计信息
        self.training_stats = defaultdict(list)

    def extract_components_from_action(self, action: str) -> Dict[str, str]:
        """从动作中提取各个组件"""
        components = {
            'thinking': None,
            'search_query': None,
            'answer': None
        }

        # 提取思考
        think_match = re.search(r'<think>(.*?)</think>', action, re.DOTALL)
        if think_match:
            components['thinking'] = think_match.group(1).strip()

        # 提取搜索查询
        search_match = re.search(r'<search>(.*?)</search>', action, re.DOTALL)
        if search_match:
            components['search_query'] = search_match.group(1).strip()

        # 提取答案
        answer_match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
        if answer_match:
            components['answer'] = answer_match.group(1).strip()

        return components

    def validate_action_format(self, action: str) -> bool:
        """验证动作格式"""
        # 检查必需的标签
        has_think = '<think>' in action and '</think>' in action

        # 如果有搜索，检查格式
        has_search_open = '<search>' in action
        has_search_close = '</search>' in action
        search_valid = (has_search_open and has_search_close) or (not has_search_open and not has_search_close)

        # 如果有答案，检查格式
        has_answer_open = '<answer>' in action
        has_answer_close = '</answer>' in action
        answer_valid = (has_answer_open and has_answer_close) or (not has_answer_open and not has_answer_close)

        return has_think and search_valid and answer_valid

    def generate_trajectory(self, question: str, reference_data: Dict[str, Any]) -> TrajectoryData:
        """生成单个轨迹"""
        episode_id = reference_data.get('id', 'unknown')
        trajectory_steps = []

        # 初始状态
        current_state = self.create_initial_prompt(question)
        step_count = 0
        final_answer = ""
        format_correct = True

        while step_count < self.max_search_steps:
            # 生成动作
            action, log_probs = self.model.generate_response(
                current_state,
                max_new_tokens=512,
                temperature=1.0,
                do_sample=True
            )

            # 验证格式
            if not self.validate_action_format(action):
                format_correct = False

            # 提取组件
            components = self.extract_components_from_action(action)

            # 计算价值函数
            inputs = self.model.tokenizer(
                current_state + action,
                return_tensors="pt",
                truncation=True,
                max_length=self.config['model']['max_length']
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                _, value = self.model.forward_with_value(**inputs)
                value = value.item()

            # 执行搜索（如果有）
            retrieved_docs = []
            if components['search_query']:
                retrieved_docs = self.search_engine.search(components['search_query'])
                # 更新状态
                info_text = f"\n<information>{' '.join(retrieved_docs[:3])}</information>\n"
                current_state += action + info_text
            else:
                current_state += action + "\n"

            # 创建轨迹步骤
            step = TrajectoryStep(
                state=current_state,
                action=action,
                log_prob=np.mean(log_probs) if log_probs else 0.0,
                value=value,
                retrieved_docs=retrieved_docs,
                search_query=components['search_query']
            )
            trajectory_steps.append(step)

            # 检查是否结束
            if components['answer']:
                final_answer = components['answer']
                break

            step_count += 1

        # 创建轨迹数据
        trajectory = TrajectoryData(
            episode_id=episode_id,
            steps=trajectory_steps,
            final_answer=final_answer,
            question=question,
            gt_answer=reference_data['answer'],
            reference_keywords=reference_data['reference_keywords'],
            golden_docs=reference_data['golden_docs'],
            format_correct=format_correct
        )

        return trajectory

    def compute_trajectory_rewards(self, trajectory: TrajectoryData) -> TrajectoryData:
        """计算轨迹奖励"""
        # 准备数据
        trajectory_data = {
            'episode_id': trajectory.episode_id,
            'steps': [],
            'final_answer': trajectory.final_answer,
            'gt_answer': trajectory.gt_answer,
            'reference_keywords': trajectory.reference_keywords,
            'golden_docs': trajectory.golden_docs,
            'format_correct': trajectory.format_correct
        }

        # 收集步骤数据
        for step in trajectory.steps:
            step_data = {
                'search_query': step.search_query,
                'retrieved_docs': step.retrieved_docs or []
            }
            trajectory_data['steps'].append(step_data)

        # 计算奖励
        reward_results = self.reward_calculator.compute_trajectory_rewards(trajectory_data)

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

    def compute_advantages(self, trajectory: TrajectoryData) -> TrajectoryData:
        """计算优势函数"""
        rewards = [step.reward for step in trajectory.steps]
        values = [step.value for step in trajectory.steps]

        # 使用GAE计算优势
        advantages, returns = self.gae_calculator.compute_gae(rewards, values)

        # 标准化优势
        if len(advantages) > 1:
            advantages = self.reward_normalizer(advantages)

        # 更新轨迹
        for i, step in enumerate(trajectory.steps):
            step.advantage = advantages[i]

        return trajectory

    def compute_ppo_loss(self, trajectories: List[TrajectoryData]) -> Dict[str, torch.Tensor]:
        """计算PPO损失"""
        policy_losses = []
        value_losses = []
        kl_divs = []

        for trajectory in trajectories:
            for step in trajectory.steps:
                # 重新计算当前策略的log概率和价值
                inputs = self.model.tokenizer(
                    step.state,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config['model']['max_length']
                )

                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 获取动作tokens
                action_inputs = self.model.tokenizer(
                    step.action,
                    return_tensors="pt",
                    add_special_tokens=False
                )
                action_token_ids = action_inputs['input_ids'][0]

                # 前向传播
                logits, new_value = self.model.forward_with_value(**inputs)

                # 计算新的log概率（简化版本）
                probs = F.softmax(logits[0, -len(action_token_ids):], dim=-1)
                new_log_prob = torch.log(probs[range(len(action_token_ids)), action_token_ids] + 1e-10).mean()

                # 计算比率
                old_log_prob = torch.tensor(step.log_prob, device=device)
                ratio = torch.exp(new_log_prob - old_log_prob)

                # 计算优势
                advantage = torch.tensor(step.advantage, device=device)

                # PPO裁剪损失
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage
                policy_loss = -torch.min(surr1, surr2)
                policy_losses.append(policy_loss)

                # 价值函数损失
                target_value = torch.tensor(step.reward + step.advantage, device=device)
                value_loss = F.mse_loss(new_value.squeeze(), target_value)
                value_losses.append(value_loss)

                # KL散度
                kl_div = old_log_prob - new_log_prob
                kl_divs.append(kl_div)

        return {
            'policy_loss': torch.stack(policy_losses).mean() if policy_losses else torch.tensor(0.0),
            'value_loss': torch.stack(value_losses).mean() if value_losses else torch.tensor(0.0),
            'kl_div': torch.stack(kl_divs).mean() if kl_divs else torch.tensor(0.0)
        }

    def train_step(self, batch_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """执行一步训练"""
        trajectories = []

        # 生成轨迹
        for data in batch_data:
            trajectory = self.generate_trajectory(data['question'], data)
            trajectory = self.compute_trajectory_rewards(trajectory)
            trajectory = self.compute_advantages(trajectory)
            trajectories.append(trajectory)

        # 计算损失
        losses = self.compute_ppo_loss(trajectories)

        # 总损失
        total_loss = (losses['policy_loss'] +
                      0.5 * losses['value_loss'] +
                      self.kl_coef * losses['kl_div'])

        # 反向传播
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()

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
            'total_loss': total_loss.item(),
            'policy_loss': losses['policy_loss'].item(),
            'value_loss': losses['value_loss'].item(),
            'kl_div': losses['kl_div'].item(),
            'avg_reward': np.mean([traj.global_reward for traj in trajectories]),
            'avg_trajectory_length': np.mean([len(traj.steps) for traj in trajectories])
        }

        # 更新统计
        for key, value in stats.items():
            self.training_stats[key].append(value)

        return stats

    def create_initial_prompt(self, question: str) -> str:
        """创建初始提示"""
        prompt = f"""You are a research assistant. Answer the question by searching for information step by step.

Question: {question}

Use the following format:
<think>your reasoning process</think>
<search>search keywords</search>
(system will provide information)
<think>continue reasoning</think>
<answer>final answer</answer>

Start your response:
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
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(load_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = defaultdict(list, checkpoint['training_stats'])

        print(f"Checkpoint loaded from {load_path}")
        return checkpoint['epoch']