"""
StepSearch主模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class StepSearchModel(nn.Module):
    """StepSearch主模型"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = model_name

        # 加载预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=config['model'].get('cache_dir', None)
        )

        # 添加特殊token
        self.special_tokens = [
            '<think>', '</think>',
            '<search>', '</search>',
            '<information>', '</information>',
            '<answer>', '</answer>'
        ]

        # 添加token到tokenizer
        num_added = self.tokenizer.add_tokens(self.special_tokens)
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"Added {num_added} special tokens to tokenizer")

        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None, **kwargs) -> CausalLMOutputWithPast:
        """前向传播"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate_response(self, prompt: str, max_new_tokens: int = 512,
                          temperature: float = 1.0, do_sample: bool = True) -> Tuple[str, List[float]]:
        """生成响应并返回token级log概率"""
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model']['max_length'] - max_new_tokens,
            padding=True
        )

        # 移动到设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )

        # 解码生成的文本
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # 计算token级log概率
        log_probs = []
        if outputs.scores:
            for i, scores in enumerate(outputs.scores):
                if i < len(generated_ids):
                    probs = F.softmax(scores[0], dim=-1)
                    token_id = generated_ids[i]
                    log_prob = torch.log(probs[token_id] + 1e-10).item()
                    log_probs.append(log_prob)

        return response, log_probs

    def extract_search_query(self, text: str) -> Optional[str]:
        """从文本中提取搜索查询"""
        pattern = r'<search>(.*?)</search>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def extract_answer(self, text: str) -> Optional[str]:
        """从文本中提取答案"""
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def extract_thinking(self, text: str) -> Optional[str]:
        """从文本中提取思考过程"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def validate_format(self, text: str) -> bool:
        """验证输出格式是否正确"""
        # 检查是否有必需的标签
        has_think = '<think>' in text and '</think>' in text
        has_answer = '<answer>' in text and '</answer>' in text

        # 检查搜索标签是否成对出现
        search_open = text.count('<search>')
        search_close = text.count('</search>')
        search_balanced = search_open == search_close

        return has_think and has_answer and search_balanced

    def create_mask_for_information(self, input_ids: torch.Tensor) -> torch.Tensor:
        """为information段创建掩码，在训练时不计算这些token的损失"""
        # 将input_ids转换为文本以便处理
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)

        # 创建掩码
        mask = torch.ones_like(input_ids[0], dtype=torch.bool)

        # 找到所有information段并设置掩码
        info_pattern = r'<information>.*?</information>'
        for match in re.finditer(info_pattern, text, re.DOTALL):
            start_pos = match.start()
            end_pos = match.end()

            # 将字符位置转换为token位置（简化版本）
            start_token = len(self.tokenizer.encode(text[:start_pos], add_special_tokens=False))
            end_token = len(self.tokenizer.encode(text[:end_pos], add_special_tokens=False))

            # 设置掩码
            mask[start_token:end_token] = False

        return mask.unsqueeze(0)

    def compute_loss_with_mask(self, logits: torch.Tensor, labels: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
        """计算带掩码的损失"""
        # 展平tensor
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        flat_mask = mask.view(-1)

        # 计算损失
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')

        # 应用掩码
        masked_loss = loss * flat_mask.float()

        # 计算平均损失
        return masked_loss.sum() / flat_mask.float().sum()

    def save_model(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """加载模型"""
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"Model loaded from {load_path}")


class StepSearchModelWithValueHead(StepSearchModel):
    """带价值头的StepSearch模型（用于PPO训练）"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)

        # 添加价值头
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # 初始化价值头参数
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward_with_value(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                           **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播并返回logits和value"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # 获取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None

        if hidden_states is None:
            # 如果没有hidden_states，需要重新前向传播获取
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
            hidden_states = outputs.hidden_states[-1]

        # 计算value（使用最后一个token的hidden state）
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        values = self.value_head(last_hidden).squeeze(-1)  # [batch_size]

        return outputs.logits, values


def create_step_search_model(config: Dict[str, Any], with_value_head: bool = False) -> StepSearchModel:
    """创建StepSearch模型"""
    model_name = config['model']['name']

    if with_value_head:
        return StepSearchModelWithValueHead(model_name, config)
    else:
        return StepSearchModel(model_name, config)