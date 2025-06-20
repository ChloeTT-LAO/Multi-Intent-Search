"""
多模态StepSearch模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import base64
import io


class MultimodalStepSearchModel(nn.Module):
    """多模态StepSearch模型"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = model_name

        # 初始化多模态模型
        self.setup_multimodal_model()

        # 添加特殊token
        self.special_tokens = [
            '<think>', '</think>',
            '<image_analysis>', '</image_analysis>',
            '<search>', '</search>',
            '<image_search>', '</image_search>',
            '<information>', '</information>',
            '<answer>', '</answer>'
        ]

        # 添加token到tokenizer
        self.add_special_tokens()

    def setup_multimodal_model(self):
        """设置多模态模型"""
        # 支持多种多模态模型
        if 'llava' in self.model_name.lower():
            self.setup_llava_model()
        elif 'qwen-vl' in self.model_name.lower():
            self.setup_qwen_vl_model()
        elif 'gpt-4v' in self.model_name.lower():
            self.setup_gpt4v_model()
        else:
            # 默认使用LLaVA架构
            self.setup_llava_model()

    def setup_llava_model(self):
        """设置LLaVA模型"""
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

            self.processor = LlavaNextProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.config['model'].get('cache_dir', None)
            )
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.config['model'].get('cache_dir', None)
            )
            self.tokenizer = self.processor.tokenizer

        except ImportError:
            print("LLaVA not available, falling back to text-only model")
            self.setup_text_only_fallback()

    def setup_qwen_vl_model(self):
        """设置Qwen-VL模型"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.config['model'].get('cache_dir', None)
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.config['model'].get('cache_dir', None)
            )
            self.tokenizer = self.processor.tokenizer

        except ImportError:
            print("Qwen-VL not available, falling back to text-only model")
            self.setup_text_only_fallback()

    def setup_gpt4v_model(self):
        """设置GPT-4V模型（API调用）"""
        import openai

        self.api_client = openai.OpenAI()
        self.model_type = "gpt-4v-api"

        # 使用GPT-4的tokenizer作为fallback
        from transformers import GPT2TokenizerFast
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def setup_text_only_fallback(self):
        """设置纯文本fallback模型"""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        text_model_name = self.config['model'].get('fallback_model', 'microsoft/DialoGPT-medium')
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(text_model_name)
        self.processor = None

    def add_special_tokens(self):
        """添加特殊token"""
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            num_added = self.tokenizer.add_tokens(self.special_tokens)
            if num_added > 0 and hasattr(self, 'model'):
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"Added {num_added} special tokens to tokenizer")

            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def process_image(self, image_input: Union[str, Image.Image, torch.Tensor]) -> Image.Image:
        """处理图片输入"""
        if isinstance(image_input, str):
            if image_input.startswith('data:image'):
                # Base64编码的图片
                header, data = image_input.split(',', 1)
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
            else:
                # 文件路径
                image = Image.open(image_input)
        elif isinstance(image_input, torch.Tensor):
            # 张量转PIL图像
            import torchvision.transforms as transforms
            to_pil = transforms.ToPILImage()
            image = to_pil(image_input)
        else:
            image = image_input

        return image.convert('RGB')

    def create_multimodal_prompt(self, text_prompt: str, image: Optional[Image.Image] = None) -> str:
        """创建多模态提示"""
        if image is None:
            return text_prompt

        # 为不同模型创建适合的提示格式
        if 'llava' in self.model_name.lower():
            return f"<image>\nUser: {text_prompt}\nAssistant:"
        elif 'qwen-vl' in self.model_name.lower():
            return f"<img>image</img>\n{text_prompt}"
        else:
            return f"Given the image, {text_prompt}"

    def generate_multimodal_response(self, text_prompt: str, image: Optional[Union[str, Image.Image]] = None,
                                     max_new_tokens: int = 512, temperature: float = 1.0,
                                     do_sample: bool = True) -> Tuple[str, List[float]]:
        """生成多模态响应"""

        # 处理图片
        processed_image = None
        if image is not None:
            processed_image = self.process_image(image)

        # 根据模型类型生成响应
        if hasattr(self, 'model_type') and self.model_type == "gpt-4v-api":
            return self.generate_with_gpt4v_api(text_prompt, processed_image, max_new_tokens, temperature)
        else:
            return self.generate_with_local_model(text_prompt, processed_image, max_new_tokens, temperature, do_sample)

    def generate_with_gpt4v_api(self, text_prompt: str, image: Optional[Image.Image],
                                max_new_tokens: int, temperature: float) -> Tuple[str, List[float]]:
        """使用GPT-4V API生成响应"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]

            # 添加图片
            if image is not None:
                # 将PIL图像转换为base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_str}"
                    }
                })

            response = self.api_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )

            generated_text = response.choices[0].message.content
            # API不提供token级概率，返回空列表
            log_probs = []

            return generated_text, log_probs

        except Exception as e:
            print(f"GPT-4V API error: {e}")
            return "", []

    def generate_with_local_model(self, text_prompt: str, image: Optional[Image.Image],
                                  max_new_tokens: int, temperature: float, do_sample: bool) -> Tuple[str, List[float]]:
        """使用本地模型生成响应"""
        if self.processor is None or image is None:
            # 回退到纯文本生成
            return self.generate_text_only(text_prompt, max_new_tokens, temperature, do_sample)

        try:
            # 创建多模态提示
            prompt = self.create_multimodal_prompt(text_prompt, image)

            # 处理输入
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
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

        except Exception as e:
            print(f"Local multimodal generation error: {e}")
            return self.generate_text_only(text_prompt, max_new_tokens, temperature, do_sample)

    def generate_text_only(self, text_prompt: str, max_new_tokens: int,
                           temperature: float, do_sample: bool) -> Tuple[str, List[float]]:
        """纯文本生成（fallback）"""
        try:
            inputs = self.tokenizer(
                text_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config['model']['max_length'] - max_new_tokens,
                padding=True
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

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

        except Exception as e:
            print(f"Text generation error: {e}")
            return "", []

    def analyze_image(self, image: Union[str, Image.Image],
                      analysis_prompt: str = "Describe this image in detail.") -> str:
        """分析图片内容"""
        if isinstance(image, str):
            image = self.process_image(image)

        response, _ = self.generate_multimodal_response(
            analysis_prompt,
            image,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False
        )

        return response

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

        # 检查搜索标签格式
        search_tags = ['<search>', '</search>', '<image_search>', '</image_search>']
        search_valid = True
        for i in range(0, len(search_tags), 2):
            open_tag, close_tag = search_tags[i], search_tags[i + 1]
            has_open = open_tag in action
            has_close = close_tag in action
            if has_open != has_close:
                search_valid = False
                break

        # 检查其他标签
        other_tags = ['<image_analysis>', '</image_analysis>', '<answer>', '</answer>']
        other_valid = True
        for i in range(0, len(other_tags), 2):
            open_tag, close_tag = other_tags[i], other_tags[i + 1]
            has_open = open_tag in action
            has_close = close_tag in action
            if has_open != has_close:
                other_valid = False
                break

        return has_think and search_valid and other_valid

    def save_model(self, save_path: str):
        """保存模型"""
        if hasattr(self, 'model') and hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        if hasattr(self, 'processor') and hasattr(self.processor, 'save_pretrained'):
            self.processor.save_pretrained(save_path)
        elif hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(save_path)
        print(f"Multimodal model saved to {save_path}")

    def load_model(self, load_path: str):
        """加载模型"""
        try:
            if hasattr(self, 'processor'):
                self.processor = type(self.processor).from_pretrained(load_path)
                self.model = type(self.model).from_pretrained(load_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(load_path)
            print(f"Multimodal model loaded from {load_path}")
        except Exception as e:
            print(f"Failed to load multimodal model: {e}")


class MultimodalStepSearchModelWithValueHead(MultimodalStepSearchModel):
    """带价值头的多模态StepSearch模型"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)

        # 添加价值头
        if hasattr(self.model.config, 'hidden_size'):
            hidden_size = self.model.config.hidden_size
        else:
            hidden_size = 768  # 默认值

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

    def forward_with_value(self, text_prompt: str, image: Optional[Union[str, Image.Image]] = None,
                           **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播并返回logits和value"""
        # 这是一个简化版本，实际实现需要根据具体的多模态模型架构调整

        # 生成响应并获取hidden states
        response, log_probs = self.generate_multimodal_response(
            text_prompt, image, max_new_tokens=1, temperature=0.1, do_sample=False
        )

        # 在实际实现中，你需要从模型的hidden states计算value
        # 这里使用一个简化的方法
        try:
            if image is not None and self.processor is not None:
                inputs = self.processor(
                    text=text_prompt,
                    images=self.process_image(image) if isinstance(image, str) else image,
                    return_tensors="pt",
                    padding=True
                )
            else:
                inputs = self.tokenizer(
                    text_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # 使用最后一层的hidden state计算value
            hidden_states = outputs.hidden_states[-1]
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
            values = self.value_head(last_hidden).squeeze(-1)  # [batch_size]

            # 返回logits和values
            logits = outputs.logits if hasattr(outputs, 'logits') else torch.zeros(1, 1, 1000)

            return logits, values

        except Exception as e:
            print(f"Error in forward_with_value: {e}")
            # 返回默认值
            return torch.zeros(1, 1, 1000), torch.zeros(1)


def create_multimodal_step_search_model(config: Dict[str, Any],
                                        with_value_head: bool = False) -> MultimodalStepSearchModel:
    """创建多模态StepSearch模型"""
    model_name = config['model']['name']

    if with_value_head:
        return MultimodalStepSearchModelWithValueHead(model_name, config)
    else:
        return MultimodalStepSearchModel(model_name, config)