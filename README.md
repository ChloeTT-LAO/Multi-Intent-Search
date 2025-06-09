# StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2505.15107-b31b1b.svg)](https://arxiv.org/abs/2505.15107)

</div>

## 📖 Overview

StepSearch is a reinforcement learning framework that enhances Large Language Models' search capabilities through **step-wise proximal policy optimization**. Unlike existing methods that rely on sparse global rewards, StepSearch introduces fine-grained, token-level supervision for each search step, combining information gain rewards with redundancy penalties.

### 🔥 Key Features

- **Step-wise RL Training**: Token-level rewards for each search step
- **Dual Reward System**: Information gain + redundancy penalty
- **Fine-grained Supervision**: Sub-question level search trajectory guidance  
- **Multi-hop QA**: Specialized for complex reasoning tasks
- **SOTA Performance**: Significant improvements over existing baselines

### 📊 Results

StepSearch achieves significant improvements over Search-R1 baselines:

| Dataset | Our EM | Our F1 | Baseline EM | Baseline F1 | EM Δ | F1 Δ |
|---------|--------|--------|-------------|-------------|------|------|
| HotpotQA | 34.5% | 45.2% | 27.2% | 36.1% | +7.3% | +9.1% |
| 2WikiMultiHop | 32.0% | 38.5% | 24.8% | 29.6% | +7.2% | +8.9% |
| MuSiQue | 17.4% | 26.1% | 8.1% | 14.6% | +9.3% | +11.5% |
| Bamboogle | 34.4% | 45.2% | 17.6% | 27.0% | +16.8% | +18.2% |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/StepSearch.git
cd StepSearch

# Create virtual environment
conda create -n stepsearch python=3.9
conda activate stepsearch

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Data Preparation

```bash
# Download and process MuSiQue musique
python scripts/prepare_data.py \
    --raw-musique-dir ./musique/raw \
    --output-dir ./musique/processed \
    --max-train-samples 1000 \
    --build-knowledge-base
```

### Training

```bash
# Train StepSearch model
python scripts/train.py \
    --musique-path ./musique/processed/train_processed.json \
    --output-dir ./checkpoints \
    --eval-musique-path ./musique/processed/dev_processed.json
```

### Evaluation

```bash
# Evaluate on multiple datasets
python scripts/evaluate.py \
    --model-path ./checkpoints/best_model \
    --datasets hotpotqa 2wiki musique bamboogle \
    --musique-dir ./musique/eval \
    --output-dir ./results \
    --compare-baselines
```

### Interactive Inference

```bash
# Start interactive mode
python scripts/inference.py \
    --model-path ./checkpoints/best_model \
    --mode interactive

# Single question
python scripts/inference.py \
    --model-path ./checkpoints/best_model \
    --mode single \
    --question "What is the capital of the country where the Eiffel Tower is located?"
```

## 🏗️ Architecture

### Core Components

- **StepSearch Model**: Enhanced LLM with search capabilities
- **StePPO Trainer**: Step-wise PPO implementation with dual rewards
- **Reward Calculator**: Information gain and redundancy penalty computation
- **Search Engine**: Flexible search backend (Wikipedia, custom knowledge base)
- **Data Pipeline**: Automated data processing with GPT-4o integration

### Framework Flow

```
Question → Think → Search → Information → Think → Search → ... → Answer
     ↓         ↓        ↓           ↓         ↓        ↓           ↓
   Input   Step-wise  Retrieval  Integration  ...   Final    Global
          Rewards     Rewards    Rewards            Rewards   Reward
```

## 📁 Project Structure

```
StepSearch/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml                 # Main configuration
├── src/
│   ├── data/
│   │   ├── data_pipeline.py        # Data processing pipeline
│   │   ├── dataset.py              # Dataset classes
│   │   └── data_utils.py           # Data utilities
│   ├── models/
│   │   ├── step_search_model.py    # Core model implementation
│   │   ├── reward_model.py         # Reward model
│   │   └── value_model.py          # Value function
│   ├── training/
│   │   ├── steppo_trainer.py       # StePPO trainer
│   │   ├── reward_calculator.py    # Reward computation
│   │   └── training_utils.py       # Training utilities
│   ├── search/
│   │   ├── search_engine.py        # Search engine interface
│   │   ├── wiki_search.py          # Wikipedia search
│   │   └── mock_search.py          # Mock search for testing
│   ├── evaluation/
│   │   ├── evaluator.py            # Evaluation framework
│   │   └── metrics.py              # Evaluation metrics
│   └── utils/
│       ├── common.py               # Common utilities
│       ├── logging_utils.py        # Logging utilities
│       └── text_utils.py           # Text processing
├── scripts/
│   ├── prepare_data.py             # Data preparation
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── inference.py                # Inference script
├── data/                           # Data directory
├── checkpoints/                    # Model checkpoints
└── logs/                          # Training logs
```

## 🔧 Configuration

The main configuration is in `config/config.yaml`:

```yaml
# Model settings
model:
  name: "Qwen/Qwen2.5-3B-Base"
  max_length: 2048

# Training settings
training:
  batch_size: 32
  learning_rate: 7e-7
  num_epochs: 10
  clip_range: 0.2

# Reward settings
reward:
  gamma_key: 0.1
  redundancy_threshold: 0.8
  max_search_steps: 5

# Search settings
search:
  engine_type: "wiki"
  top_k: 3
```

## 📚 Advanced Usage

### Custom Search Engine

```python
from src.search.search_engine import SearchEngine

class CustomSearchEngine(SearchEngine):
    def search(self, query: str, top_k: int = 3) -> List[str]:
        # Implement your search logic
        return retrieved_documents
    
    def add_documents(self, documents: List[str]) -> None:
        # Add documents to your index
        pass
```

### Custom Reward Function

```python
from src.training.reward_calculator import StepSearchRewardCalculator

class CustomRewardCalculator(StepSearchRewardCalculator):
    def compute_step_reward(self, retrieved_docs, golden_docs, history, episode_id, step):
        # Implement custom step reward logic
        return custom_reward
```

### Training with Custom Data

```python
from src.data.dataset import StepSearchDataset

# Prepare your musique in the required format
custom_data = [
    {
        'question': 'Your question',
        'answer': 'Expected answer',
        'subquestions': [
            {
                'sub_question': 'Sub-question',
                'search_queries': ['query1', 'query2']
            }
        ]
    }
]

# Create dataset and train
dataset = StepSearchDataset(custom_data, tokenizer)
# ... continue with training
```

## 📊 Monitoring and Logging

StepSearch provides comprehensive logging and monitoring:

- **Training Logs**: Detailed training metrics and loss curves
- **Evaluation Metrics**: EM, F1, search efficiency metrics
- **Search Analytics**: Query success rates, retrieval quality
- **Model Checkpoints**: Automatic saving of best models

View logs using:

```bash
# View training logs
tail -f logs/train.log

# View evaluation results
cat results/evaluation_summary.json

# Monitor with TensorBoard (if configured)
tensorboard --logdir logs/tensorboard
```

## 🔬 Research and Development

### Reproducing Paper Results

1. **Data Preparation**: Use the exact MuSiQue processing pipeline
2. **Model Training**: Follow the hyperparameters in the paper
3. **Evaluation**: Test on the four standard benchmarks
4. **Comparison**: Compare against Search-R1 baseline

### Extending StepSearch

- **New Reward Functions**: Implement domain-specific rewards
- **Different Models**: Support for other LLM architectures
- **Multi-modal**: Extend to image+text search tasks
- **Efficiency**: Optimize for faster training and inference

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/ scripts/
flake8 src/ scripts/

# Type checking
mypy src/
```

## 📄 Citation

If you use StepSearch in your research, please cite our paper:

```bibtex
@article{wang2025stepsearch,
  title={StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization},
  author={Wang, Ziliang and Zheng, Xuhui and An, Kang and Ouyang, Cijun and Cai, Jialu and Wang, Yuhang and Wu, Yichao},
  journal={arXiv preprint arXiv:2505.15107},
  year={2025}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/StepSearch/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/StepSearch/discussions)
- **Email**: stepsearch@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MuSiQue dataset creators for the foundation data
- OpenAI for GPT-4o API support
- Hugging Face for the transformers library
- The research community for valuable feedback

---

<div align="center">

**🔍 Enhancing LLM Search, One Step at a Time 🔍**

</div>