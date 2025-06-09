"""
StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="stepsearch",
    version="1.0.0",
    author="StepSearch Team",
    author_email="stepsearch@example.com",
    description="Step-wise Proximal Policy Optimization for LLM Search Enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/StepSearch",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "rich": [
            "rich>=13.4.0",
        ],
        "serving": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stepsearch-train=scripts.train:main",
            "stepsearch-eval=scripts.evaluate:main",
            "stepsearch-infer=scripts.inference:main",
            "stepsearch-prepare=scripts.prepare_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "stepsearch": [
            "config/*.yaml",
            "musique/templates/*.txt",
        ],
    },
    keywords=[
        "machine learning",
        "natural language processing",
        "reinforcement learning",
        "question answering",
        "information retrieval",
        "large language models",
        "search enhancement",
        "step-wise learning",
        "proximal policy optimization"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/StepSearch/issues",
        "Source": "https://github.com/your-username/StepSearch",
        "Documentation": "https://stepsearch.readthedocs.io/",
    },
)