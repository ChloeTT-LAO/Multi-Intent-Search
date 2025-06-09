#!/usr/bin/env python3
"""
构建知识库脚本
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.search.search_engine import TFIDFSearchEngine


def download_wikipedia_sample():
    """下载Wikipedia样本数据"""
    # 这里使用一些公开的Wikipedia数据源或API
    sample_articles = [
        {
            "title": "Artificial Intelligence",
            "content": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals."
        },
        {
            "title": "Machine Learning",
            "content": "Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values."
        },
        {
            "title": "Natural Language Processing",
            "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data."
        },
        {
            "title": "Deep Learning",
            "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics and drug design."
        },
        {
            "title": "Python Programming",
            "content": "Python is a high-level, interpreted, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming."
        },
        {
            "title": "Data Science",
            "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data, and apply knowledge and actionable insights from data across a broad range of application domains."
        }
    ]
    return sample_articles


def build_knowledge_base_from_musique(musique_data_path: str) -> List[str]:
    """从MuSiQue数据构建知识库"""
    documents = []

    if not Path(musique_data_path).exists():
        print(f"MuSiQue data not found at {musique_data_path}")
        return documents

    with open(musique_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # 从问答对构建文档
        question = item.get('question', '')
        answer = item.get('answer', '')

        if question and answer:
            doc = f"Question: {question} Answer: {answer}"
            documents.append(doc)

        # 如果有supporting facts，也加入知识库
        if 'supporting_facts' in item:
            for fact in item['supporting_facts']:
                if isinstance(fact, dict) and 'text' in fact:
                    documents.append(fact['text'])
                elif isinstance(fact, str):
                    documents.append(fact)

    return documents


def build_general_knowledge_base() -> List[str]:
    """构建通用知识库"""
    knowledge_base = [
        # 地理知识
        "Beijing is the capital of China and has a population of over 21 million people. It is located in northern China.",
        "Paris is the capital and most populous city of France. It is famous for landmarks like the Eiffel Tower and the Louvre Museum.",
        "Tokyo is the capital of Japan and the most populous metropolitan area in the world with over 37 million people.",
        "London is the capital of England and the United Kingdom. It is situated on the River Thames in southeastern England.",
        "New York City is the most populous city in the United States, located in the state of New York.",
        "Sydney is the largest city in Australia and is famous for the Sydney Opera House and Sydney Harbour Bridge.",
        "Berlin is the capital and largest city of Germany. It has a rich history and was divided during the Cold War.",
        "Rome is the capital city of Italy and was the center of the ancient Roman Empire.",
        "Moscow is the capital of Russia and the largest city in Europe by population.",
        "Cairo is the capital of Egypt and is located on the banks of the Nile River.",

        # 历史知识
        "World War II lasted from 1939 to 1945 and involved most of the world's nations.",
        "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th centuries.",
        "The American Civil War was fought from 1861 to 1865 between the Union and Confederate states.",
        "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing.",
        "Ancient Egypt was known for its pyramids, pharaohs, and hieroglyphic writing system.",
        "The Roman Empire was one of the largest empires in ancient history, lasting over 1000 years.",
        "Christopher Columbus reached the Americas in 1492, beginning European exploration of the New World.",
        "The French Revolution began in 1789 and led to major political and social changes in France.",

        # 科学知识
        "DNA is a molecule that carries genetic instructions in all living organisms.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "Water has the chemical formula H2O, consisting of two hydrogen atoms and one oxygen atom.",
        "The periodic table organizes chemical elements by their atomic number and properties.",
        "Gravity is a fundamental force that attracts objects with mass toward each other.",
        "The human body has 206 bones in the adult skeleton and about 600 muscles.",
        "Cells are the basic units of life, and all living things are composed of one or more cells.",

        # 技术知识
        "Artificial Intelligence refers to computer systems that can perform tasks requiring human-like intelligence.",
        "Machine Learning is a subset of AI that enables computers to learn and improve from data without being explicitly programmed.",
        "The Internet is a global network of interconnected computers that communicate using standardized protocols.",
        "Blockchain is a distributed ledger technology that maintains a continuously growing list of records.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Python is a high-level programming language known for its simplicity and versatility.",
        "HTML is the standard markup language for creating web pages and web applications.",
        "SQL is a programming language designed for managing and querying relational databases.",
    ]

    return knowledge_base


def main():
    """主函数"""
    logger = setup_logger("build_kb")

    # 创建知识库目录
    kb_dir = Path("./data/knowledge_base")
    kb_dir.mkdir(parents=True, exist_ok=True)

    print("Building knowledge base...")

    # 方法1: 从已处理的MuSiQue数据构建
    musique_path = "./data/processed/train_processed.json"
    documents = []

    if Path(musique_path).exists():
        print("Building from MuSiQue data...")
        documents.extend(build_knowledge_base_from_musique(musique_path))

    # 方法2: 添加通用知识
    print("Adding general knowledge...")
    documents.extend(build_general_knowledge_base())

    # 方法3: 添加Wikipedia样本
    print("Adding Wikipedia sample...")
    wiki_articles = download_wikipedia_sample()
    for article in wiki_articles:
        doc = f"Title: {article['title']}\nContent: {article['content']}"
        documents.append(doc)

    # 去重
    unique_documents = list(set(documents))
    print(f"Total unique documents: {len(unique_documents)}")

    # 保存文档
    documents_file = kb_dir / "documents.json"
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(unique_documents, f, ensure_ascii=False, indent=2)

    print(f"Documents saved to: {documents_file}")

    # 构建并保存TF-IDF索引
    print("Building TF-IDF index...")
    search_engine = TFIDFSearchEngine(unique_documents)

    index_dir = kb_dir / "tfidf_index"
    search_engine.save_index(str(index_dir))

    print(f"TF-IDF index saved to: {index_dir}")

    # 测试搜索
    print("\nTesting search engine...")
    test_queries = [
        "capital of France",
        "artificial intelligence",
        "World War II",
        "Python programming"
    ]

    for query in test_queries:
        results = search_engine.search(query, top_k=2)
        print(f"\nQuery: {query}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result[:100]}...")

    print("\nKnowledge base construction completed!")


if __name__ == "__main__":
    main()