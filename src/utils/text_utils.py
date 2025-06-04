"""
文本处理工具
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter


def normalize_text(text: str) -> str:
    """标准化文本"""
    if not text:
        return ""

    # Unicode标准化
    text = unicodedata.normalize('NFKD', text)

    # 转小写
    text = text.lower()

    # 移除多余空格
    text = ' '.join(text.split())

    return text


def remove_punctuation(text: str, keep_chars: Set[str] = None) -> str:
    """移除标点符号"""
    if not text:
        return ""

    if keep_chars is None:
        keep_chars = set()

    # 创建翻译表
    translator = str.maketrans('', '', ''.join(c for c in string.punctuation if c not in keep_chars))

    return text.translate(translator)


def extract_keywords(text: str, min_length: int = 2, max_keywords: int = 10) -> List[str]:
    """提取关键词"""
    if not text:
        return []

    # 标准化文本
    normalized = normalize_text(text)

    # 移除标点符号
    no_punct = remove_punctuation(normalized)

    # 分词
    words = no_punct.split()

    # 过滤短词和停用词
    stopwords = get_stopwords()
    keywords = [word for word in words
                if len(word) >= min_length and word not in stopwords]

    # 统计频率并排序
    word_counts = Counter(keywords)

    # 返回最常见的关键词
    return [word for word, _ in word_counts.most_common(max_keywords)]


def get_stopwords() -> Set[str]:
    """获取英文停用词集合"""
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'would', 'you', 'your', 'have', 'had',
        'this', 'they', 'them', 'their', 'there', 'these', 'those', 'then',
        'than', 'or', 'but', 'not', 'what', 'when', 'where', 'who', 'which',
        'why', 'how', 'all', 'some', 'any', 'each', 'few', 'more', 'most',
        'other', 'such', 'only', 'own', 'same', 'so', 'can', 'could',
        'should', 'do', 'does', 'did', 'done', 'been', 'being', 'were'
    }
    return stopwords


def extract_entities(text: str) -> Dict[str, List[str]]:
    """简单的实体提取（基于规则）"""
    if not text:
        return {}

    entities = {
        'numbers': [],
        'dates': [],
        'urls': [],
        'emails': [],
        'capitalized_words': []
    }

    # 数字
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    entities['numbers'] = numbers

    # 日期（简单模式）
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text)
    entities['dates'] = dates

    # URL
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    entities['urls'] = urls

    # 邮箱
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    entities['emails'] = emails

    # 大写单词（可能是专有名词）
    capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
    entities['capitalized_words'] = capitalized

    return entities


def clean_text_for_search(text: str) -> str:
    """为搜索清理文本"""
    if not text:
        return ""

    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 移除特殊字符，保留字母数字和基本标点
    text = re.sub(r'[^\w\s\.\?\!,;:]', ' ', text)

    # 标准化空格
    text = ' '.join(text.split())

    return text.strip()


def split_sentences(text: str) -> List[str]:
    """分句"""
    if not text:
        return []

    # 简单的分句规则
    sentences = re.split(r'[.!?]+', text)

    # 清理并过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """截断文本"""
    if not text or len(text) <= max_length:
        return text

    # 尝试在单词边界截断
    if max_length > len(suffix):
        truncate_at = max_length - len(suffix)
        space_index = text.rfind(' ', 0, truncate_at)

        if space_index > max_length * 0.7:  # 如果找到合适的空格位置
            return text[:space_index] + suffix
        else:
            return text[:truncate_at] + suffix

    return text[:max_length]


def extract_questions(text: str) -> List[str]:
    """提取问题"""
    if not text:
        return []

    # 查找问句模式
    question_patterns = [
        r'[.!?]\s*([A-Z][^.!?]*\?)',  # 句子后的问句
        r'^([A-Z][^.!?]*\?)',  # 开头的问句
        r'([Ww]hat|[Ww]hen|[Ww]here|[Ww]ho|[Ww]hy|[Hh]ow|[Ww]hich|[Dd]o|[Dd]oes|[Dd]id|[Cc]an|[Cc]ould|[Ww]ould|[Ss]hould)[^.!?]*\?'
    ]

    questions = []
    for pattern in question_patterns:
        matches = re.findall(pattern, text)
        questions.extend(matches)

    # 去重并清理
    unique_questions = []
    seen = set()

    for q in questions:
        q = q.strip()
        if q and q not in seen:
            unique_questions.append(q)
            seen.add(q)

    return unique_questions


def extract_search_terms(question: str) -> List[str]:
    """从问题中提取搜索词"""
    if not question:
        return []

    # 标准化问题
    question = normalize_text(question)

    # 移除疑问词
    question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose'}
    words = question.split()

    # 过滤停用词和疑问词
    stopwords = get_stopwords()
    search_terms = []

    for word in words:
        word = remove_punctuation(word)
        if (word and
                len(word) > 2 and
                word not in stopwords and
                word not in question_words):
            search_terms.append(word)

    return search_terms


def compute_text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（简单版本）"""
    if not text1 or not text2:
        return 0.0

    # 标准化文本
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    # 分词
    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 or not words2:
        return 0.0

    # 计算Jaccard相似度
    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def extract_noun_phrases(text: str) -> List[str]:
    """简单的名词短语提取"""
    if not text:
        return []

    # 简单的名词短语模式（大写字母开头的连续词）
    patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 连续的大写单词
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # 两个大写单词
    ]

    noun_phrases = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        noun_phrases.extend(matches)

    # 去重
    return list(set(noun_phrases))


def format_search_query(terms: List[str], max_length: int = 100) -> str:
    """格式化搜索查询"""
    if not terms:
        return ""

    # 连接搜索词
    query = ' '.join(terms)

    # 截断到最大长度
    if len(query) > max_length:
        query = truncate_text(query, max_length, suffix="")

    return query.strip()


class TextProcessor:
    """文本处理器类"""

    def __init__(self, language: str = 'en'):
        self.language = language
        self.stopwords = get_stopwords()

    def preprocess(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""

        # 清理HTML和特殊字符
        text = clean_text_for_search(text)

        # 标准化
        text = normalize_text(text)

        return text

    def extract_search_queries(self, question: str, num_queries: int = 3) -> List[str]:
        """从问题提取多个搜索查询"""
        if not question:
            return []

        queries = []

        # 方法1：直接使用问题
        queries.append(question.strip('?'))

        # 方法2：提取关键搜索词
        search_terms = extract_search_terms(question)
        if search_terms:
            queries.append(' '.join(search_terms[:5]))  # 前5个关键词

        # 方法3：提取名词短语
        noun_phrases = extract_noun_phrases(question)
        if noun_phrases:
            queries.append(' '.join(noun_phrases[:3]))  # 前3个名词短语

        # 方法4：组合重要词汇
        if len(search_terms) > 2:
            # 选择最重要的词汇组合
            important_terms = search_terms[:3]
            queries.append(' '.join(important_terms))

        # 去重并限制数量
        unique_queries = []
        seen = set()

        for query in queries:
            query = query.strip()
            if query and query not in seen and len(query) > 2:
                unique_queries.append(query)
                seen.add(query)

                if len(unique_queries) >= num_queries:
                    break

        return unique_queries

    def clean_retrieved_text(self, text: str, max_length: int = 500) -> str:
        """清理检索到的文本"""
        if not text:
            return ""

        # 预处理
        text = self.preprocess(text)

        # 分句
        sentences = split_sentences(text)

        # 选择最相关的句子（这里简化为前几句）
        if sentences:
            combined = '. '.join(sentences[:3])  # 取前3句
            return truncate_text(combined, max_length)

        return truncate_text(text, max_length)