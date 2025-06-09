"""
Wikipedia搜索引擎实现
"""

import json
import pickle
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import logging

from .search_engine import SearchEngine

logger = logging.getLogger(__name__)

class WikipediaAPI:
    """Wikipedia API客户端"""

    def __init__(self, language: str = 'en', delay: float = 0.1):
        self.language = language
        self.delay = delay
        self.base_url = f"https://{language}.wikipedia.org/api/rest_v1"
        self.search_url = f"https://{language}.wikipedia.org/w/api.php"

        # 会话设置
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'StepSearch/1.0 (https://github.com/stepsearch/stepsearch)'
        })

    def search_titles(self, query: str, limit: int = 10) -> List[str]:
        """搜索文章标题"""
        params = {
            'action': 'opensearch',
            'search': query,
            'limit': limit,
            'namespace': 0,
            'format': 'json'
        }

        try:
            response = self.session.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if len(data) >= 2:
                return data[1]  # 返回标题列表
            return []

        except Exception as e:
            logger.warning(f"Wikipedia title search failed: {e}")
            return []

    def get_page_content(self, title: str) -> Optional[str]:
        """获取页面内容"""
        try:
            # URL编码标题
            encoded_title = quote(title.replace(' ', '_'))
            url = f"{self.base_url}/page/summary/{encoded_title}"

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # 提取摘要
            extract = data.get('extract', '')

            # 如果摘要太短，尝试获取更多内容
            if len(extract) < 100:
                full_content = self.get_full_page_content(title)
                if full_content:
                    return full_content[:1000]  # 限制长度

            return extract

        except Exception as e:
            logger.warning(f"Failed to get content for '{title}': {e}")
            return None
        finally:
            time.sleep(self.delay)  # 避免请求过于频繁

    def get_full_page_content(self, title: str) -> Optional[str]:
        """获取完整页面内容"""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
            'exsectionformat': 'plain'
        }

        try:
            response = self.session.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            pages = data.get('query', {}).get('pages', {})

            for page_id, page_data in pages.items():
                if page_id != '-1':  # 页面存在
                    return page_data.get('extract', '')

            return None

        except Exception as e:
            logger.warning(f"Failed to get full content for '{title}': {e}")
            return None

class CachedWikipediaSearch(SearchEngine):
    """带缓存的Wikipedia搜索引擎"""

    def __init__(self, cache_dir: str = './cache/wikipedia', language: str = 'en'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.language = language
        self.api = WikipediaAPI(language)

        # 缓存文件
        self.title_cache_file = self.cache_dir / f'title_cache_{language}.json'
        self.content_cache_file = self.cache_dir / f'content_cache_{language}.pkl'

        # 加载缓存
        self.title_cache = self.load_title_cache()
        self.content_cache = self.load_content_cache()

    def load_title_cache(self) -> Dict[str, List[str]]:
        """加载标题缓存"""
        if self.title_cache_file.exists():
            try:
                with open(self.title_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load title cache: {e}")
        return {}

    def save_title_cache(self):
        """保存标题缓存"""
        try:
            with open(self.title_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.title_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save title cache: {e}")

    def load_content_cache(self) -> Dict[str, str]:
        """加载内容缓存"""
        if self.content_cache_file.exists():
            try:
                with open(self.content_cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load content cache: {e}")
        return {}

    def save_content_cache(self):
        """保存内容缓存"""
        try:
            with open(self.content_cache_file, 'wb') as f:
                pickle.dump(self.content_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save content cache: {e}")

    def search_titles_cached(self, query: str, limit: int = 10) -> List[str]:
        """缓存的标题搜索"""
        cache_key = f"{query.lower()}:{limit}"

        if cache_key in self.title_cache:
            return self.title_cache[cache_key]

        # 从API获取
        titles = self.api.search_titles(query, limit)

        # 缓存结果
        self.title_cache[cache_key] = titles

        # 定期保存缓存
        if len(self.title_cache) % 100 == 0:
            self.save_title_cache()

        return titles

    def get_content_cached(self, title: str) -> Optional[str]:
        """缓存的内容获取"""
        if title in self.content_cache:
            return self.content_cache[title]

        # 从API获取
        content = self.api.get_page_content(title)

        if content:
            # 缓存结果
            self.content_cache[title] = content

            # 定期保存缓存
            if len(self.content_cache) % 50 == 0:
                self.save_content_cache()

        return content

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """搜索相关文档"""
        if not query.strip():
            return []

        try:
            # 搜索相关标题
            titles = self.search_titles_cached(query, limit=top_k * 2)

            # 获取内容
            documents = []
            for title in titles[:top_k]:
                content = self.get_content_cached(title)
                if content and len(content.strip()) > 50:  # 过滤太短的内容
                    # 格式化文档
                    doc = f"Title: {title}\n{content}"
                    documents.append(doc)

                if len(documents) >= top_k:
                    break

            return documents

        except Exception as e:
            logger.error(f"Wikipedia search failed for query '{query}': {e}")
            return []

    def add_documents(self, documents: List[str]) -> None:
        """添加文档到搜索引擎（Wikipedia不支持此操作）"""
        logger.warning("Adding documents to Wikipedia search is not supported")

    def clear_cache(self):
        """清空缓存"""
        self.title_cache.clear()
        self.content_cache.clear()

        # 删除缓存文件
        if self.title_cache_file.exists():
            self.title_cache_file.unlink()
        if self.content_cache_file.exists():
            self.content_cache_file.unlink()

    def cache_statistics(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'title_cache_size': len(self.title_cache),
            'content_cache_size': len(self.content_cache)
        }

    def preload_content(self, titles: List[str]):
        """预加载内容"""
        logger.info(f"Preloading content for {len(titles)} titles...")

        for i, title in enumerate(titles):
            if title not in self.content_cache:
                content = self.api.get_page_content(title)
                if content:
                    self.content_cache[title] = content

            if (i + 1) % 10 == 0:
                logger.info(f"Preloaded {i + 1}/{len(titles)} titles")

        # 保存缓存
        self.save_content_cache()
        logger.info("Preloading completed")

    def __del__(self):
        """析构函数，保存缓存"""
        try:
            self.save_title_cache()
            self.save_content_cache()
        except:
            pass

class OfflineWikipediaSearch(SearchEngine):
    """离线Wikipedia搜索引擎"""

    def __init__(self, dump_path: str, index_path: str = None):
        self.dump_path = Path(dump_path)
        self.index_path = Path(index_path) if index_path else self.dump_path.parent / 'wiki_index'

        self.articles = {}
        self.index = None

        # 加载数据
        self.load_wikipedia_dump()
        self.build_search_index()

    def load_wikipedia_dump(self):
        """加载Wikipedia dump"""
        logger.info(f"Loading Wikipedia dump from {self.dump_path}")

        if self.dump_path.is_file():
            # 单个文件
            self.load_dump_file(self.dump_path)
        elif self.dump_path.is_dir():
            # 目录中的多个文件
            for file_path in self.dump_path.rglob("*.jsonl"):
                self.load_dump_file(file_path)
            for file_path in self.dump_path.rglob("*.json"):
                self.load_dump_file(file_path)

        logger.info(f"Loaded {len(self.articles)} articles")

    def load_dump_file(self, file_path: Path):
        """加载单个dump文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # JSONL格式
                    for line in f:
                        try:
                            article = json.loads(line.strip())
                            if 'title' in article and 'text' in article:
                                self.articles[article['title']] = article['text']
                        except json.JSONDecodeError:
                            continue
                else:
                    # JSON格式
                    data = json.load(f)
                    if isinstance(data, list):
                        for article in data:
                            if 'title' in article and 'text' in article:
                                self.articles[article['title']] = article['text']
                    elif isinstance(data, dict):
                        for title, text in data.items():
                            self.articles[title] = text

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    def build_search_index(self):
        """构建搜索索引"""
        if not self.articles:
            logger.warning("No articles loaded, cannot build index")
            return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import pickle

            # 检查是否已有索引
            index_file = self.index_path / 'tfidf_index.pkl'
            vectorizer_file = self.index_path / 'vectorizer.pkl'
            titles_file = self.index_path / 'titles.pkl'

            if (index_file.exists() and vectorizer_file.exists() and titles_file.exists()):
                logger.info("Loading existing search index...")
                with open(index_file, 'rb') as f:
                    self.index = pickle.load(f)
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(titles_file, 'rb') as f:
                    self.titles = pickle.load(f)
                logger.info("Search index loaded")
                return

            logger.info("Building search index...")
            self.index_path.mkdir(parents=True, exist_ok=True)

            # 准备文档
            self.titles = list(self.articles.keys())
            documents = [self.articles[title] for title in self.titles]

            # 构建TF-IDF索引
            self.vectorizer = TfidfVectorizer(
                max_features=50000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )

            self.index = self.vectorizer.fit_transform(documents)

            # 保存索引
            with open(index_file, 'wb') as f:
                pickle.dump(self.index, f)
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(titles_file, 'wb') as f:
                pickle.dump(self.titles, f)

            logger.info("Search index built and saved")

        except ImportError:
            logger.warning("scikit-learn not available, using simple search")
            self.index = None
            self.vectorizer = None
            self.titles = list(self.articles.keys())

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """搜索相关文档"""
        if not query.strip() or not self.articles:
            return []

        if self.index is not None and self.vectorizer is not None:
            return self.search_with_tfidf(query, top_k)
        else:
            return self.search_simple(query, top_k)

    def search_with_tfidf(self, query: str, top_k: int) -> List[str]:
        """使用TF-IDF搜索"""
        try:
            # 向量化查询
            query_vector = self.vectorizer.transform([query])

            # 计算相似度
            similarities = cosine_similarity(query_vector, self.index)[0]

            # 获取top_k结果
            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # 最小相关度阈值
                    title = self.titles[idx]
                    content = self.articles[title]
                    # 截断内容
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    results.append(f"Title: {title}\n{content}")

            return results

        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return self.search_simple(query, top_k)

    def search_simple(self, query: str, top_k: int) -> List[str]:
        """简单搜索（关键词匹配）"""
        query_words = set(query.lower().split())

        scored_articles = []
        for title, content in self.articles.items():
            # 计算标题和内容的匹配度
            title_words = set(title.lower().split())
            content_words = set(content.lower().split())

            title_score = len(query_words & title_words) / len(query_words) if query_words else 0
            content_score = len(query_words & content_words) / len(query_words) if query_words else 0

            total_score = 2 * title_score + content_score

            if total_score > 0:
                scored_articles.append((total_score, title, content))

        # 排序并返回top_k
        scored_articles.sort(reverse=True)

        results = []
        for score, title, content in scored_articles[:top_k]:
            if len(content) > 1000:
                content = content[:1000] + "..."
            results.append(f"Title: {title}\n{content}")

        return results

    def add_documents(self, documents: List[str]) -> None:
        """添加文档（不支持）"""
        logger.warning("Adding documents to offline Wikipedia search is not supported")

    def get_article(self, title: str) -> Optional[str]:
        """获取特定文章"""
        return self.articles.get(title)

    def list_articles(self, limit: int = 10) -> List[str]:
        """列出文章标题"""
        return list(self.articles.keys())[:limit]

    def statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_articles': len(self.articles),
            'has_index': self.index is not None,
            'index_path': str(self.index_path)
        }

def create_wikipedia_search(config: Dict[str, Any]) -> SearchEngine:
    """创建Wikipedia搜索引擎"""
    search_type = config.get('wikipedia_type', 'cached')

    if search_type == 'cached':
        return CachedWikipediaSearch(
            cache_dir=config.get('cache_dir', './cache/wikipedia'),
            language=config.get('language', 'en')
        )
    elif search_type == 'offline':
        return OfflineWikipediaSearch(
            dump_path=config.get('dump_path', './musique/wikipedia'),
            index_path=config.get('index_path', None)
        )
    else:
        raise ValueError(f"Unknown Wikipedia search type: {search_type}")