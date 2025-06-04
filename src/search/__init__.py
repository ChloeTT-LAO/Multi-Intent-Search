"""
StepSearch搜索引擎模块
"""

from .search_engine import (
    SearchEngine,
    TFIDFSearchEngine,
    WikipediaSearchEngine,
    MockSearchEngine,
    HybridSearchEngine,
    create_search_engine,
    build_knowledge_base_from_datasets
)

from .wiki_search import (
    WikipediaAPI,
    CachedWikipediaSearch,
    OfflineWikipediaSearch,
    create_wikipedia_search
)

from .mock_search import (
    MockSearchEngine,
    AdvancedMockSearchEngine,
    DomainMockSearchEngine,
    ConfigurableMockSearchEngine,
    InteractiveMockSearchEngine,
    create_mock_search_engine
)

__all__ = [
    # 基础搜索引擎
    'SearchEngine',
    'TFIDFSearchEngine',
    'WikipediaSearchEngine',
    'MockSearchEngine',
    'HybridSearchEngine',
    'create_search_engine',
    'build_knowledge_base_from_datasets',

    # Wikipedia搜索
    'WikipediaAPI',
    'CachedWikipediaSearch',
    'OfflineWikipediaSearch',
    'create_wikipedia_search',

    # 模拟搜索引擎
    'AdvancedMockSearchEngine',
    'DomainMockSearchEngine',
    'ConfigurableMockSearchEngine',
    'InteractiveMockSearchEngine',
    'create_mock_search_engine'
]