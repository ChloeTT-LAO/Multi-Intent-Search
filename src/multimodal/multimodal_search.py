@abstractmethod
def search_multimodal(self, text_query: str = None, image_query: Union[str, Image.Image] = None,
                      top_k: int = 3) -> List[Dict[str, Any]]:
    """多模态搜索"""
    pass


def search(self, query: str, top_k: int = 3) -> List[str]:
    """兼容原有接口的文本搜索"""
    results = self.search_multimodal(text_query=query, top_k=top_k)
    return [result.get('text', '') for result in results]


class ImageTextSearchEngine(MultimodalSearchEngine):
    """图像-文本搜索引擎"""

    def __init__(self, documents: List[Dict[str, Any]] = None):
        """
        初始化
        documents: 包含'text'和可选'image'字段的文档列表
        """
        self.documents = documents or []
        self.text_search_engine = None
        self.image_encoder = None

        # 初始化文本搜索
        self.setup_text_search()

        # 初始化图像搜索
        self.setup_image_search()

    def setup_text_search(self):
        """设置文本搜索"""
        from ..search.search_engine import TFIDFSearchEngine

        texts = [doc.get('text', '') for doc in self.documents]
        if texts:
            self.text_search_engine = TFIDFSearchEngine(texts)

    def setup_image_search(self):
        """设置图像搜索"""
        try:
            import clip
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

            # 预计算图像embeddings
            self.image_embeddings = []
            self.setup_image_embeddings()

        except ImportError:
            print("CLIP not available, image search disabled")
            self.clip_model = None

    def setup_image_embeddings(self):
        """预计算图像embeddings"""
        if self.clip_model is None:
            return

        self.image_embeddings = []
        device = next(self.clip_model.parameters()).device

        for doc in self.documents:
            if 'image' in doc and doc['image'] is not None:
                try:
                    image = self.process_image(doc['image'])
                    image_tensor = self.clip_preprocess(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = self.clip_model.encode_image(image_tensor)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        self.image_embeddings.append(embedding.cpu().numpy())
                except Exception as e:
                    print(f"Error processing image: {e}")
                    self.image_embeddings.append(None)
            else:
                self.image_embeddings.append(None)

    def process_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """处理图像输入"""
        if isinstance(image_input, str):
            if image_input.startswith('data:image'):
                # Base64编码
                header, data = image_input.split(',', 1)
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
            else:
                # 文件路径
                image = Image.open(image_input)
        else:
            image = image_input

        return image.convert('RGB')

    def search_by_text(self, text_query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """文本搜索"""
        if self.text_search_engine is None:
            return []

        results = self.text_search_engine.search(text_query, top_k=len(self.documents))

        # 计算相似度分数
        scored_results = []
        for i, result_text in enumerate(results):
            # 找到对应的文档索引
            for doc_idx, doc in enumerate(self.documents):
                if doc.get('text', '') == result_text:
                    scored_results.append((doc_idx, 1.0 - i / len(results)))
                    break

        return scored_results[:top_k]

    def search_by_image(self, image_query: Union[str, Image.Image], top_k: int = 3) -> List[Tuple[int, float]]:
        """图像搜索"""
        if self.clip_model is None:
            return []

        try:
            # 处理查询图像
            query_image = self.process_image(image_query)

            device = next(self.clip_model.parameters()).device
            query_tensor = self.clip_preprocess(query_image).unsqueeze(0).to(device)

            # 编码查询图像
            with torch.no_grad():
                query_embedding = self.clip_model.encode_image(query_tensor)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
                query_embedding = query_embedding.cpu().numpy()

            # 计算相似度
            similarities = []
            for i, img_embedding in enumerate(self.image_embeddings):
                if img_embedding is not None:
                    similarity = np.dot(query_embedding, img_embedding.T)[0][0]
                    similarities.append((i, similarity))
                else:
                    similarities.append((i, 0.0))

            # 排序并返回top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            print(f"Image search error: {e}")
            return []

    def search_multimodal(self, text_query: str = None, image_query: Union[str, Image.Image] = None,
                          top_k: int = 3) -> List[Dict[str, Any]]:
        """多模态搜索"""
        if not text_query and not image_query:
            return []

        all_scores = {}

        # 文本搜索
        if text_query:
            text_results = self.search_by_text(text_query, top_k * 2)
            for doc_idx, score in text_results:
                all_scores[doc_idx] = all_scores.get(doc_idx, 0) + 0.5 * score

        # 图像搜索
        if image_query:
            image_results = self.search_by_image(image_query, top_k * 2)
            for doc_idx, score in image_results:
                all_scores[doc_idx] = all_scores.get(doc_idx, 0) + 0.5 * score

        # 排序并返回结果
        sorted_results = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_idx, score in sorted_results[:top_k]:
            result = {
                'text': self.documents[doc_idx].get('text', ''),
                'image': self.documents[doc_idx].get('image', None),
                'score': score,
                'doc_index': doc_idx
            }
            final_results.append(result)

        return final_results

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档"""
        self.documents.extend(documents)
        # 重新设置搜索引擎
        self.setup_text_search()
        if self.clip_model is not None:
            self.setup_image_embeddings()


class CLIPSearchEngine(MultimodalSearchEngine):
    """基于CLIP的搜索引擎"""

    def __init__(self, model_name: str = "ViT-B/32"):
        self.model_name = model_name
        self.documents = []
        self.text_embeddings = []
        self.image_embeddings = []

        # 加载CLIP模型
        self.setup_clip_model()

    def setup_clip_model(self):
        """设置CLIP模型"""
        try:
            import clip
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load(self.model_name, device=device)
            self.device = device

        except ImportError:
            print("CLIP not available")
            self.clip_model = None

    def encode_text(self, text: str) -> np.ndarray:
        """编码文本"""
        if self.clip_model is None:
            return np.zeros(512)

        try:
            import clip

            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_embedding = self.clip_model.encode_text(text_tokens)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            return text_embedding.cpu().numpy()[0]

        except Exception as e:
            print(f"Text encoding error: {e}")
            return np.zeros(512)

    def encode_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """编码图像"""
        if self.clip_model is None:
            return np.zeros(512)

        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                return np.zeros(512)

            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_embedding = self.clip_model.encode_image(image_tensor)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

            return image_embedding.cpu().numpy()[0]

        except Exception as e:
            print(f"Image encoding error: {e}")
            return np.zeros(512)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档并计算embeddings"""
        for doc in documents:
            self.documents.append(doc)

            # 编码文本
            text = doc.get('text', '')
            text_embedding = self.encode_text(text)
            self.text_embeddings.append(text_embedding)

            # 编码图像
            image = doc.get('image', None)
            if image is not None:
                image_embedding = self.encode_image(image)
                self.image_embeddings.append(image_embedding)
            else:
                self.image_embeddings.append(np.zeros(512))

    def search_multimodal(self, text_query: str = None, image_query: Union[str, Image.Image] = None,
                          top_k: int = 3) -> List[Dict[str, Any]]:
        """CLIP多模态搜索"""
        if not self.documents:
            return []

        if not text_query and not image_query:
            return []

        query_embedding = np.zeros(512)

        # 编码文本查询
        if text_query:
            text_emb = self.encode_text(text_query)
            query_embedding += text_emb

        # 编码图像查询
        if image_query:
            image_emb = self.encode_image(image_query)
            query_embedding += image_emb

        # 标准化查询embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # 计算与所有文档的相似度
        similarities = []
        for i, (text_emb, image_emb) in enumerate(zip(self.text_embeddings, self.image_embeddings)):
            # 计算文本相似度
            text_sim = np.dot(query_embedding, text_emb)

            # 计算图像相似度
            image_sim = np.dot(query_embedding, image_emb) if np.any(image_emb) else 0

            # 组合相似度
            combined_sim = 0.6 * text_sim + 0.4 * image_sim
            similarities.append((i, combined_sim))

        # 排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in similarities[:top_k]:
            result = {
                'text': self.documents[doc_idx].get('text', ''),
                'image': self.documents[doc_idx].get('image', None),
                'score': score,
                'doc_index': doc_idx
            }
            results.append(result)

        return results


class HybridMultimodalSearchEngine(MultimodalSearchEngine):
    """混合多模态搜索引擎"""

    def __init__(self, search_engines: List[MultimodalSearchEngine], weights: List[float] = None):
        self.search_engines = search_engines
        self.weights = weights or [1.0] * len(search_engines)

        if len(self.weights) != len(self.search_engines):
            raise ValueError("Number of weights must match number of search engines")

    def search_multimodal(self, text_query: str = None, image_query: Union[str, Image.Image] = None,
                          top_k: int = 3) -> List[Dict[str, Any]]:
        """混合多模态搜索"""
        all_results = {}

        for engine, weight in zip(self.search_engines, self.weights):
            try:
                results = engine.search_multimodal(text_query, image_query, top_k * 2)

                for result in results:
                    # 使用文档的唯一标识符（这里简化为文本）
                    doc_key = result.get('text', '')[:100]  # 使用前100个字符作为key

                    if doc_key not in all_results:
                        all_results[doc_key] = result.copy()
                        all_results[doc_key]['score'] = result['score'] * weight
                    else:
                        all_results[doc_key]['score'] += result['score'] * weight

            except Exception as e:
                print(f"Error in search engine: {e}")
                continue

        # 排序并返回top_k
        sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
        return sorted_results[:top_k]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """向所有搜索引擎添加文档"""
        for engine in self.search_engines:
            try:
                engine.add_documents(documents)
            except Exception as e:
                print(f"Error adding documents to engine: {e}")


def create_multimodal_search_engine(config: Dict[str, Any]) -> MultimodalSearchEngine:
    """创建多模态搜索引擎"""
    search_config = config['search']
    engine_type = search_config.get('engine_type', 'imagetext')

    if engine_type == 'imagetext':
        # 加载多模态文档
        documents_path = search_config.get('multimodal_documents_path', None)
        documents = []

        if documents_path:
            import json
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)

        return ImageTextSearchEngine(documents)

    elif engine_type == 'clip':
        model_name = search_config.get('clip_model', 'ViT-B/32')
        engine = CLIPSearchEngine(model_name)

        # 加载文档
        documents_path = search_config.get('multimodal_documents_path', None)
        if documents_path:
            import json
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            engine.add_documents(documents)

        return engine

    elif engine_type == 'hybrid_multimodal':
        engines = []
        weights = search_config.get('weights', [])

        for i, sub_config in enumerate(search_config['engines']):
            sub_engine = create_multimodal_search_engine({'search': sub_config})
            engines.append(sub_engine)

        return HybridMultimodalSearchEngine(engines, weights if weights else None)

    else:
        # 回退到文本搜索引擎
        from ..search.search_engine import create_search_engine
        text_engine = create_search_engine(config)

        # 包装为多模态搜索引擎
        class TextOnlyMultimodalWrapper(MultimodalSearchEngine):
            def __init__(self, text_engine):
                self.text_engine = text_engine

            def search_multimodal(self, text_query=None, image_query=None, top_k=3):
                if text_query:
                    texts = self.text_engine.search(text_query, top_k)
                    return [{'text': text, 'image': None, 'score': 1.0, 'doc_index': i}
                            for i, text in enumerate(texts)]
                return []

            def add_documents(self, documents):
                texts = [doc.get('text', '') for doc in documents]
                self.text_engine.add_documents(texts)

        return TextOnlyMultimodalWrapper(text_engine)