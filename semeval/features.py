"""Small feature pipelines; transformer inference is cached and never fine-tuned."""
import hashlib
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def example_text(row):
    if row["task"] == 1:
        left = max(0, row["start"] - 500)
        right = min(len(row["text"]), row["end_exclusive"] + 500)
        return row["mention"] + "\n" + row["text"][left:right]
    return row["text"]


class FrozenEncoder:
    def __init__(self, cache=".cache/embeddings", model=ENCODER):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode_one(self, text):
        key = hashlib.sha256((self.model_name + "|token-chunks-v1|" + text).encode()).hexdigest()
        path = self.cache / (key + ".npy")
        if path.exists():
            return np.load(path)
        tokenizer = self.model.tokenizer
        budget = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(" ", add_special_tokens=False) or [tokenizer.unk_token_id]
        # Use token IDs directly: decoding/re-tokenizing can expand and truncate chunks.
        chunks = [ids[start:start + budget] for start in range(0, len(ids), budget)]
        import torch
        vectors = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(chunks), 8):
                batch = chunks[start:start + 8]
                features = tokenizer.pad(
                    [tokenizer.prepare_for_model(chunk, add_special_tokens=True) for chunk in batch],
                    padding=True, return_tensors="pt")
                vectors.extend(self.model(features)["sentence_embedding"].cpu().numpy())
        vector = np.average(vectors, axis=0, weights=[len(chunk) for chunk in chunks]).astype("float32")
        vector /= max(np.linalg.norm(vector), 1e-12)
        np.save(path, vector)
        return vector

    def encode(self, texts):
        return np.stack([self.encode_one(text) for text in texts])


class Features:
    def __init__(self, method, seed=42):
        self.method = method
        self.seed = seed
        self.vectorizer = None
        self.svd = None

    def fit_transform(self, texts):
        if self.method == "transformer":
            return FrozenEncoder().encode(texts)
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                        max_features=50000, sublinear_tf=True, dtype=np.float32)
        matrix = self.vectorizer.fit_transform(texts)
        if self.method == "lsa":
            dimensions = min(256, matrix.shape[0] - 1, matrix.shape[1] - 1)
            if dimensions < 1:
                raise ValueError("LSA needs at least two training examples and features")
            self.svd = TruncatedSVD(n_components=dimensions, random_state=self.seed)
            return Normalizer().fit_transform(self.svd.fit_transform(matrix))
        return matrix

    def transform(self, texts):
        if self.method == "transformer":
            return FrozenEncoder().encode(texts)
        matrix = self.vectorizer.transform(texts)
        return Normalizer().fit_transform(self.svd.transform(matrix)) if self.svd else matrix

