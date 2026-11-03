"""Small feature pipelines; transformer inference is cached and never fine-tuned."""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def example_text(row):
    if row["task"] == 1:
        left = max(0, row["start"] - 500)
        right = min(len(row["text"]), row["end_exclusive"] + 500)
        return row["mention"] + "\n" + row["text"][left:right]
    return row["text"]


class FrozenEncoder:
    def __init__(self, cache=".cache/embeddings", model=ENCODER, revision=ENCODER_REVISION):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.revision = revision
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, revision=self.revision, device="cpu")
        return self._model

    def encode_one(self, text):
        key = hashlib.sha256((self.model_name + "|" + self.revision + "|token-chunks-v2|" + text).encode()).hexdigest()
        path = self.cache / (key + ".npy")
        if path.exists():
            return np.load(path)
        tokenizer = self.model.tokenizer
        # Fast-tokenizer overflow retains every token and correctly applies the
        # backend special-token processor (generic prepare_for_model does not).
        encoded = tokenizer(text, truncation=True, max_length=self.model.max_seq_length,
                            return_overflowing_tokens=True, return_special_tokens_mask=True)
        chunks = [{key: encoded[key][index] for key in tokenizer.model_input_names if key in encoded}
                  for index in range(len(encoded["input_ids"]))]
        weights = [max(1, sum(value == 0 for value in mask)) for mask in encoded["special_tokens_mask"]]
        import torch
        vectors = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(chunks), 8):
                batch = chunks[start:start + 8]
                features = tokenizer.pad(batch, padding=True, return_tensors="pt")
                vectors.extend(self.model(features)["sentence_embedding"].cpu().numpy())
        vector = np.average(vectors, axis=0, weights=weights).astype("float32")
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
            return get_encoder().encode(texts)
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
            return get_encoder().encode(texts)
        matrix = self.vectorizer.transform(texts)
        return Normalizer().fit_transform(self.svd.transform(matrix)) if self.svd else matrix


@lru_cache(maxsize=1)
def get_encoder():
    """Reuse model weights across articles and experiments, not inside pickles."""
    return FrozenEncoder()

