import numpy as np
from gensim.models import Word2Vec

class Word2VecFeatures:
    """Generate Word2Vec sentence embeddings."""

    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count)

    def train_word2vec(self, tokenized_texts):
        """Train Word2Vec model on tokenized texts."""
        self.model.build_vocab(tokenized_texts)
        self.model.train(tokenized_texts, total_examples=len(tokenized_texts), epochs=10)

    def get_sentence_embedding(self, tokens):
        """Compute mean pooled sentence embedding."""
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

    def transform_dataset(self, texts):
        """Generate embeddings for full dataset."""
        embeddings = np.vstack([self.get_sentence_embedding(text.split()) for text in texts])
        return embeddings
 
