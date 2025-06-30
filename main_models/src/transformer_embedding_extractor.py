import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingExtractor:
    """Extract CLS embeddings from transformer models."""

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def extract_embeddings(self, texts):
        """Generate embeddings for given texts."""
        embeddings = []
        for text in tqdm(texts, desc="Extracting embeddings"):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
        return np.array(embeddings)
 
