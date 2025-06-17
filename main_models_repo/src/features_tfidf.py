from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFeatures:
    """Extract TF-IDF features from text data."""

    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        """Fit and transform training texts."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """Transform validation or test texts."""
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """Get TF-IDF feature names."""
        return self.vectorizer.get_feature_names_out()
 
