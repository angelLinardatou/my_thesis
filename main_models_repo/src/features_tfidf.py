from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_vectorizer(max_features=5000):
    """Create a TfidfVectorizer with specified max features."""
    return TfidfVectorizer(max_features=max_features)

def fit_transform_tfidf(vectorizer, texts):
    """Fit the TF-IDF vectorizer and transform the training texts."""
    return vectorizer.fit_transform(texts)

def transform_tfidf(vectorizer, texts):
    """Transform validation or test texts."""
    return vectorizer.transform(texts)

def get_tfidf_feature_names(vectorizer):
    """Get TF-IDF feature names."""
    return vectorizer.get_feature_names_out()
