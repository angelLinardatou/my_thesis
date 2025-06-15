import re

class TextCleaner:
    """Clean text by removing unnecessary characters."""

    def clean_text(self, text: str) -> str:
        """Apply basic text cleaning rules."""
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Zα-ωΑ-Ω\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
 
