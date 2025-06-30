import re

def clean_text(self, text: str) -> str:
    """
    Clean raw text using basic regular expression rules.

    This function performs the following steps:
    1. Converts all characters to lowercase.
    2. Removes URLs (e.g., starting with http).
    3. Removes any character that is not a letter (both Latin and Greek alphabets are preserved).
    4. Removes extra whitespace (multiple spaces → one).

    Args:
        text (str): The input raw text string.

    Returns:
        str: The cleaned version of the input text.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Zα-ωΑ-Ω\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
