import pandas as pd
from pathlib import Path

class DataLoader:
    """Load and preprocess multi-label emotion dataset."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def load_dataset(self, filename: str):
        """Load Excel dataset from base directory."""
        file_path = self.base_dir / filename
        df = pd.read_excel(file_path)
        return df
 
