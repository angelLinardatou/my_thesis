import pandas as pd
from pathlib import Path

class DataLoaderManager:
    """Load and preprocess dataset files for Greek sentiment and emotion tasks."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def load_dataset(self, filename: str):
        """Load CSV dataset from base directory."""
        file_path = self.base_dir / filename
        df = pd.read_csv(file_path)
        return df

    def map_labels(self, df, mapping_column, label_column, mapping_dict):
        """Map text labels to numerical format."""
        df[label_column] = df[mapping_column].map(mapping_dict)
        return df
 
