import pandas as pd
from pathlib import Path

class AnnotationLoader:
    """Load and prepare annotation Excel files."""

    def __init__(self, annotation_dir: Path):
        self.annotation_dir = annotation_dir

    def load_files(self):
        """Load Excel files from annotation directory."""
        xlsx_files = list(self.annotation_dir.glob('*.xlsx'))
        dataframes = {f.name: pd.read_excel(f, skiprows=1) for f in xlsx_files}
        for fname, df in dataframes.items():
            df.columns = ['id', 'text', 'anger', 'fear', 'joy', 'sadness', 'surprise']
        return dataframes
