import pandas as pd
from pathlib import Path 

def load_dataset(base_dir: Path, filename: str)-> pd.DataFrame:
    """Load Excel dataset from base directory."""
    file_path = self.base_dir / filename
    df = pd.read_csv(file_path)
    return df
 
