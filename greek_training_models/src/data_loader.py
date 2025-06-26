import pandas as pd
from pathlib import Path
    
def load_dataset(base_dir: Path, filename: str)-> pd.DataFrame:
       """Load CSV dataset from base directory."""
       file_path = base_dir / filename
       df = pd.read_csv(file_path)
       return df

def map_labels(df: pd.DataFrame, mapping_column: str, label_column: str, mapping_dict: dict)-> pd.DataFrame:
      """Map text labels to numerical format."""
      df[label_column] = df[mapping_column].map(mapping_dict)
      return df
 
