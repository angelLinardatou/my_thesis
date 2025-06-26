import pandas as pd
from pathlib import Path


def load_annotation_files(
    annotation_dir: Path
) -> dict[str, pd.DataFrame]:
    """Load Excel files from annotation directory."""
    files = list(annotation_dir.glob("*.xlsx"))
    return {
        file.stem: pd.read_excel(file, header=1).rename(str.lower, axis=1)
        for file in files
    }
