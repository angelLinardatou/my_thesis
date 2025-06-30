import pandas as pd


def compute_annotation_statistics(
    dataframes: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Compute describe() for each annotation file."""
    return {filename: df.describe() for filename, df in dataframes.items()}
