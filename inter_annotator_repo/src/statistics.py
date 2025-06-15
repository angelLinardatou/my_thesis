import pandas as pd

class AnnotationStatistics:
    """Compute descriptive statistics for annotations."""

    def __init__(self, dataframes: dict):
        self.dataframes = dataframes

    def compute_statistics(self):
        """Compute describe() for each file."""
        stats = {}
        for filename, df in self.dataframes.items():
            stats[filename] = df.describe()
        return stats
