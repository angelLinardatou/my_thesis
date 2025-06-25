import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

class AgreementCalculator:
    """Compute pairwise inter-annotator agreement using Cohen's Kappa."""

    def __init__(self, annotations: pd.DataFrame, emotion: str):
        self.annotations = annotations
        self.emotion = emotion

    def compute_kappa_matrix(self):
        """Compute upper triangular matrix of pairwise Kappa scores."""
        n_annotators = self.annotations.shape[1]
        kappa_matrix = np.full((n_annotators, n_annotators), np.nan)

        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                 y1 = self.annotations.iloc[:, i]
                 y2 = self.annotations.iloc[:, j]

                 # Clean NaN
                 mask = y1.notna() & y2.notna()
                 y1 = y1[mask].astype(str)
                 y2 = y2[mask].astype(str)

                 kappa = cohen_kappa_score(y1, y2)

                 kappa_matrix[i, j] = kappa

        return kappa_matrix
