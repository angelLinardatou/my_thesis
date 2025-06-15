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
                kappa = cohen_kappa_score(self.annotations.iloc[:, i], self.annotations.iloc[:, j])
                kappa_matrix[i, j] = kappa

        return kappa_matrix
