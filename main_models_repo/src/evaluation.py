import pandas as pd
from sklearn.metrics import classification_report

class Evaluator:
    """Evaluate multi-label classification models."""

    def __init__(self, label_names):
        self.label_names = label_names

    def evaluate_and_save(self, Y_true, Y_pred, output_path):
        """Generate classification report and save to CSV."""
        report = classification_report(Y_true, Y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(output_path)
        print(report_df)
 
