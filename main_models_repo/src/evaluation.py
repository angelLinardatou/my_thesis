import pandas as pd
from sklearn.metrics import classification_report 
from pathlib import Path

def evaluate_and_save(Y_true, Y_pred, label_names: list[str], output_path: Path) -> None:
    """Generate classification report and save to CSV."""
    report = classification_report(Y_true, Y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_path)
    print(report_df)
 
