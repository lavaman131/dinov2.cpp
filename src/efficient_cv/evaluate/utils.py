import pandas as pd
from typing import Union
from pathlib import Path


def read_ground_truth_from_csv(csv_file: Union[str, Path]) -> pd.DataFrame:
    # Read the CSV file into a DataFrame
    ground_truth_data = pd.read_csv(csv_file)
    assert {"file_name", "label", "class_index"}.issubset(ground_truth_data.columns), (
        "CSV file must contain 'file_name', 'label', and 'class_index' columns"
    )

    # Drop rows where 'class_index' is NaN (empty)
    ground_truth_data = ground_truth_data.dropna(subset=["class_index"])

    # Return the list of class indices
    return ground_truth_data
