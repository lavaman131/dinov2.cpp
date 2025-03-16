import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List
import torch
from efficient_cv.evaluate.utils import read_ground_truth_from_csv
import torch.nn.functional as F
import warnings

from typing import Sequence, Optional, Mapping, Text
import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F


def calculate_accuracy(
    outputs: torch.Tensor, file_names: List[str], ground_truth_path: Union[str, Path]
) -> float:
    # Read the ground truth indices from the provided CSV file
    ground_truth_df = read_ground_truth_from_csv(csv_file=ground_truth_path)

    probs = F.softmax(outputs, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    preds = preds.numpy()

    ground_truth_df.set_index("file_name", inplace=True)

    correct = np.sum(preds == ground_truth_df.loc[file_names]["class_index"].to_numpy())
    total = ground_truth_df.shape[0]

    accuracy = correct / total
    print(f"Correct predictions: {correct}/{total}")
    return accuracy
