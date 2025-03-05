import pandas as pd
import numpy as np
from scipy.special import softmax

def read_ground_truth_from_csv(csv_file):
    # Read the CSV file into a DataFrame
    ground_truth_data = pd.read_csv(csv_file)
    
    # Drop rows where 'class_index' is NaN (empty)
    ground_truth_data = ground_truth_data.dropna(subset=['class_index'])
    
    # Return the list of class indices
    return ground_truth_data['class_index'].tolist()

def evaluate_track1(output_array, output_dir, ground_truth_dir):    
    # Read the ground truth indices from the provided CSV file
    ground_truth_indices = read_ground_truth_from_csv(ground_truth_dir)

    # Initialize counters to track correct predictions
    correct = 0
    total = len(ground_truth_indices)

    # Compare predictions with ground truth
    for i, result in enumerate(output_array):
        # Apply softmax to the model output to get probabilities
        softmax_results = softmax(result)

        # Get the top prediction (class with highest probability)
        top_prediction = np.argmax(softmax_results)

        # Compare the top prediction to the ground truth
        if top_prediction == ground_truth_indices[i]:
            correct += 1 

    accuracy = correct / total
    print(f"Correct predictions: {correct}/{total}")
    return accuracy
