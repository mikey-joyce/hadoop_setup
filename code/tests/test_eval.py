"""
This script tests the compute_metric function from eval.py on sample logits and labels
for a 3-class classification problem. It prints out the computed metric output.
"""

import sys
import os

# Add the parent directory to sys.path so we can import eval.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from eval import compute_metric, compute_f1_accuracy

def main():
    # Create sample logits and labels for a 3-class classification problem.
    # The logits can take any real value; here we choose example values.
    predictions = np.array([
        [ -1.2,  2.3,  0.5 ],  # Sample 1 logits
        [  3.1, -0.2, -1.5 ],  # Sample 2 logits
        [  0.1,  0.2,  1.7 ],  # Sample 3 logits
        [  2.0,  0.1, -0.3 ]   # Sample 4 logits
    ])
    # True labels for these samples.
    labels = np.array([1, 0, 1, 0])
    
    # Call compute_metric with a tuple of (predictions, labels)
    metrics = compute_metric((predictions, labels))
    more_metrics= compute_f1_accuracy((predictions, labels))
    
    print("Computed metrics:")
    print(metrics)
    
    print("F1 and accuracy metrics:")
    print(more_metrics)

if __name__ == "__main__":
    main()