import pytest
import numpy as np
from definition_4f6cbc89047f47e6a9aad48b02b39e29 import calculate_performance_metrics

# Helper function for comparing float values with a tolerance
def assert_metrics_almost_equal(actual, expected, tolerance=1e-9):
    for key in expected:
        assert key in actual, f"Expected key '{key}' not found in actual metrics."
        if isinstance(expected[key], float):
            assert actual[key] == pytest.approx(expected[key], rel=tolerance), \
                f"Metric '{key}' mismatch. Expected {expected[key]}, got {actual[key]}."
        else:
            assert actual[key] == expected[key], \
                f"Metric '{key}' mismatch. Expected {expected[key]}, got {actual[key]}."

@pytest.mark.parametrize(
    "true_labels_input, predicted_labels_input, positive_class, expected_result",
    [
        # Test Case 1: Standard Balanced Case with mixed labels and some prediction errors
        # Expected: TP=2, FN=1, FP=1, TN=2
        # Accuracy: (2+2)/6 = 0.666..., FPR: 1/(1+2) = 0.333..., FNR: 1/(1+2) = 0.333...
        (np.array(['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Negative']),
         np.array(['Positive', 'Negative', 'Negative', 'Positive', 'Positive', 'Negative']),
         'Positive',
         {'Accuracy': 4/6, 'False Positive Rate': 1/3, 'False Negative Rate': 1/3}),

        # Test Case 2: Perfect Prediction - No errors at all
        # Expected: TP=2, FN=0, FP=0, TN=2
        # Accuracy: 1.0, FPR: 0.0, FNR: 0.0
        (np.array(['Positive', 'Negative', 'Positive', 'Negative']),
         np.array(['Positive', 'Negative', 'Positive', 'Negative']),
         'Positive',
         {'Accuracy': 1.0, 'False Positive Rate': 0.0, 'False Negative Rate': 0.0}),

        # Test Case 3: All Misclassified - Worst-case scenario where every prediction is wrong
        # Expected: TP=0, FN=2, FP=2, TN=0
        # Accuracy: 0.0, FPR: 1.0, FNR: 1.0
        (np.array(['Positive', 'Negative', 'Positive', 'Negative']),
         np.array(['Negative', 'Positive', 'Negative', 'Positive']),
         'Positive',
         {'Accuracy': 0.0, 'False Positive Rate': 1.0, 'False Negative Rate': 1.0}),

        # Test Case 4: Edge Case - Only 'Positive' class present in true labels (no true negatives)
        # Expected: TP=2, FN=1, FP=0, TN=0
        # Accuracy: (2+0)/3 = 2/3, FPR: 0/(0+0) -> 0.0 (as per function's division by zero handling), FNR: 1/(1+2) = 1/3
        (np.array(['Positive', 'Positive', 'Positive']),
         np.array(['Positive', 'Negative', 'Positive']),
         'Positive',
         {'Accuracy': 2/3, 'False Positive Rate': 0.0, 'False Negative Rate': 1/3}),

        # Test Case 5: Invalid input type for labels (e.g., None instead of np.ndarray)
        # Expecting a TypeError from sklearn functions or array conversion
        (None, np.array(['Positive', 'Negative']), 'Positive', TypeError),
    ]
)
def test_calculate_performance_metrics(true_labels_input, predicted_labels_input, positive_class, expected_result):
    if isinstance(expected_result, type) and issubclass(expected_result, Exception):
        with pytest.raises(expected_result):
            calculate_performance_metrics(true_labels_input, predicted_labels_input, positive_class)
    else:
        actual_metrics = calculate_performance_metrics(true_labels_input, predicted_labels_input, positive_class)
        assert_metrics_almost_equal(actual_metrics, expected_result)