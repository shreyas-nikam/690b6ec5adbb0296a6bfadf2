import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Keep the definition_f069157be4814d14bcb7bac51eb5cf17 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_f069157be4814d14bcb7bac51eb5cf17 import get_ai_predictions_and_confidence

# --- Mocks for testing ---
class MockRandomForestClassifier:
    """
    A mock class to simulate the behavior of sklearn.ensemble.RandomForestClassifier
    for testing purposes, providing predictable outputs for .predict(),
    .predict_proba(), and .classes_.
    """
    def __init__(self, classes_, predictions, probabilities):
        self.classes_ = np.array(classes_)
        self._predictions = np.array(predictions)
        self._probabilities = np.array(probabilities)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if X.empty:
                return np.array([])
            # Simulate prediction for each row in X
            return self._predictions[:len(X)] 
        raise TypeError("Input must be a pandas DataFrame or similar array-like structure.")

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            if X.empty:
                return np.empty((0, len(self.classes_)))
            # Simulate probabilities for each row in X
            return self._probabilities[:len(X)]
        raise TypeError("Input must be a pandas DataFrame or similar array-like structure.")

# --- Test Cases ---

def test_get_ai_predictions_and_confidence_happy_path():
    """
    Test case covering the expected functionality with valid inputs.
    Verifies correct predictions and positive class confidence scores.
    """
    # 1. Setup mock data and model
    test_features_df = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3],
        'feature2': [1, 2, 3]
    })
    positive_class = 'Positive'
    
    mock_classes = ['Negative', 'Positive']
    mock_predictions = ['Positive', 'Negative', 'Positive']
    mock_probabilities = [
        [0.1, 0.9],  # Case 1: Predict Positive (conf=0.9)
        [0.8, 0.2],  # Case 2: Predict Negative (Positive conf=0.2)
        [0.4, 0.6]   # Case 3: Predict Positive (conf=0.6)
    ]
    mock_model = MockRandomForestClassifier(mock_classes, mock_predictions, mock_probabilities)

    # 2. Call the function under test
    predictions, confidence = get_ai_predictions_and_confidence(mock_model, test_features_df, positive_class)

    # 3. Assertions
    np.testing.assert_array_equal(predictions, np.array(['Positive', 'Negative', 'Positive']))
    np.testing.assert_array_almost_equal(confidence, np.array([0.9, 0.2, 0.6]))
    assert isinstance(predictions, np.ndarray)
    assert isinstance(confidence, np.ndarray)
    assert predictions.shape == (3,)
    assert confidence.shape == (3,)

def test_get_ai_predictions_and_confidence_empty_dataframe():
    """
    Test case for an edge scenario where the input DataFrame of test features is empty.
    Expected: Should return empty numpy arrays for predictions and confidence.
    """
    # 1. Setup mock data and model for empty input
    test_features_df = pd.DataFrame(columns=['feature1', 'feature2'])
    positive_class = 'Positive'
    
    mock_classes = ['Negative', 'Positive']
    mock_predictions = [] # Expected predictions for empty input
    mock_probabilities = [] # Expected probabilities for empty input
    mock_model = MockRandomForestClassifier(mock_classes, mock_predictions, mock_probabilities)

    # 2. Call the function under test
    predictions, confidence = get_ai_predictions_and_confidence(mock_model, test_features_df, positive_class)

    # 3. Assertions
    np.testing.assert_array_equal(predictions, np.array([]))
    np.testing.assert_array_equal(confidence, np.array([]))
    assert predictions.shape == (0,)
    assert confidence.shape == (0,)

def test_get_ai_predictions_and_confidence_positive_class_not_found():
    """
    Test case for an edge scenario where the `positive_class` argument is not
    present in the model's `classes_` array.
    Expected: An IndexError because `np.where` will return an empty array,
              and attempting to access `[0][0]` will fail.
    """
    # 1. Setup mock data and model
    test_features_df = pd.DataFrame({'feature1': [0.1]})
    positive_class = 'DiagnosisC' # This class is not in mock_classes
    
    mock_classes = ['DiagnosisA', 'DiagnosisB']
    mock_predictions = ['DiagnosisA']
    mock_probabilities = [[0.7, 0.3]]
    mock_model = MockRandomForestClassifier(mock_classes, mock_predictions, mock_probabilities)

    # 2. Assert that an IndexError is raised
    with pytest.raises(IndexError) as excinfo:
        get_ai_predictions_and_confidence(mock_model, test_features_df, positive_class)
    assert "index 0 is out of bounds for axis 0 with size 0" in str(excinfo.value) # Specific error for [0][0] on empty result

def test_get_ai_predictions_and_confidence_invalid_model_object():
    """
    Test case for an edge scenario where an invalid object (e.g., None)
    is passed as the `model` argument.
    Expected: An AttributeError when trying to call `.predict()` or
              `.predict_proba()` on a non-model object.
    """
    # 1. Setup invalid model and mock data
    test_features_df = pd.DataFrame({'feature1': [0.1]})
    positive_class = 'Positive'
    invalid_model = None

    # 2. Assert that an AttributeError is raised
    with pytest.raises(AttributeError):
        get_ai_predictions_and_confidence(invalid_model, test_features_df, positive_class)

def test_get_ai_predictions_and_confidence_invalid_test_features_df_type():
    """
    Test case for an edge scenario where `test_features_df` is not a pandas DataFrame
    (e.g., a simple list or numpy array).
    Expected: A TypeError or AttributeError from the model's `predict` or `predict_proba`
              methods, as they expect DataFrame-like inputs.
    """
    # 1. Setup mock model and invalid test features
    invalid_test_features = [[1, 2], [3, 4]] # A list of lists, not a DataFrame
    positive_class = 'Positive'
    
    mock_classes = ['Negative', 'Positive']
    mock_predictions = ['Positive', 'Negative']
    mock_probabilities = [[0.1, 0.9], [0.8, 0.2]]
    mock_model = MockRandomForestClassifier(mock_classes, mock_predictions, mock_probabilities)

    # 2. Assert that a TypeError is raised by the mock model (or AttributeError by sklearn)
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame or similar array-like structure."):
        get_ai_predictions_and_confidence(mock_model, invalid_test_features, positive_class)