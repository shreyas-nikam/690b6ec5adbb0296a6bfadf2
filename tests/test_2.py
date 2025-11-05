import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Keep the placeholder as requested
from definition_825ddc11e7a5458eaff8353689190cfd import train_ai_model

# Mock sklearn components for controlled testing environments.
# This avoids actual model training and data preprocessing, focusing on function logic.
@pytest.fixture
def mock_sklearn(mocker):
    # Mock RandomForestClassifier instance
    mock_rf_model = MagicMock()
    mocker.patch('sklearn.ensemble.RandomForestClassifier', return_value=mock_rf_model)

    # Mock OneHotEncoder instance
    mock_ohe_instance = MagicMock()
    # Dummy output for OHE: 4 rows (matching sample_dataframe), 5 columns (dummy categorical features)
    mock_ohe_instance.fit_transform.return_value = np.array([
        [0, 1, 1, 0, 0],  # Dummy OHE for sample_dataframe row 0
        [1, 0, 0, 0, 1],  # Dummy OHE for sample_dataframe row 1
        [0, 1, 0, 1, 0],  # Dummy OHE for sample_dataframe row 2
        [1, 0, 1, 0, 0]   # Dummy OHE for sample_dataframe row 3
    ])
    # Dummy feature names, including original categorical feature names
    mock_ohe_instance.get_feature_names_out.return_value = [
        'patient_gender_Female', 'patient_gender_Male', 
        'symptom_severity_Mild', 'symptom_severity_Moderate', 'symptom_severity_Severe'
    ]
    mocker.patch('sklearn.preprocessing.OneHotEncoder', return_value=mock_ohe_instance)

    # Mock ColumnTransformer instance
    mock_ct_instance = MagicMock()
    # Dummy processed features: OHE output + numerical features from sample_dataframe (patient_age, lab_result_A)
    # Total columns: 5 (OHE) + 2 (numerical) = 7
    mock_ct_instance.fit_transform.return_value = np.array([
        [0, 1, 1, 0, 0, 30, 50.5],
        [1, 0, 0, 0, 1, 45, 62.1],
        [0, 1, 0, 1, 0, 60, 40.0],
        [1, 0, 1, 0, 0, 25, 75.2]
    ])
    mock_ct_instance.named_transformers_ = {'cat': mock_ohe_instance} # Link to OHE mock for feature names
    mocker.patch('sklearn.compose.ColumnTransformer', return_value=mock_ct_instance)

    # Mock train_test_split. It must return actual pandas DataFrames/Series for type checks.
    # The return values here are based on a 70/30 split of a 4-row DataFrame, resulting in 2 train / 2 test samples.
    mock_X_train = pd.DataFrame({
        'pg_F': [0, 0], 'pg_M': [1, 1], 'ss_M': [1, 0], 'ss_Mod': [0, 1], 'ss_S': [0, 0],
        'age': [30, 60], 'lab_A': [50.5, 40.0]
    }, index=[0, 2])
    mock_X_test = pd.DataFrame({
        'pg_F': [1, 1], 'pg_M': [0, 0], 'ss_M': [0, 1], 'ss_Mod': [0, 0], 'ss_S': [1, 0],
        'age': [45, 25], 'lab_A': [62.1, 75.2]
    }, index=[1, 3])
    mock_y_train = pd.Series(['Positive', 'Positive'], index=[0, 2])
    mock_y_test = pd.Series(['Negative', 'Negative'], index=[1, 3])
    mocker.patch('sklearn.model_selection.train_test_split', return_value=(
        mock_X_train, mock_X_test, mock_y_train, mock_y_test
    ))

    # Return specific mock instances for assertions
    return {
        'RandomForestClassifier': mock_rf_model,
        'ColumnTransformer': mock_ct_instance,
        'OneHotEncoder': mock_ohe_instance,
        'train_test_split': mocker.patch('sklearn.model_selection.train_test_split') # This is the function itself
    }

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame with mixed numerical and categorical features and a target."""
    data = {
        'case_id': [1, 2, 3, 4],
        'patient_age': [30, 45, 60, 25],
        'patient_gender': ['Male', 'Female', 'Male', 'Female'],
        'lab_result_A': [50.5, 62.1, 40.0, 75.2],
        'symptom_severity': ['Mild', 'Severe', 'Moderate', 'Mild'],
        'true_diagnosis': ['Positive', 'Negative', 'Positive', 'Negative']
    }
    return pd.DataFrame(data)

def test_train_ai_model_standard_functionality(sample_dataframe, mock_sklearn):
    """
    Tests the standard functionality with mixed numerical and categorical features.
    Verifies component calls, return types, and arguments for core sklearn functions.
    """
    target_col = 'true_diagnosis'
    random_state = 42

    model, X_train, y_train, X_test, y_test, preprocessor = train_ai_model(sample_dataframe, target_col, random_state)

    # Assert correct types of returned objects
    assert isinstance(model, MagicMock) # It's a mock of RandomForestClassifier
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert isinstance(preprocessor, MagicMock) # It's a mock of ColumnTransformer

    # Assert that preprocessing components were called
    mock_sklearn['ColumnTransformer'].fit_transform.assert_called_once()
    mock_sklearn['OneHotEncoder'].get_feature_names_out.assert_called_once()

    # Assert that train_test_split was called with correct arguments
    mock_sklearn['train_test_split'].assert_called_once()
    call_args, call_kwargs = mock_sklearn['train_test_split'].call_args
    assert 'stratify' in call_kwargs
    assert isinstance(call_kwargs['stratify'], pd.Series) # 'stratify' argument should be a pandas Series

    # Assert model fitting
    mock_sklearn['RandomForestClassifier'].fit.assert_called_once()

def test_train_ai_model_empty_dataframe(mock_sklearn):
    """
    Tests the function with an empty DataFrame input.
    Expects a ValueError from `train_test_split` due to empty data or stratification.
    """
    empty_df = pd.DataFrame(columns=['case_id', 'patient_age', 'true_diagnosis'])
    target_col = 'true_diagnosis'
    random_state = 42

    # Mock train_test_split to raise ValueError for empty input when stratify is used
    mock_sklearn['train_test_split'].side_effect = ValueError("Cannot stratify an empty dataset.")

    with pytest.raises(ValueError, match="Cannot stratify an empty dataset."):
        train_ai_model(empty_df, target_col, random_state)

def test_train_ai_model_target_col_not_found(sample_dataframe, mock_sklearn):
    """
    Tests the function when the target column is not found in the DataFrame.
    Expects a KeyError from `df.drop()`.
    """
    invalid_target_col = 'non_existent_column'
    random_state = 42

    # We don't need to mock df.drop specifically to raise KeyError;
    # pandas' default behavior will do so if the column is missing.
    with pytest.raises(KeyError, match=f"\\['{invalid_target_col}'\\] not found in axis"):
        train_ai_model(sample_dataframe, invalid_target_col, random_state)

def test_train_ai_model_only_numerical_features(mocker, mock_sklearn):
    """
    Tests the function with a DataFrame containing only numerical features.
    Verifies that it processes correctly without issues from `OneHotEncoder`.
    """
    data = {
        'case_id': [1, 2, 3, 4],
        'patient_age': [30, 45, 60, 25],
        'lab_result_A': [50.5, 62.1, 40.0, 75.2],
        'true_diagnosis': ['Positive', 'Negative', 'Positive', 'Negative']
    }
    numerical_df = pd.DataFrame(data)
    target_col = 'true_diagnosis'
    random_state = 42

    # Update mocks for OneHotEncoder and ColumnTransformer to reflect no categorical features
    mock_ohe_instance_no_cat = MagicMock()
    mock_ohe_instance_no_cat.fit_transform.return_value = np.array([]) # Empty array as no categorical features
    mock_ohe_instance_no_cat.get_feature_names_out.return_value = [] # No OHE feature names
    mocker.patch('sklearn.preprocessing.OneHotEncoder', return_value=mock_ohe_instance_no_cat)

    mock_ct_instance_no_cat = MagicMock()
    # Return numerical features directly as there are no categorical to transform
    mock_ct_instance_no_cat.fit_transform.return_value = numerical_df.drop(columns=['case_id', target_col]).values
    mock_ct_instance_no_cat.named_transformers_ = {'cat': mock_ohe_instance_no_cat} # 'cat' still exists but acts on empty
    mocker.patch('sklearn.compose.ColumnTransformer', return_value=mock_ct_instance_no_cat)

    # Run the function
    model, X_train, y_train, X_test, y_test, preprocessor = train_ai_model(numerical_df, target_col, random_state)

    # Assert correct types of returned objects
    assert isinstance(model, MagicMock)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert isinstance(preprocessor, MagicMock)

    # Assert that preprocessing and model fitting occurred
    mock_sklearn['ColumnTransformer'].fit_transform.assert_called_once()
    mock_sklearn['train_test_split'].assert_called_once()
    mock_sklearn['RandomForestClassifier'].fit.assert_called_once()


def test_train_ai_model_single_class_target(sample_dataframe, mock_sklearn):
    """
    Tests the function with a target column that has only one unique class.
    Expects a ValueError from `train_test_split` when `stratify` is used.
    """
    # Create a DataFrame where 'true_diagnosis' is all 'Positive'
    single_class_df = sample_dataframe.copy()
    single_class_df['true_diagnosis'] = 'Positive'
    target_col = 'true_diagnosis'
    random_state = 42

    # Mock train_test_split to raise ValueError for single-class stratification
    mock_sklearn['train_test_split'].side_effect = ValueError(
        "The least populated class in y has only 1 member, which is too few. The minimum number of groups is 2."
    )

    with pytest.raises(ValueError, match="The least populated class in y has only 1 member"):
        train_ai_model(single_class_df, target_col, random_state)