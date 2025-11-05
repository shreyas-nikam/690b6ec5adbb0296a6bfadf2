import pytest
import pandas as pd
import numpy as np
import re # For matching dynamic parts of output

# definition_31715209b0de48f0a721ada6e7f5cf2d
from definition_31715209b0de48f0a721ada6e7f5cf2d import perform_data_validation
# </your_module>

@pytest.fixture
def base_df():
    """Provides a basic valid DataFrame for testing."""
    data = {
        'case_id': [1, 2, 3, 4, 5],
        'patient_age': [30, 45, 60, 25, 70],
        'patient_gender': ['Male', 'Female', 'Other', 'Female', 'Male'],
        'lab_result_A': [55.2, 70.1, 40.5, 62.3, 35.8],
        'lab_result_B': [3.1, 8.9, 1.2, 6.7, 4.0],
        'symptom_severity': ['Mild', 'Severe', 'Moderate', 'Mild', 'Severe'],
        'previous_diagnosis': ['No', 'Yes', 'No', 'No', 'Yes'],
        'true_diagnosis': ['Positive', 'Negative', 'Positive', 'Positive', 'Negative']
    }
    return pd.DataFrame(data)

def test_perform_data_validation_happy_path(base_df, capsys):
    """
    Test case 1: Valid DataFrame with all expected columns, types,
    unique primary key, and no missing critical values.
    """
    critical_cols = ['patient_age', 'lab_result_A', 'true_diagnosis']
    perform_data_validation(base_df, critical_cols)
    captured = capsys.readouterr()

    assert "--- Data Validation Report ---" in captured.out
    assert "SUCCESS: Column 'case_id' present with expected dtype int64." in captured.out
    assert "SUCCESS: 'case_id' column is unique (primary key validated)." in captured.out
    assert "SUCCESS: No missing values in critical fields: patient_age, lab_result_A, true_diagnosis." in captured.out
    assert "ERROR" not in captured.out
    assert "WARNING" not in captured.out
    assert "Summary Statistics for Numeric Columns:" in captured.out

def test_perform_data_validation_missing_critical_column_raises_key_error(base_df, capsys):
    """
    Test case 2: DataFrame is missing an expected column that is also in critical_cols.
    This should print an ERROR and then raise a KeyError when accessing df[critical_cols].
    """
    df_missing_col = base_df.drop(columns=['patient_age'])
    critical_cols = ['patient_age', 'lab_result_A', 'true_diagnosis']
    
    with pytest.raises(KeyError) as excinfo:
        perform_data_validation(df_missing_col, critical_cols)
    
    assert "patient_age" in str(excinfo.value)
    
    captured = capsys.readouterr()
    assert "ERROR: Missing expected column 'patient_age'." in captured.out

def test_perform_data_validation_incorrect_data_type_warning(base_df, capsys):
    """
    Test case 3: DataFrame has a column with an unexpected data type,
    leading to a WARNING message but no error.
    """
    df_incorrect_type = base_df.copy()
    # Change patient_age to float to trigger a warning for expected int64
    df_incorrect_type['patient_age'] = df_incorrect_type['patient_age'].astype(float)
    critical_cols = ['patient_age', 'lab_result_A', 'true_diagnosis']
    perform_data_validation(df_incorrect_type, critical_cols)
    captured = capsys.readouterr()

    assert "WARNING: Column 'patient_age' has unexpected dtype float64, expected <class 'numpy.int64'>." in captured.out
    assert "SUCCESS: 'case_id' column is unique (primary key validated)." in captured.out
    assert "SUCCESS: No missing values in critical fields" in captured.out
    assert "ERROR" not in captured.out

def test_perform_data_validation_non_unique_case_id_error(base_df, capsys):
    """
    Test case 4: DataFrame has non-unique 'case_id' values, leading to an ERROR.
    """
    df_non_unique_id = base_df.copy()
    df_non_unique_id.loc[1, 'case_id'] = 1 # Make case_id 1 appear twice
    critical_cols = ['patient_age', 'lab_result_A', 'true_diagnosis']
    perform_data_validation(df_non_unique_id, critical_cols)
    captured = capsys.readouterr()

    assert "ERROR: 'case_id' column is not unique. Duplicate primary keys found." in captured.out
    assert "SUCCESS: Column 'case_id' present with expected dtype int64." in captured.out
    assert "SUCCESS: No missing values in critical fields" in captured.out
    assert "WARNING" not in captured.out

def test_perform_data_validation_missing_values_in_critical_cols_error_and_warning(base_df, capsys):
    """
    Test case 5: DataFrame has missing values in a critical column.
    This leads to an ERROR message for missing values and a WARNING for dtype change
    (int to float due to NaN).
    """
    df_with_nan = base_df.copy()
    df_with_nan.loc[0, 'patient_age'] = np.nan # Introduce NaN in a critical column
    critical_cols = ['patient_age', 'lab_result_A', 'true_diagnosis']
    perform_data_validation(df_with_nan, critical_cols)
    captured = capsys.readouterr()

    assert "ERROR: Missing values found in critical fields:" in captured.out
    # We use regex to match the summary line, as whitespace might vary.
    assert re.search(r"patient_age\s+1", captured.out) is not None
    assert "WARNING: Column 'patient_age' has unexpected dtype float64, expected <class 'numpy.int64'>." in captured.out
    assert "SUCCESS: 'case_id' column is unique (primary key validated)." in captured.out
    # Ensure other specific errors are not present
    assert "ERROR: 'case_id' column is not unique." not in captured.out
    assert "ERROR: Missing expected column" not in captured.out