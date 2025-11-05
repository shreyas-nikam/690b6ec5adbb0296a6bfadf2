import pytest
import pandas as pd
import numpy as np
from definition_38843cf0ecf14689a309a30e54a52f83 import generate_synthetic_clinical_data

def test_generate_synthetic_clinical_data_basic_functionality():
    \"\"\"
    Tests that the function generates a DataFrame with the correct number of rows,
    expected columns, and appropriate data types for basic valid inputs.
    \"\"\"
    num_cases = 100
    seed = 42
    df = generate_synthetic_clinical_data(num_cases, seed)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == num_cases
    
    expected_columns = [
        'case_id', 'patient_age', 'patient_gender', 'lab_result_A',
        'lab_result_B', 'symptom_severity', 'previous_diagnosis', 'true_diagnosis'
    ]
    assert list(df.columns) == expected_columns
    
    # Check data types for a representative set of columns
    assert pd.api.types.is_integer_dtype(df['case_id'])
    assert pd.api.types.is_integer_dtype(df['patient_age'])
    assert pd.api.types.is_float_dtype(df['lab_result_A'])
    assert pd.api.types.is_object_dtype(df['patient_gender']) # Categorical typically stored as object/string
    assert pd.api.types.is_object_dtype(df['true_diagnosis']) # Binary string 'Positive'/'Negative'

def test_generate_synthetic_clinical_data_zero_cases_edge_case():
    \"\"\"
    Tests the edge case where num_cases is 0, ensuring an empty DataFrame
    with the correct schema is returned.
    \"\"\"
    num_cases = 0
    seed = 42
    df = generate_synthetic_clinical_data(num_cases, seed)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    
    expected_columns = [
        'case_id', 'patient_age', 'patient_gender', 'lab_result_A',
        'lab_result_B', 'symptom_severity', 'previous_diagnosis', 'true_diagnosis'
    ]
    assert list(df.columns) == expected_columns # Ensures schema is defined even for empty DF

def test_generate_synthetic_clinical_data_reproducibility_with_seed():
    \"\"\"
    Tests that calling the function with the same num_cases and seed
    produces identical DataFrames, ensuring reproducibility.
    \"\"\"
    num_cases = 50
    seed = 123
    df1 = generate_synthetic_clinical_data(num_cases, seed)
    df2 = generate_synthetic_clinical_data(num_cases, seed)

    pd.testing.assert_frame_equal(df1, df2)

def test_generate_synthetic_clinical_data_negative_num_cases_raises_value_error():
    \"\"\"
    Tests that passing a negative value for num_cases raises a ValueError,
    as `np.random` functions like `randint` require non-negative size.
    \"\"\"
    with pytest.raises(ValueError, match="size must be non-negative"):
        generate_synthetic_clinical_data(-1, 42)

@pytest.mark.parametrize("invalid_num_cases", [
    "not_an_int",
    10.5, # Float is not an integer
    [1, 2, 3] # List is not an integer
])
def test_generate_synthetic_clinical_data_invalid_num_cases_type_raises_error(invalid_num_cases):
    \"\"\"
    Tests that passing non-integer types for num_cases raises either a TypeError
    or ValueError, depending on how numpy handles the specific invalid type.
    \"\"\"
    with pytest.raises((TypeError, ValueError)):
        generate_synthetic_clinical_data(invalid_num_cases, 42)
