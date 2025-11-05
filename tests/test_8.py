import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch, MagicMock

# definition_c077f1d41a744f749382c5c4b1bf48e8 block
from definition_c077f1d41a744f749382c5c4b1bf48e8 import plot_confidence_vs_override
# /definition_c077f1d41a744f749382c5c4b1bf48e8 block

# Helper function to create a dummy DataFrame for testing
def create_simulation_df(num_rows: int, has_overrides: bool = True, all_overrides: bool = False, seed: int = 42):
    np.random.seed(seed)
    
    # Ensure columns exist even for num_rows=0 to avoid KeyError when accessing df.columns
    base_columns = ['true_diagnosis', 'ai_prediction', 'human_final_decision', 'ai_confidence']
    if num_rows == 0:
        return pd.DataFrame(columns=base_columns)

    ai_confidence = np.random.rand(num_rows)
    true_diagnosis = np.random.choice(['Positive', 'Negative'], num_rows)
    
    if all_overrides:
        # AI always makes wrong prediction, human always corrects
        ai_prediction = np.where(true_diagnosis == 'Positive', 'Negative', 'Positive')
        human_final_decision = true_diagnosis 
    elif has_overrides:
        # Mix of AI predictions, human sometimes corrects, sometimes accepts
        ai_prediction = np.random.choice(['Positive', 'Negative'], num_rows, p=[0.6, 0.4])
        # Human decision: 50% chance to follow true_diagnosis if AI is wrong, else follow AI
        human_final_decision = np.where(np.random.rand(num_rows) < 0.5, ai_prediction, true_diagnosis)
    else: # No overrides: AI is always correct, human always accepts
        ai_prediction = true_diagnosis
        human_final_decision = true_diagnosis

    df = pd.DataFrame({
        'true_diagnosis': true_diagnosis,
        'ai_prediction': ai_prediction,
        'human_final_decision': human_final_decision,
        'ai_confidence': ai_confidence
    })
    return df

# Fixture to mock matplotlib and seaborn functions globally for plotting tests
@pytest.fixture
def mock_plot_functions():
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.show') as mock_show, \
         patch('seaborn.scatterplot') as mock_scatterplot, \
         patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.title') as mock_title, \
         patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
         patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
         patch('matplotlib.pyplot.xticks') as mock_xticks, \
         patch('matplotlib.pyplot.yticks') as mock_yticks, \
         patch('matplotlib.pyplot.grid') as mock_grid, \
         patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
        yield {
            "savefig": mock_savefig, "show": mock_show, "scatterplot": mock_scatterplot,
            "figure": mock_figure, "title": mock_title, "xlabel": mock_xlabel,
            "ylabel": mock_ylabel, "xticks": mock_xticks, "yticks": mock_yticks,
            "grid": mock_grid, "tight_layout": mock_tight_layout
        }

# Test Case 1: Standard data, ensure plotting functions are called as expected
def test_plot_confidence_vs_override_standard_data(mock_plot_functions):
    """
    Tests that the function executes without error and calls the expected plotting
    and saving functions when provided with a typical DataFrame.
    """
    df = create_simulation_df(100, has_overrides=True)
    plot_confidence_vs_override(df)

    mock_plot_functions["figure"].assert_called_once()
    mock_plot_functions["scatterplot"].assert_called_once()
    mock_plot_functions["title"].assert_called_once_with('AI Confidence vs. Human Override Frequency', fontsize=16)
    mock_plot_functions["xlabel"].assert_called_once_with('AI Confidence (Probability of Positive Class)', fontsize=14)
    mock_plot_functions["ylabel"].assert_called_once_with('Frequency of Human Override', fontsize=14)
    mock_plot_functions["xticks"].assert_called_once()
    mock_plot_functions["yticks"].assert_called_once()
    mock_plot_functions["grid"].assert_called_once()
    mock_plot_functions["tight_layout"].assert_called_once()
    mock_plot_functions["savefig"].assert_called_once_with('confidence_vs_override.png', dpi=300)
    mock_plot_functions["show"].assert_called_once()

# Test Case 2: Empty DataFrame
def test_plot_confidence_vs_override_empty_df(mock_plot_functions):
    """
    Tests the function's behavior with an empty DataFrame. It should still attempt
    to create a plot, gracefully handling the lack of data.
    """
    df = create_simulation_df(0) # Empty DataFrame with expected columns
    plot_confidence_vs_override(df)

    # Plotting functions should still be called
    mock_plot_functions["figure"].assert_called_once()
    mock_plot_functions["scatterplot"].assert_called_once()
    mock_plot_functions["savefig"].assert_called_once()
    mock_plot_functions["show"].assert_called_once()
    
    # Verify that scatterplot was called with an empty DataFrame for data
    args, kwargs = mock_plot_functions["scatterplot"].call_args
    plotted_data = kwargs['data']
    assert plotted_data.empty 

# Test Case 3: DataFrame with no human overrides
def test_plot_confidence_vs_override_no_overrides(mock_plot_functions):
    """
    Tests the scenario where humans never override the AI's decision.
    The 'Override_Frequency' in the plotted data should be zero.
    """
    df = create_simulation_df(50, has_overrides=False)
    plot_confidence_vs_override(df)
    mock_plot_functions["scatterplot"].assert_called_once()
    
    args, kwargs = mock_plot_functions["scatterplot"].call_args
    plotted_data = kwargs['data']
    
    assert 'Override_Frequency' in plotted_data.columns
    # With no overrides, the 'override_decision' column in plot_df will be all False.
    # Therefore, after groupby and value_counts(normalize=True), 'Override_Frequency' will be 0.0 for all bins.
    assert (plotted_data['Override_Frequency'] == 0.0).all()

# Test Case 4: DataFrame with all human overrides
def test_plot_confidence_vs_override_all_overrides(mock_plot_functions):
    """
    Tests the scenario where humans always override the AI's decision.
    The 'Override_Frequency' in the plotted data should be one (or very close to it)
    for bins that contain data.
    """
    df = create_simulation_df(50, all_overrides=True)
    plot_confidence_vs_override(df)
    mock_plot_functions["scatterplot"].assert_called_once()
    
    args, kwargs = mock_plot_functions["scatterplot"].call_args
    plotted_data = kwargs['data']
    
    assert 'Override_Frequency' in plotted_data.columns
    # With all overrides, 'override_decision' will be all True.
    # Therefore, 'Override_Frequency' should be 1.0 (for bins that are populated).
    if not plotted_data.empty:
        assert (plotted_data['Override_Frequency'] >= 0.999).all() # Use epsilon for float comparison
    else:
        pytest.fail("Plot data should not be empty for 'all_overrides' scenario with sufficient rows.")

# Test Case 5: Invalid inputs (not a DataFrame or missing critical columns)
@pytest.mark.parametrize("invalid_input, expected_exception, error_match", [
    ("not a dataframe", AttributeError, ""), # Expect AttributeError for .copy() or similar
    (None, AttributeError, ""),               # Expect AttributeError for .copy()
    ([1, 2, 3], AttributeError, ""),          # Expect AttributeError for .copy()
    (pd.DataFrame({'ai_prediction': [], 'human_final_decision': [], 'ai_confidence_MISSING': []}), KeyError, "'ai_confidence'"), # Missing column
    (pd.DataFrame({'ai_prediction_MISSING': [], 'human_final_decision': [], 'ai_confidence': []}), KeyError, "'ai_prediction'"), # Missing column
])
def test_plot_confidence_vs_override_invalid_inputs(invalid_input, expected_exception, error_match):
    """
    Tests error handling for invalid input types (not a DataFrame) and DataFrames
    missing essential columns (`ai_prediction`, `human_final_decision`, `ai_confidence`).
    """
    with pytest.raises(expected_exception, match=error_match):
        plot_confidence_vs_override(invalid_input)