import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Keep the definition_e60b3c041480486c8adef36c4e57ef16 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_e60b3c041480486c8adef36c4e57ef16 import plot_trend_metrics


# Helper function to create a dummy DataFrame for testing
def create_dummy_df(num_rows, seed=42):
    np.random.seed(seed)
    data = {
        'true_diagnosis': np.random.choice(['Positive', 'Negative'], num_rows),
        'ai_prediction': np.random.choice(['Positive', 'Negative'], num_rows),
        'human_final_decision': np.random.choice(['Positive', 'Negative'], num_rows),
    }
    return pd.DataFrame(data)

@pytest.fixture(autouse=True)
def mock_plot_functions(mocker):
    """Fixture to mock matplotlib and seaborn plot display/save functions."""
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.savefig')
    mocker.patch('seaborn.lineplot', autospec=True)
    mocker.patch('matplotlib.pyplot.figure', autospec=True)

def test_plot_trend_metrics_happy_path():
    """
    Test with valid simulation results and a reasonable window size.
    Verifies that plotting functions are called as expected.
    """
    df = create_dummy_df(num_rows=100)
    window_size = 10
    
    plot_trend_metrics(df, window_size)
    
    # Assert that matplotlib.pyplot.show and .savefig were called
    plt.show.assert_called_once()
    plt.savefig.assert_called_once()
    # Assert that seaborn.lineplot was called twice (for AI-Only and AI+Human)
    assert sns.lineplot.call_count == 2
    
    # Optional: check some arguments to lineplot
    args1, kwargs1 = sns.lineplot.call_args_list[0]
    assert kwargs1['label'] == 'AI-Only Rolling Accuracy'
    args2, kwargs2 = sns.lineplot.call_args_list[1]
    assert kwargs2['label'] == 'AI+Human Rolling Accuracy'
    
    # Verify that the internal rolling calculation resulted in non-empty Series
    assert not kwargs1['data']['rolling_ai_accuracy'].empty
    assert not kwargs2['data']['rolling_ai_human_accuracy'].empty

@pytest.mark.parametrize("num_rows, window_size, expected_rolling_full_nan", [
    (0, 10, True),  # Empty DataFrame
    (5, 10, True),  # DataFrame with fewer rows than window_size
    (9, 10, False) # DataFrame with almost window_size rows, first few NaNs, rest values
])
def test_plot_trend_metrics_empty_or_insufficient_data(num_rows, window_size, expected_rolling_full_nan):
    """
    Test with an empty DataFrame or a DataFrame with insufficient data for the window size.
    Should handle gracefully without errors and call plotting functions.
    """
    df = create_dummy_df(num_rows=num_rows)
    
    plot_trend_metrics(df, window_size)
    
    plt.show.assert_called_once()
    plt.savefig.assert_called_once()
    assert sns.lineplot.call_count == 2

    args1, kwargs1 = sns.lineplot.call_args_list[0]
    rolling_ai_accuracy = kwargs1['data']['rolling_ai_accuracy']
    
    if num_rows == 0:
        assert rolling_ai_accuracy.empty
    elif expected_rolling_full_nan:
        assert rolling_ai_accuracy.isnull().all()
    else:
        # For num_rows=9, window_size=10, the first (window_size-1) values will be NaN
        # Only the last value will be calculated based on the window.
        assert rolling_ai_accuracy.iloc[:window_size-1].isnull().all()
        assert not rolling_ai_accuracy.iloc[window_size-1:].isnull().all()

def test_plot_trend_metrics_window_size_one():
    """
    Test with window_size = 1. Rolling mean should be identical to original data.
    """
    df = create_dummy_df(num_rows=50)
    window_size = 1
    
    plot_trend_metrics(df, window_size)
    
    plt.show.assert_called_once()
    plt.savefig.assert_called_once()
    assert sns.lineplot.call_count == 2

    # Check if rolling accuracy for window_size=1 matches the raw correct flags
    ai_only_correct_series = (df['true_diagnosis'] == df['ai_prediction']).astype(int)
    ai_human_correct_series = (df['true_diagnosis'] == df['human_final_decision']).astype(int)

    args1, kwargs1 = sns.lineplot.call_args_list[0]
    pd.testing.assert_series_equal(kwargs1['data']['rolling_ai_accuracy'], ai_only_correct_series.rolling(window=window_size).mean(), check_names=False)
    args2, kwargs2 = sns.lineplot.call_args_list[1]
    pd.testing.assert_series_equal(kwargs2['data']['rolling_ai_human_accuracy'], ai_human_correct_series.rolling(window=window_size).mean(), check_names=False)

@pytest.mark.parametrize("invalid_df_input, expected_error", [
    (None, TypeError),
    ("not a dataframe", AttributeError),
    ([1, 2, 3], AttributeError), # list does not have .copy() or column operations
    (pd.DataFrame({'col': [1,2]}), KeyError) # Missing required columns
])
def test_plot_trend_metrics_invalid_dataframe_input(invalid_df_input, expected_error):
    """
    Test with non-DataFrame input or DataFrame missing required columns for simulation_results_df.
    Expects a TypeError, AttributeError, or KeyError.
    """
    window_size = 10
    
    with pytest.raises(expected_error):
        plot_trend_metrics(invalid_df_input, window_size)

@pytest.mark.parametrize("invalid_window_size, expected_error, error_match", [
    (0, ValueError, "window must be >= 1"),
    (-5, ValueError, "window must be >= 1"),
    ("ten", TypeError, None), # Type error from pandas for non-integer window
    (10.5, TypeError, None) # Type error from pandas for non-integer window
])
def test_plot_trend_metrics_invalid_window_size(invalid_window_size, expected_error, error_match):
    """
    Test with invalid window_size (e.g., 0, negative, non-integer).
    Expects a ValueError or TypeError from pandas rolling method.
    """
    df = create_dummy_df(num_rows=100)
    
    if error_match:
        with pytest.raises(expected_error, match=error_match):
            plot_trend_metrics(df, invalid_window_size)
    else:
        with pytest.raises(expected_error):
            plot_trend_metrics(df, invalid_window_size)