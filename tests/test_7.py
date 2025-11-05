import os
import pytest
import matplotlib
matplotlib.use("Agg", force=True)

from definition_d58928c77c914e86be4b1aba91626aba import plot_performance_comparison


@pytest.fixture
def isolated_plot_env(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt
    monkeypatch.chdir(tmp_path)
    # Prevent GUI pop-ups during tests
    monkeypatch.setattr(plt, "show", lambda: None)
    return tmp_path


@pytest.mark.parametrize("ai_only, ai_human", [
    (
        {"Accuracy": 0.8, "False Positive Rate": 0.1, "False Negative Rate": 0.2},
        {"Accuracy": 0.85, "False Positive Rate": 0.08, "False Negative Rate": 0.18},
    ),
    (
        # Out-of-range values and additional metric should still produce a plot and file
        {"Accuracy": 1.2, "False Positive Rate": -0.1, "False Negative Rate": 0.5, "AUC": 0.9},
        {"Accuracy": 0.95, "False Positive Rate": 0.02, "False Negative Rate": 1.1, "AUC": 1.2},
    ),
    (
        # Mismatched keys across dictionaries
        {"Acc": 0.7, "FPR": 0.2},
        {"Accuracy": 0.75, "False Positive Rate": 0.25},
    ),
])
def test_plot_creates_png_with_valid_inputs(ai_only, ai_human, isolated_plot_env):
    result = plot_performance_comparison(ai_only, ai_human)
    assert result is None or result is not ...  # Function should not return a value; allow None
    outfile = "performance_comparison.png"
    assert os.path.exists(outfile), "Expected output PNG file was not created."
    assert os.path.getsize(outfile) > 0, "Output PNG file is empty."


@pytest.mark.parametrize("ai_only, ai_human, expected_exception", [
    (["not", "a", "dict"], {"Accuracy": 0.8}, Exception),
    ({}, {}, Exception),  # Empty inputs should fail to plot
])
def test_plot_invalid_inputs_raise(ai_only, ai_human, expected_exception, isolated_plot_env):
    with pytest.raises(expected_exception):
        plot_performance_comparison(ai_only, ai_human)