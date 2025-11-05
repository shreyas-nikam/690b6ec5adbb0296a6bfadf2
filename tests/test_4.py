import pytest
import numpy as np
from unittest.mock import patch

# Keep the definition_62016dc993c84c71a263472206f05a2c block as it is. DO NOT REPLACE or REMOVE the block.
from definition_62016dc993c84c71a263472206f05a2c import simulate_human_decision

@pytest.mark.parametrize(
    "ai_prediction, ai_confidence, true_label, ui_explainability_enabled, anomaly_highlighting_enabled, "
    "human_trust_threshold, human_expertise_level, positive_class, mock_rand_value, expected_decision",
    [
        # Test Case 1: AI is correct and confident, no scrutiny features -> Human accepts AI.
        # ai_predicted_class_confidence (0.95) >= human_trust_threshold (0.7). No anomaly. No scrutiny.
        ('Positive', 0.95, 'Positive', False, False, 0.7, 0.8, 'Positive', None, 'Positive'),
        
        # Test Case 2: AI is incorrect and unconfident, UI/UX enabled. Human overrides correctly.
        # AI predicts 'Positive' with low confidence (0.2), true is 'Negative'.
        # ai_predicted_class_confidence (0.2) < human_trust_threshold (0.7). Anomaly (0.2 < 0.3). Scrutiny.
        # AI is wrong. ui_explainability_enabled (True) gives override_success_chance = 0.8 * 0.6 = 0.48.
        # Mock rand() < 0.48 to simulate a correct override.
        ('Positive', 0.2, 'Negative', True, True, 0.7, 0.8, 'Positive', 0.2, 'Negative'),
        
        # Test Case 3: AI is correct but slightly unconfident. Human overrides incorrectly due to low expertise.
        # AI predicts 'Positive' with confidence (0.6), true is 'Positive'.
        # ai_predicted_class_confidence (0.6) < human_trust_threshold (0.7). Scrutiny.
        # AI is correct. ui_explainability_enabled (False) gives override_error_chance = (1 - 0.2) * 0.2 = 0.16.
        # Mock rand() < 0.16 to simulate an incorrect override.
        ('Positive', 0.6, 'Positive', False, False, 0.7, 0.2, 'Positive', 0.1, 'Negative'),
        
        # Test Case 4: AI is incorrect, predicts 'Negative' but confident in 'Positive' (0.9), anomaly highlights. Human fails to correct.
        # AI predicts 'Negative', but ai_confidence (of 'Positive') is 0.9. True is 'Positive'.
        # ai_predicted_class_confidence (for 'Negative') = 1 - 0.9 = 0.1 < human_trust_threshold (0.7). Scrutiny.
        # Anomaly also triggered: (ai_prediction != positive_class and ai_confidence > 0.7) is True.
        # AI is wrong. ui_explainability_enabled (True) gives override_success_chance = 0.8 * 0.6 = 0.48.
        # Mock rand() >= 0.48 to simulate human *failing* to correct.
        ('Negative', 0.9, 'Positive', True, True, 0.7, 0.8, 'Positive', 0.5, 'Negative'),

        # Test Case 5: AI is correct and confident about 'Negative' class, no scrutiny features -> Human accepts AI.
        # AI predicts 'Negative', ai_confidence (of 'Positive') is 0.1 (meaning confidence in 'Negative' is 0.9). True is 'Negative'.
        # ai_predicted_class_confidence (for 'Negative') = 0.9 >= human_trust_threshold (0.7). No anomaly. No scrutiny.
        ('Negative', 0.1, 'Negative', False, False, 0.7, 0.8, 'Positive', None, 'Negative'),
    ]
)
def test_simulate_human_decision(
    ai_prediction, ai_confidence, true_label, ui_explainability_enabled, anomaly_highlighting_enabled,
    human_trust_threshold, human_expertise_level, positive_class, mock_rand_value, expected_decision
):
    """
    Test various scenarios for human decision simulation, including cases with and without UI/UX features,
    different AI confidence levels, and human expertise, mocking random outcomes for determinism.
    """
    if mock_rand_value is not None:
        with patch('numpy.random.rand', return_value=mock_rand_value):
            result = simulate_human_decision(
                ai_prediction, ai_confidence, true_label, ui_explainability_enabled,
                anomaly_highlighting_enabled, human_trust_threshold, human_expertise_level,
                positive_class
            )
            assert result == expected_decision
    else:
        # For cases where np.random.rand() is not expected to be called (no scrutiny)
        # or its value wouldn't change the outcome (e.g., if override_success_chance is 0)
        result = simulate_human_decision(
            ai_prediction, ai_confidence, true_label, ui_explainability_enabled,
            anomaly_highlighting_enabled, human_trust_threshold, human_expertise_level,
            positive_class
        )
        assert result == expected_decision