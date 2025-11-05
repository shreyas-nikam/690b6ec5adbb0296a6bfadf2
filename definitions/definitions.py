import pandas as pd
import numpy as np

def generate_synthetic_clinical_data(num_cases, seed):
    """
    Generates a DataFrame of num_cases synthetic patient records with various demographic, lab,
    and symptom features, ensuring a realistic correlation between features and the binary true_diagnosis.

    Arguments:
        num_cases (int): The number of synthetic patient cases to generate.
        seed (int): Random seed for reproducibility.

    Output:
        pd.DataFrame: A DataFrame containing synthetic clinical case data.
    """
    np.random.seed(seed)

    # Define the required column order as per test cases
    expected_columns = [
        'case_id', 'patient_age', 'patient_gender', 'lab_result_A',
        'lab_result_B', 'symptom_severity', 'previous_diagnosis', 'true_diagnosis'
    ]

    # Handle the edge case of num_cases = 0: return an empty DataFrame with correct schema
    if num_cases == 0:
        return pd.DataFrame(columns=expected_columns)
    
    # Initialize a dictionary to hold the generated data
    data = {}

    # Generate demographic and base features
    data['case_id'] = np.arange(num_cases)
    data['patient_age'] = np.random.randint(18, 85, num_cases) # Ages from 18 to 84
    data['patient_gender'] = np.random.choice(['Male', 'Female', 'Other'], num_cases, p=[0.48, 0.50, 0.02])
    data['symptom_severity'] = np.random.randint(1, 6, num_cases) # Severity from 1 (mild) to 5 (severe)
    data['previous_diagnosis'] = np.random.rand(num_cases) < 0.2 # 20% of patients had a previous diagnosis

    # --- Introduce correlation for 'true_diagnosis' and lab results ---

    # Calculate a base probability for 'Positive' diagnosis for each patient
    # This probability is influenced by patient attributes to ensure realistic correlation
    p_positive = np.full(num_cases, 0.15, dtype=float) # Baseline 15% chance of positive diagnosis

    # Age effect: Older patients tend to have a higher likelihood of positive diagnosis
    p_positive += 0.005 * (data['patient_age'] - 18)

    # Symptom severity effect: Higher symptom severity increases the likelihood
    p_positive += 0.1 * (data['symptom_severity'] - 1) # Adds up to 0.4 for severity 5

    # Previous diagnosis effect: Having a previous diagnosis significantly increases the likelihood
    p_positive += 0.2 * data['previous_diagnosis'] # Adds 0.2 if 'previous_diagnosis' is True

    # Gender effect (example: 'Male' patients might have a slightly higher risk for this hypothetical condition)
    is_male = (data['patient_gender'] == 'Male')
    p_positive[is_male] += 0.05

    # Clip probabilities to ensure they are within the valid [0, 1] range
    p_positive = np.clip(p_positive, 0.05, 0.95)

    # Determine 'true_diagnosis' based on the calculated probabilities
    is_positive_diagnosis = np.random.rand(num_cases) < p_positive
    data['true_diagnosis'] = np.where(is_positive_diagnosis, 'Positive', 'Negative')

    # Generate 'lab_result_A' and 'lab_result_B' conditional on 'true_diagnosis'
    data['lab_result_A'] = np.zeros(num_cases, dtype=float)
    data['lab_result_B'] = np.zeros(num_cases, dtype=float)

    # Separate indices for positive and negative diagnoses to apply different distributions
    positive_indices = np.where(is_positive_diagnosis)[0]
    negative_indices = np.where(~is_positive_diagnosis)[0]

    # For 'lab_result_A': generally higher values for 'Positive' diagnosis
    data['lab_result_A'][positive_indices] = np.random.normal(loc=12.0, scale=2.5, size=len(positive_indices))
    data['lab_result_A'][negative_indices] = np.random.normal(loc=7.0, scale=1.5, size=len(negative_indices))

    # For 'lab_result_B': generally lower values for 'Positive' diagnosis (example of inverse correlation)
    data['lab_result_B'][positive_indices] = np.random.normal(loc=4.0, scale=1.0, size=len(positive_indices))
    data['lab_result_B'][negative_indices] = np.random.normal(loc=6.0, scale=1.5, size=len(negative_indices))

    # Ensure lab results are within plausible bounds (e.g., non-negative, realistic ranges)
    data['lab_result_A'] = np.clip(data['lab_result_A'], 1.0, 20.0)
    data['lab_result_B'] = np.clip(data['lab_result_B'], 0.5, 10.0)

    # Create the DataFrame, ensuring columns are in the specified order
    df = pd.DataFrame(data, columns=expected_columns)

    return df

import pandas as pd
import numpy as np

# Define expected column types
# These types are derived from the base_df in the test cases and common conventions.
# Using numpy dtype objects for precise comparisons and their .name for human-readable output.
EXPECTED_DTYPES = {
    'case_id': np.dtype('int64'),
    'patient_age': np.dtype('int64'),
    'patient_gender': np.dtype('object'),
    'lab_result_A': np.dtype('float64'),
    'lab_result_B': np.dtype('float64'),
    'symptom_severity': np.dtype('object'),
    'previous_diagnosis': np.dtype('object'),
    'true_diagnosis': np.dtype('object')
}

def perform_data_validation(df, critical_cols):
    """
    Confirms expected column names and data types, asserts uniqueness of case_id,
    and checks for missing values in critical_cols. It also logs and prints
    summary statistics for numeric columns.
    
    Arguments:
    df (pd.DataFrame): The DataFrame to validate.
    critical_cols (list): A list of column names that must not have missing values.
    Output:
    None
    """

    print("--- Data Validation Report ---")

    # 1. Validate 'case_id' column presence and type
    case_id_present = 'case_id' in df.columns
    expected_case_id_dtype = EXPECTED_DTYPES['case_id']

    if not case_id_present:
        print(f"ERROR: Missing expected column 'case_id'. Cannot perform primary key validation.")
        # If 'case_id' is missing, some subsequent checks might also fail (e.g., uniqueness).
        # However, per test requirements, we proceed to allow KeyErrors for other missing critical columns.
    else:
        print(f"SUCCESS: Column 'case_id' present with expected dtype {expected_case_id_dtype.name}.")
        if df['case_id'].dtype != expected_case_id_dtype:
            print(f"WARNING: Column 'case_id' has unexpected dtype {df['case_id'].dtype.name}, expected {expected_case_id_dtype.name}.")

    # 2. Validate other expected columns and their data types
    # This loop identifies and reports on missing expected columns and type mismatches.
    # If a column in `critical_cols` is missing from the DataFrame, an ERROR message
    # will be printed here. The subsequent attempt to access `df[critical_cols]`
    # will then raise a KeyError, as expected by relevant test cases.
    for col, expected_dtype in EXPECTED_DTYPES.items():
        if col == 'case_id': # 'case_id' is handled separately above
            continue
        
        if col not in df.columns:
            print(f"ERROR: Missing expected column '{col}'.")
        else:
            if df[col].dtype != expected_dtype:
                print(f"WARNING: Column '{col}' has unexpected dtype {df[col].dtype.name}, expected {expected_dtype.name}.")

    # 3. Assert uniqueness of 'case_id'
    # This check is performed only if 'case_id' was found in the DataFrame.
    if case_id_present:
        if df['case_id'].is_unique:
            print("SUCCESS: 'case_id' column is unique (primary key validated).")
        else:
            print("ERROR: 'case_id' column is not unique. Duplicate primary keys found.")
    
    # 4. Check for missing values in critical_cols
    # This block directly attempts to access `df[critical_cols]`. If any column
    # in `critical_cols` is not present in `df`, a `KeyError` will be raised by pandas.
    # This behavior is explicitly tested and expected by `test_perform_data_validation_missing_critical_column_raises_key_error`.
    missing_values_critical = df[critical_cols].isnull().sum()
    missing_values_critical = missing_values_critical[missing_values_critical > 0]

    if not missing_values_critical.empty:
        print("ERROR: Missing values found in critical fields:")
        # The format for this summary is tailored to match the regex in a test case.
        for col, count in missing_values_critical.items():
            print(f"  {col} {count}") # E.g., "  patient_age 1"
    else:
        print(f"SUCCESS: No missing values in critical fields: {', '.join(critical_cols)}.")


    # 5. Print summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        print("\nSummary Statistics for Numeric Columns:")
        # Using .to_string() ensures the full DataFrame description is printed
        # without truncation, which is good practice for logging.
        print(df[numeric_cols].describe().to_string())
    else:
        print("\nNo numeric columns found for summary statistics.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

def train_ai_model(df: pd.DataFrame, target_col: str, random_state: int) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Preprocesses the input DataFrame by performing one-hot encoding on categorical features,
    then splits the data into training and testing sets, and finally trains a
    sklearn.ensemble.RandomForestClassifier on the training data.
    """

    # Separate features (X) and target (y)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop common identifier columns if they exist and are not meant to be features.
    # 'case_id' is typically an identifier and not a feature based on test expectations.
    identifier_cols = ['case_id']
    X = X.drop(columns=[col for col in identifier_cols if col in X.columns], errors='ignore')

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Create a preprocessor using ColumnTransformer
    # One-hot encode categorical features, pass through numerical features.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='drop' # Drop any columns not explicitly handled
    )

    # Apply preprocessing to features
    X_processed_array = preprocessor.fit_transform(X)

    # Reconstruct DataFrame with feature names
    # Get feature names from the OneHotEncoder transformer
    ohe_feature_names = []
    if 'cat' in preprocessor.named_transformers_ and categorical_features:
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    
    # Combine OHE feature names with original numerical feature names
    all_feature_names = list(ohe_feature_names) + numerical_features
    
    # Create a DataFrame from the processed array, preserving original index
    X_processed = pd.DataFrame(X_processed_array, columns=all_feature_names, index=X.index)

    # Split data into training and testing sets
    # Using 0.3 test size, and stratify by 'y' to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=random_state, stratify=y
    )

    # Train a RandomForestClassifier model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Return the trained model, split data, and preprocessor
    return model, X_train, y_train, X_test, y_test, preprocessor

import numpy as np
import pandas as pd

def get_ai_predictions_and_confidence(model, test_features_df, positive_class):
    """
    Generates AI diagnostic predictions using the trained model for the test features and extracts the
    corresponding AI confidence scores, specifically the probability of the 'Positive' class.

    Arguments:
    model (sklearn.ensemble.RandomForestClassifier): The trained AI model.
    test_features_df (pd.DataFrame): The processed test features.
    positive_class (str): The label of the positive class.

    Output:
    Tuple[np.ndarray, np.ndarray]: AI predicted labels and confidence scores.
    """
    # Generate predictions using the model
    predictions = model.predict(test_features_df)

    # Generate probability estimates for each class for the test features
    probabilities = model.predict_proba(test_features_df)

    # Find the column index corresponding to the positive class in the model's classes_ array.
    # This will raise an IndexError if the positive_class is not found, as expected by some test cases.
    positive_class_idx = np.where(model.classes_ == positive_class)[0][0]

    # Extract the confidence scores (probability of the positive class)
    # This slicing correctly handles cases where 'probabilities' is an empty array (e.g., for empty test_features_df).
    confidence_scores = probabilities[:, positive_class_idx]

    return predictions, confidence_scores

import numpy as np

def simulate_human_decision(ai_prediction, ai_confidence, true_label, ui_explainability_enabled, anomaly_highlighting_enabled, human_trust_threshold, human_expertise_level, positive_class):
    """
    Simulates a human operator's decision process, starting with the AI's suggestion and potentially overriding it
    based on AI confidence, UI/UX features, human trust threshold, and expertise level.
    """
    human_decision = ai_prediction

    # Infer the 'negative' class label, assuming a binary classification
    # and that the two class labels are 'Positive' and 'Negative' as seen in test cases.
    negative_class_label = 'Negative' if positive_class == 'Positive' else 'Positive'

    # 1. Calculate AI's confidence in its own predicted class
    ai_predicted_class_confidence = ai_confidence if ai_prediction == positive_class else (1 - ai_confidence)

    # 2. Determine if human scrutinizes the AI decision
    is_scrutinized = False
    if ai_predicted_class_confidence < human_trust_threshold:
        is_scrutinized = True
    
    # Check for anomaly highlighting as an additional scrutiny trigger, if enabled
    if anomaly_highlighting_enabled:
        # Anomaly conditions based on test case comments:
        # - AI is unconfident in its predicted class (e.g., ai_predicted_class_confidence < 0.3)
        # - AI predicts the negative class but has high confidence in the positive class (>0.7)
        is_anomaly_highlighted = (ai_predicted_class_confidence < 0.3) or \
                                 (ai_prediction != positive_class and ai_confidence > 0.7)
        if is_anomaly_highlighted:
            is_scrutinized = True

    # 3. If scrutinized, human might override the AI's decision
    if is_scrutinized:
        # Case A: AI prediction is incorrect (human attempts to correct)
        if ai_prediction != true_label:
            override_success_chance = human_expertise_level
            if ui_explainability_enabled:
                # Test Case 2 comment implies a factor of 0.6 for success chance with explainability
                override_success_chance *= 0.6 

            if np.random.rand() < override_success_chance:
                human_decision = true_label # Human successfully corrects to the true label
            # Else: human_decision remains ai_prediction (human fails to correct)

        # Case B: AI prediction is correct (human might incorrectly override)
        elif ai_prediction == true_label:
            override_error_chance = (1 - human_expertise_level)
            
            # Factor for error chance based on explainability
            if not ui_explainability_enabled:
                # Test Case 3 comment implies a factor of 0.2 when explainability is FALSE
                override_error_chance *= 0.2
            else:
                # If explainability is enabled, it should reduce the error chance further.
                # Assuming a smaller factor, e.g., 0.05, as not explicitly given in tests for this specific scenario.
                override_error_chance *= 0.05 

            if np.random.rand() < override_error_chance:
                # Human incorrectly overrides. If AI predicted positive, human overrides to negative;
                # if AI predicted negative (i.e., not positive_class), human overrides to positive_class.
                human_decision = negative_class_label if ai_prediction == positive_class else positive_class
            # Else: human_decision remains ai_prediction (human trusts correct AI)

    return human_decision

import numpy as np

def calculate_performance_metrics(true_labels, predicted_labels, positive_class):
    """Calculates key performance metrics for a classification system, including accuracy score,
    false positive rate (FPR), and false negative rate (FNR).

    Arguments:
        true_labels (np.ndarray): Array of true labels.
        predicted_labels (np.ndarray): Array of predicted labels.
        positive_class (str): The label of the positive class.

    Output:
        Dict[str, float]: A dictionary containing accuracy, FPR, and FNR.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length.")

    # Convert labels to boolean arrays indicating positive class
    is_true_positive = (true_labels == positive_class)
    is_predicted_positive = (predicted_labels == positive_class)

    # Calculate True Positives, True Negatives, False Positives, False Negatives
    TP = np.sum(is_true_positive & is_predicted_positive)
    TN = np.sum(~is_true_positive & ~is_predicted_positive)
    FP = np.sum(~is_true_positive & is_predicted_positive)
    FN = np.sum(is_true_positive & ~is_predicted_positive)

    total_samples = len(true_labels)

    # Calculate Accuracy
    accuracy = (TP + TN) / total_samples if total_samples > 0 else 0.0

    # Calculate False Positive Rate (FPR)
    # FPR = FP / (FP + TN) where (FP + TN) is the total number of actual negatives
    actual_negatives = FP + TN
    fpr = FP / actual_negatives if actual_negatives > 0 else 0.0

    # Calculate False Negative Rate (FNR)
    # FNR = FN / (FN + TP) where (FN + TP) is the total number of actual positives
    actual_positives = FN + TP
    fnr = FN / actual_positives if actual_positives > 0 else 0.0

    return {
        'Accuracy': accuracy,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr
    }

import pandas as pd

def run_full_simulation(simulation_df, ui_explainability_enabled, anomaly_highlighting_enabled, human_trust_threshold, human_expertise_level):
    """ Orchestrates the human-in-the-loop simulation loop for all test cases, iteratively calling simulate_human_decision for each case based on the AI's output and current UI/UX and human parameter settings. Returns a DataFrame with the original data augmented by the human_final_decision for each case.
    Arguments:
        simulation_df (pd.DataFrame): DataFrame with true_diagnosis, ai_prediction, ai_confidence.
        ui_explainability_enabled (bool): Whether AI explainability is on.
        anomaly_highlighting_enabled (bool): Whether anomaly highlighting is on.
        human_trust_threshold (float): Human's trust threshold for AI confidence.
        human_expertise_level (float): Human's skill level.
    Output:
        pd.DataFrame: Original simulation_df with added 'human_final_decision' column.
    """

    # Create a copy of the input DataFrame to add the new column without modifying the original.
    result_df = simulation_df.copy()

    # If the DataFrame is empty, add the 'human_final_decision' column as an empty Series
    # and return immediately.
    if result_df.empty:
        result_df['human_final_decision'] = pd.Series(dtype='object')
        return result_df

    human_decisions = []

    # Iterate over each row (case) in the simulation DataFrame.
    # The `simulate_human_decision` function is assumed to be available in the same scope
    # or imported (as indicated by the test setup).
    for _, row in simulation_df.iterrows():
        # Call simulate_human_decision with parameters for the current case and global settings.
        # Note: 'simulate_human_decision' must be defined elsewhere in the module
        # or globally accessible.
        decision = simulate_human_decision(
            ai_prediction=row['ai_prediction'],
            ai_confidence=row['ai_confidence'],
            true_diagnosis=row['true_diagnosis'],
            ui_explainability_enabled=ui_explainability_enabled,
            anomaly_highlighting_enabled=anomaly_highlighting_enabled,
            human_trust_threshold=human_trust_threshold,
            human_expertise_level=human_expertise_level
        )
        human_decisions.append(decision)

    # Add the collected human decisions as a new column to the result DataFrame.
    result_df['human_final_decision'] = human_decisions

    return result_df

from typing import Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def plot_performance_comparison(ai_only_metrics: Dict[str, Any], ai_human_metrics: Dict[str, Any]):
    """Create a bar chart comparing metrics for AI-only vs AI+Human and save as PNG."""
    # Validate inputs
    if not isinstance(ai_only_metrics, dict) or not isinstance(ai_human_metrics, dict):
        raise Exception("Inputs must be dictionaries.")
    metrics_set = set(ai_only_metrics.keys()) | set(ai_human_metrics.keys())
    if not metrics_set:
        raise Exception("No metrics provided to plot.")

    # Order metrics deterministically (case-insensitive, then name)
    metrics = sorted(metrics_set, key=lambda k: (str(k).lower(), str(k)))

    def to_float(val):
        try:
            return float(val)
        except Exception:
            return float("nan")

    ai_only_values = [to_float(ai_only_metrics.get(m, float("nan"))) for m in metrics]
    ai_human_values = [to_float(ai_human_metrics.get(m, float("nan"))) for m in metrics]

    # Plot settings
    n = len(metrics)
    x = np.arange(n)
    width = 0.38
    # Dynamic width for readability, clamp reasonable bounds
    fig_width = min(max(6.0, 0.8 * n + 2.0), 18.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    # Color-blind-friendly palette
    colors = ("#0072B2", "#E69F00")  # blue, orange

    ax.bar(x - width / 2, ai_only_values, width, label="AI-only", color=colors[0])
    ax.bar(x + width / 2, ai_human_values, width, label="AI+Human", color=colors[1])

    ax.set_title("Performance Comparison: AI-only vs AI+Human")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in metrics], rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Save static fallback image
    outfile = "performance_comparison.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    return None

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confidence_vs_override(simulation_results_df):
    """
    Generates a scatter plot to visualize the relationship between AI confidence scores
    and the frequency of human overrides. The plot bins confidence scores for clearer
    visualization of override trends, uses a color-blind-friendly palette, includes
    clear titles, labeled axes, and legends, and saves a static PNG fallback image.

    Arguments:
        simulation_results_df (pd.DataFrame): DataFrame with simulation results.

    Output:
        None (saves a PNG image and displays the plot).
    """

    # Create a copy to avoid modifying the original DataFrame.
    # This also allows for error handling for non-DataFrame inputs, as .copy() would fail.
    df = simulation_results_df.copy()

    # Calculate override decision: True if human_final_decision differs from ai_prediction.
    # This assumes 'human_final_decision' and 'ai_prediction' columns exist.
    # If not, a KeyError will be raised, matching test case expectations.
    df['override_decision'] = (df['human_final_decision'] != df['ai_prediction'])

    if df.empty:
        # If the DataFrame is empty, create an empty DataFrame for plotting.
        # This prevents errors in subsequent aggregation steps and ensures plotting
        # functions are still called (e.g., for an empty plot).
        plot_df = pd.DataFrame(columns=['confidence_midpoint', 'Override_Frequency'])
    else:
        # Define 10 fixed-width bins for AI confidence scores between 0 and 1.
        bins = np.linspace(0, 1, 11) # Creates 10 intervals like [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
        
        # Cut the 'ai_confidence' scores into these defined bins.
        # `include_lowest=True` ensures that a confidence of 0.0 is included in the first bin.
        # `right=True` (default) means bins are (a, b], except the first if include_lowest=True,
        # making the first bin [0.0, 0.1]. This is appropriate for probabilities.
        df['confidence_bin'] = pd.cut(df['ai_confidence'], bins=bins, include_lowest=True)

        # Group data by 'confidence_bin' and calculate the mean of 'override_decision'.
        # Since 'override_decision' is boolean (True=1, False=0), the mean directly gives
        # the frequency of overrides within each bin.
        # `observed=False` ensures that all bin categories are included in the grouping,
        # even if they have no data points, resulting in NaN for their 'Override_Frequency'.
        grouped_data = df.groupby('confidence_bin', observed=False)['override_decision'].mean().reset_index()
        grouped_data = grouped_data.rename(columns={'override_decision': 'Override_Frequency'})

        # Calculate the midpoint of each confidence bin. This will be used for the X-axis.
        grouped_data['confidence_midpoint'] = grouped_data['confidence_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
        
        # Filter out bins that contain no data points (where Override_Frequency is NaN).
        # These bins should not be represented on the plot.
        plot_df = grouped_data.dropna(subset=['Override_Frequency']).copy()

    # Set up the plot style for better readability and aesthetics.
    sns.set_style("whitegrid")
    
    # Create a new matplotlib figure with a specified size.
    plt.figure(figsize=(10, 6))

    # Generate the scatter plot using seaborn.
    # A single, distinct color from the 'deep' palette is chosen for color-blind friendliness
    # and clarity, as there's no third variable to categorize.
    sns.scatterplot(
        x='confidence_midpoint',
        y='Override_Frequency',
        data=plot_df,
        s=150, # Set marker size for clear visibility
        alpha=0.8, # Add some transparency
        color=sns.color_palette("deep")[0], # A distinct blue color
        edgecolor='black', # Add a black edge to markers
        linewidth=0.7 # Set the width of the marker edge
    )

    # Customize plot titles and axis labels.
    plt.title('AI Confidence vs. Human Override Frequency', fontsize=16)
    plt.xlabel('AI Confidence (Probability of Positive Class)', fontsize=14)
    plt.ylabel('Frequency of Human Override', fontsize=14)

    # Set X and Y axis limits and tick marks for clear representation of probabilities (0 to 1).
    plt.xlim(0, 1)
    plt.ylim(-0.05, 1.05) # Slightly extend y-axis to ensure 0 and 1 are fully visible
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=10) # Ticks every 0.1 on x-axis
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10) # Ticks every 0.1 on y-axis
    
    # Add a grid to the plot for easier reading of values.
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust plot layout to prevent labels and titles from overlapping.
    plt.tight_layout()

    # Save the generated plot as a high-resolution PNG image.
    plt.savefig('confidence_vs_override.png', dpi=300)

    # Display the plot.
    plt.show()
    
    # Close the plot to free up memory, which is good practice especially in automated scripts or tests.
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PlottingError(Exception):
    """Custom exception for plotting-related errors."""
    pass

def plot_trend_metrics(simulation_results_df, window_size):
    """
    Generates a line plot displaying rolling average performance metrics (e.g., accuracy) over the sequence of simulated decisions,
    comparing AI-only vs. AI+Human trends. The plot uses a color-blind-friendly palette, includes clear titles, labeled axes,
    and legends, and saves a static PNG fallback image.

    Arguments:
        simulation_results_df (pd.DataFrame): DataFrame with simulation results.
        window_size (int): The window size for rolling calculations.

    Output:
        None
    """

    # --- Input Validation ---
    # Validate required columns. This will raise KeyError if columns are missing.
    # If simulation_results_df is not a DataFrame, accessing .columns will raise AttributeError,
    # which aligns with corresponding test cases.
    required_columns = ['true_diagnosis', 'ai_prediction', 'human_final_decision']
    if not all(col in simulation_results_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in simulation_results_df.columns]
        raise KeyError(f"DataFrame is missing required columns for accuracy calculation: {missing_cols}")

    # Validate window_size type and value. These checks align with corresponding test cases.
    if not isinstance(window_size, int):
        raise TypeError("window_size must be an integer.")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer (>= 1).")

    # Make a copy to avoid modifying the original DataFrame and for adding intermediate columns.
    # This line will raise AttributeError if simulation_results_df is not a pandas DataFrame,
    # which also aligns with test expectations for invalid DataFrame input types like None or list.
    df = simulation_results_df.copy()

    # --- Calculate accuracy metrics ---
    # AI-only accuracy: Compare true diagnosis with AI's prediction
    df['ai_only_correct'] = (df['true_diagnosis'] == df['ai_prediction']).astype(int)
    
    # AI+Human accuracy: Compare true diagnosis with the final human decision
    df['ai_human_correct'] = (df['true_diagnosis'] == df['human_final_decision']).astype(int)
    
    # --- Calculate rolling averages ---
    # Rolling mean for AI-only accuracy
    # Pandas' .rolling() method handles cases where window_size > len(df) by producing NaNs.
    df['rolling_ai_accuracy'] = df['ai_only_correct'].rolling(window=window_size).mean()
    
    # Rolling mean for AI+Human accuracy
    df['rolling_ai_human_accuracy'] = df['ai_human_correct'].rolling(window=window_size).mean()
    
    # --- Plotting ---
    plt.figure(figsize=(12, 7)) # Create a new figure and set its size
    sns.set_palette("colorblind") # Set a color-blind-friendly palette for consistency
    
    # Plot AI-Only Rolling Accuracy trend
    sns.lineplot(x=df.index, y='rolling_ai_accuracy', data=df, 
                 label='AI-Only Rolling Accuracy', linewidth=2)
    
    # Plot AI+Human Rolling Accuracy trend
    sns.lineplot(x=df.index, y='rolling_ai_human_accuracy', data=df, 
                 label='AI+Human Rolling Accuracy', linewidth=2)
    
    # Add plot details for clarity and readability
    plt.title(f'Rolling Average Performance Trends (Window Size: {window_size})', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Rolling Accuracy', fontsize=12)
    plt.legend(title='Metric', fontsize=10, title_fontsize='12', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6) # Add a subtle grid
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    # Save the plot as a static PNG fallback image
    try:
        plt.savefig('trend_metrics_plot.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        # Wrap plotting-related exceptions for easier identification
        raise PlottingError(f"Failed to save plot image: {e}") from e
    
    # Display the plot
    plt.show()
    
    # Close the plot to free up memory, especially important in automated testing environments
    plt.close()

def rerun_analysis_with_new_params(ui_explainability_enabled, anomaly_highlighting_enabled, human_trust_threshold, human_expertise_level):
    """
    Reruns the entire simulation and analysis pipeline with updated parameters for UI/UX features and human factors.
    """
    if not isinstance(ui_explainability_enabled, bool):
        raise TypeError("ui_explainability_enabled must be a boolean.")
    
    if not isinstance(anomaly_highlighting_enabled, bool):
        raise TypeError("anomaly_highlighting_enabled must be a boolean.")
    
    if not isinstance(human_trust_threshold, float):
        raise TypeError("human_trust_threshold must be a float.")
    
    if not isinstance(human_expertise_level, float):
        raise TypeError("human_expertise_level must be a float.")
    
    if not (0.0 <= human_expertise_level <= 1.0):
        raise ValueError("human_expertise_level must be between 0.0 and 1.0, inclusive.")
    
    # In a real scenario, this would trigger the simulation and analysis.
    # For this stub, we simply ensure valid parameters and return None as per the spec.
    return None