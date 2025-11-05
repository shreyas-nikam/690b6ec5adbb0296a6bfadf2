
# Streamlit Application Requirements Specification: Human-in-the-Loop Override Simulator

## 1. Application Overview

This Streamlit application will provide an interactive simulation of 'Human Oversight Roles' within an AI-assisted decision-making system. Users will step into the role of human operators, reviewing AI-generated diagnostic suggestions in a synthetic clinical context. They will make override decisions, observe the impact of their interventions, and analyze how UI/UX features and human factors influence overall system safety and performance.

### Learning Goals

The application is designed to help users:
-   Understand the key insights contained in the uploaded document and supporting data.
-   Explain assurance-case methodology, evaluate uncertainty and model risk in real-world conditions, and translate "fit-for-purpose" into measurable evidence.
-   Explore the importance of designing effective UI/UX to help users understand model outputs and catch anomalies.
-   Understand 'Human Oversight Roles' and how human intervention impacts system safety and performance in critical applications.

## 2. User Interface Requirements

The application will follow a linear flow, guiding the user through the simulation setup, execution, and analysis.

### Layout and Navigation Structure

The application will be structured into distinct sections, presented sequentially on a single page, making use of Streamlit's `st.header`, `st.subheader`, and `st.markdown` for content separation.

1.  **Introduction & Overview:**
    *   Application title and purpose.
    *   Learning outcomes and scope/constraints.
    *   Methodology overview.
2.  **Data Overview:**
    *   Details about the synthetic clinical data.
    *   Initial data display and validation report.
3.  **AI Model Setup:**
    *   Confirmation of AI model training.
4.  **Simulation Controls:**
    *   Interactive widgets for human factors and UI/UX toggles.
5.  **Simulation Results & Interactive Analysis:**
    *   Display of updated performance metrics.
    *   Interactive plots showing comparative performance, override frequency, and performance trends.
6.  **Conclusion & Next Steps:**
    *   Summary of key takeaways.
    *   Future considerations.
7.  **References:**
    *   Citations for external resources.

### Input Widgets and Controls

The application will feature interactive controls, primarily located in a dedicated "Simulation Controls" section, allowing users to modify simulation parameters dynamically.

| Widget Type      | Description                                          | Streamlit Equivalent | Value/Range      | Default |
| :--------------- | :--------------------------------------------------- | :------------------- | :--------------- | :------ |
| Checkbox         | Toggle AI Explainability                             | `st.checkbox`        | Boolean          | True    |
| Checkbox         | Toggle Anomaly Highlighting                          | `st.checkbox`        | Boolean          | True    |
| Slider (Float)   | Human Trust Threshold (for AI confidence)            | `st.slider`          | 0.0 to 1.0 (step 0.05) | 0.7     |
| Slider (Float)   | Human Expertise Level (for correcting AI errors)     | `st.slider`          | 0.0 to 1.0 (step 0.05) | 0.6     |

### Visualization Components (Charts, Graphs, Tables)

All visualizations will be rendered using `st.pyplot` for Matplotlib/Seaborn plots, ensuring clear titles, labeled axes, and legends with a color-blind-friendly palette.

1.  **Data Table:**
    *   Display `clinical_data.head()` (first 5 rows of generated synthetic data).
    *   Display `simulation_base_df.head()` (AI predictions and confidence).
    *   Display `initial_simulation_results_df.head()` (AI vs. Human decision).
2.  **Comparative System Performance (Bar Chart):**
    *   **Type:** Bar chart.
    *   **Purpose:** Compare Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR) between "AI-Only" and "AI+Human" systems.
    *   **Input:** `ai_only_metrics`, `ai_human_metrics` dictionaries.
    *   **Style:** `viridis` palette, font size $\ge 12$ pt, clear titles, labeled axes, and legends.
3.  **AI Confidence vs. Human Override Frequency (Scatter Plot):**
    *   **Type:** Scatter plot.
    *   **Purpose:** Visualize the relationship between AI confidence scores (binned) and the frequency of human overrides.
    *   **Input:** `simulation_results_df`.
    *   **Style:** `deep` color palette, font size $\ge 12$ pt, clear titles, labeled axes, and legends.
4.  **Rolling Average Performance Trends (Line Plot):**
    *   **Type:** Line plot.
    *   **Purpose:** Display rolling average accuracy for "AI-Only" and "AI+Human" systems over simulation steps.
    *   **Input:** `simulation_results_df`, `window_size` (default 50).
    *   **Style:** `colorblind` palette, font size $\ge 12$ pt, clear titles, labeled axes, and legends.

### Interactive Elements and Feedback Mechanisms

-   Changes to any input widget (checkboxes, sliders) will automatically trigger a rerun of the simulation, metric calculation, and plot updates.
-   Text feedback will be provided using `st.write` or `st.info` to indicate when the simulation is rerunning and when analysis is complete.
-   Updated performance metrics will be displayed directly below the controls.

## 3. Additional Requirements

### Annotation and Tooltip Specifications

All input widgets will include help text or tooltips, using Streamlit's `help` parameter, providing context for each control.
-   **Show AI Explainability:** "When enabled, human operators are better informed and more likely to make correct overrides."
-   **Highlight Anomalies:** "When enabled, potential inconsistencies or low-confidence predictions are flagged, prompting human scrutiny."
-   **Human Trust Threshold:** "AI confidence score below this threshold makes human more likely to scrutinize/override."
-   **Human Expertise Level:** "Higher level means human is better at correcting AI errors when they decide to override."

### Save the states of the fields properly so that changes are not lost

Streamlit's `st.session_state` will be utilized to maintain the state of all input widgets across reruns.
-   The trained AI model and preprocessor will be cached using `st.cache_resource` to prevent retraining on every rerun.
-   The initial synthetic data will be generated once and stored in `st.session_state` or cached with `st.cache_data`.

## 4. Notebook Content and Code Requirements

All relevant Markdown content from the Jupyter Notebook will be rendered using `st.markdown`, and Python functions will be integrated into the Streamlit script.

### Application Title and Introduction
```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress Matplotlib global use warning

st.title("Human-in-the-Loop Override Simulator: Clinical Decision Support")

st.markdown("""
This application provides a practical simulation of 'Human Oversight Roles' within an AI-assisted decision-making system. Focusing on a synthetic clinical decision support scenario, users will act as human operators reviewing AI-generated diagnostic suggestions, making override decisions, and observing the resultant impact of human intervention on overall system safety and performance. This lab directly applies concepts from Unit 3: Assurance Foundations for Critical ML, particularly 'Human Oversight Roles' and the importance of effective UI/UX design.
""")

st.subheader("Learning Outcomes")
st.markdown("""
-   Understand the key insights contained in the uploaded document and supporting data.
-   Explain assurance-case methodology, evaluate uncertainty and model risk in real-world conditions, and translate "fit-for-purpose" into measurable evidence.
-   Explore the importance of designing effective UI/UX to help users understand model outputs and catch anomalies.
-   Understand 'Human Oversight Roles' and how human intervention impacts system safety and performance in critical applications.
""")

st.subheader("Scope & Constraints")
st.markdown("""
-   The lab must execute end-to-end on a mid-spec laptop (8 GB RAM) in fewer than 5 minutes.
-   Only open-source Python libraries from PyPI may be used.
-   All major steps include both code comments and brief narrative cells describing **what** is happening and **why**.
""")

st.header("2. Environment Setup: Library Imports and Global Settings")
st.markdown("All required libraries have been successfully imported and warnings are suppressed.")

st.header("3. Data/Inputs Overview: Synthetic Clinical Data")
st.markdown("""
To effectively simulate human oversight in AI-assisted clinical decision-making, we will generate a synthetic dataset. This approach allows us to control various parameters, ensuring the dataset reflects realistic patient characteristics and diagnostic complexities without using sensitive real-world patient data. The synthetic data will include patient demographics, lab results, symptom severity, and a 'true diagnosis,' which our AI model will attempt to predict.

**Assumptions:**
-   The synthetic data is representative of a typical clinical scenario for a binary classification problem (e.g., presence or absence of a specific condition).
-   Features are designed to have plausible correlations with the `true_diagnosis` to make the AI's task meaningful.

This synthetic dataset serves as the foundation for exploring the intricacies of human-in-the-loop systems, providing a safe and controlled environment to test different UI/UX strategies and human intervention patterns.
""")
```

### Data Generation
```python
@st.cache_data
def generate_synthetic_clinical_data(num_cases: int, seed: int) -> pd.DataFrame:
    """
    Generates a DataFrame of num_cases synthetic patient records with various demographic, lab,
    and symptom features, ensuring a realistic correlation between features and the binary true_diagnosis.
    """
    np.random.seed(seed)
    expected_columns = [
        'case_id', 'patient_age', 'patient_gender', 'lab_result_A',
        'lab_result_B', 'symptom_severity', 'previous_diagnosis', 'true_diagnosis'
    ]
    if num_cases == 0:
        return pd.DataFrame(columns=expected_columns)
    
    data = {}
    data['case_id'] = np.arange(num_cases)
    data['patient_age'] = np.random.randint(18, 85, num_cases)
    data['patient_gender'] = np.random.choice(['Male', 'Female', 'Other'], num_cases, p=[0.48, 0.50, 0.02])
    data['symptom_severity'] = np.random.randint(1, 6, num_cases)
    data['previous_diagnosis'] = np.random.rand(num_cases) < 0.2

    p_positive = np.full(num_cases, 0.15, dtype=float)
    p_positive += 0.005 * (data['patient_age'] - 18)
    p_positive += 0.1 * (data['symptom_severity'] - 1)
    p_positive += 0.2 * data['previous_diagnosis']
    is_male = (data['patient_gender'] == 'Male')
    p_positive[is_male] += 0.05
    p_positive = np.clip(p_positive, 0.05, 0.95)

    is_positive_diagnosis = np.random.rand(num_cases) < p_positive
    data['true_diagnosis'] = np.where(is_positive_diagnosis, 'Positive', 'Negative')

    data['lab_result_A'] = np.zeros(num_cases, dtype=float)
    data['lab_result_B'] = np.zeros(num_cases, dtype=float)

    positive_indices = np.where(is_positive_diagnosis)[0]
    negative_indices = np.where(~is_positive_diagnosis)[0]

    data['lab_result_A'][positive_indices] = np.random.normal(loc=12.0, scale=2.5, size=len(positive_indices))
    data['lab_result_A'][negative_indices] = np.random.normal(loc=7.0, scale=1.5, size=len(negative_indices))

    data['lab_result_B'][positive_indices] = np.random.normal(loc=4.0, scale=1.0, size=len(positive_indices))
    data['lab_result_B'][negative_indices] = np.random.normal(loc=6.0, scale=1.5, size=len(negative_indices))

    data['lab_result_A'] = np.clip(data['lab_result_A'], 1.0, 20.0)
    data['lab_result_B'] = np.clip(data['lab_result_B'], 0.5, 10.0)

    df = pd.DataFrame(data, columns=expected_columns)
    return df

# Generate data initially or load from session state
if 'clinical_data' not in st.session_state:
    st.session_state.clinical_data = generate_synthetic_clinical_data(num_cases=500, seed=42)

st.write(f"Generated a dataset of {len(st.session_state.clinical_data)} synthetic clinical cases.")
st.write("First 5 rows of the dataset:")
st.dataframe(st.session_state.clinical_data.head())

st.write("Dataset Information:")
# Display info in a more Streamlit-friendly way, e.g., a table
buffer = pd.io.common.StringIO()
st.session_state.clinical_data.info(buf=buffer)
s = buffer.getvalue()
st.code(s)

st.markdown("""
**Interpretation:**
A synthetic dataset of 500 clinical cases has been generated. We can observe the mix of numeric (e.g., `patient_age`, `lab_result_A`, `lab_result_B`) and categorical features (e.g., `patient_gender`, `symptom_severity`), along with the `true_diagnosis` target variable. This dataset is designed to be lightweight (less than 5 MB) for efficient execution and provides a realistic basis for our simulation. The `info()` output confirms the data types and absence of initial nulls, preparing us for further validation.
""")
```

### Data Validation
```python
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

def perform_data_validation(df: pd.DataFrame, critical_cols: list) -> None:
    """
    Confirms expected column names and data types, asserts uniqueness of case_id,
    and checks for missing values in critical_cols. It also logs and prints
    summary statistics for numeric columns.
    """
    st.subheader("Data Validation Report")

    if df.empty:
        st.warning("Input DataFrame is empty. No validation performed.")
        return

    case_id_present = 'case_id' in df.columns
    expected_case_id_dtype = EXPECTED_DTYPES['case_id']

    if not case_id_present:
        st.error(f"ERROR: Missing expected column 'case_id'. Cannot perform primary key validation.")
    else:
        st.success(f"SUCCESS: Column 'case_id' present with expected dtype {expected_case_id_dtype.name}.")
        if df['case_id'].dtype != expected_case_id_dtype:
            st.warning(f"WARNING: Column 'case_id' has unexpected dtype {df['case_id'].dtype.name}, expected {expected_case_id_dtype.name}.")

    for col, expected_dtype in EXPECTED_DTYPES.items():
        if col == 'case_id':
            continue
        if col not in df.columns:
            st.error(f"ERROR: Missing expected column '{col}'.")
        else:
            if df[col].dtype != expected_dtype:
                st.warning(f"WARNING: Column '{col}' has unexpected dtype {df[col].dtype.name}, expected {expected_dtype.name}.")

    if case_id_present:
        if df['case_id'].is_unique:
            st.success("'case_id' column is unique (primary key validated).")
        else:
            st.error("'case_id' column is not unique. Duplicate primary keys found.")
    
    missing_values_critical = df[critical_cols].isnull().sum()
    missing_values_critical = missing_values_critical[missing_values_critical > 0]

    if not missing_values_critical.empty:
        st.error("ERROR: Missing values found in critical fields:")
        for col, count in missing_values_critical.items():
            st.write(f"  - {col}: {count}")
    else:
        st.success(f"SUCCESS: No missing values in critical fields: {', '.join(critical_cols)}.")

    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        st.markdown("\nSummary Statistics for Numeric Columns:")
        st.dataframe(df[numeric_cols].describe())
    else:
        st.markdown("\nNo numeric columns found for summary statistics.")

st.header("5. Sectioned Implementation")
st.subheader("5.1 Utility Function: Generating Synthetic Clinical Data")
st.markdown("""
**Context & Business Value:**
To provide a realistic yet controlled environment for our simulation, we first need to generate a synthetic dataset of patient records. This function, `generate_synthetic_clinical_data`, is designed to create diverse patient profiles with various clinical features and a `true_diagnosis`. By using synthetic data, we can freely experiment with different scenarios without privacy concerns, directly supporting the exploration of 'Human Oversight Roles' and UI/UX design.

This function is crucial as it sets the stage for training the AI model and simulating human interactions, providing the foundational input for the entire system.
""")

st.subheader("5.2 Utility Function: Data Validation and Initial Exploration")
st.markdown("""
**Context & Business Value:**
Data validation is a critical step in any data-driven project, especially in clinical applications where data quality directly impacts patient outcomes. The `perform_data_validation` function ensures that our generated synthetic data conforms to expected standards. It checks for correct column names and data types, verifies the uniqueness of the `case_id` (acting as a primary key), and asserts the absence of missing values in critical fields. This rigorous validation minimizes the risk of downstream errors and builds confidence in the reliability of our simulation results.

By confirming data integrity upfront, we ensure that our AI model training and human override simulations are based on a sound and consistent dataset, which is fundamental for generating trustworthy insights into assurance foundations.
""")

critical_fields = ['patient_age', 'lab_result_A', 'true_diagnosis']
perform_data_validation(st.session_state.clinical_data, critical_fields)

st.markdown("""
**Interpretation:**
The data validation report confirms that our synthetic dataset meets the expected structural and quality standards. It verifies column presence, data types, and the uniqueness of `case_id`, which is essential for data integrity. The absence of missing values in critical fields (`patient_age`, `lab_result_A`, `true_diagnosis`) ensures that our core features for AI training are complete. Summary statistics for numeric columns provide an initial look at the data distribution, confirming realistic ranges for age and lab results. This robust validation ensures a reliable foundation for the subsequent AI model training and simulation, mitigating risks associated with poor data quality.
""")
```

### AI Model Training
```python
@st.cache_resource
def train_ai_model(df: pd.DataFrame, target_col: str, random_state: int) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Preprocesses the input DataFrame by performing one-hot encoding on categorical features,
    then splits the data into training and testing sets, and finally trains a
    sklearn.ensemble.RandomForestClassifier on the training data.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])
    identifier_cols = ['case_id']
    X = X.drop(columns=[col for col in identifier_cols if col in X.columns], errors='ignore')

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='drop'
    )

    X_processed_array = preprocessor.fit_transform(X)
    ohe_feature_names = []
    if 'cat' in preprocessor.named_transformers_ and categorical_features:
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    
    all_feature_names = list(ohe_feature_names) + numerical_features
    X_processed = pd.DataFrame(X_processed_array, columns=all_feature_names, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)

    st.success("RandomForestClassifier trained successfully.")
    return model, X_train, y_train, X_test, y_test, preprocessor

st.subheader("5.3 Utility Function: AI Model Training (Simulated)")
st.markdown("""
**Context & Business Value:**
Our simulated AI assistant is a core component of the human-in-the-loop system. The `train_ai_model` function establishes this AI, represented by a `RandomForestClassifier`, which will predict `true_diagnosis` based on patient features. This model serves as the "AI suggestion" in our simulation, providing a baseline for human operators to review and potentially override.

Before training, the function preprocesses categorical features using one-hot encoding, transforming them into a numerical format suitable for the `RandomForestClassifier`. The data is then split into training and testing sets to evaluate the AI's performance on unseen data. The trained AI model is critical for demonstrating how human oversight can enhance or correct AI outputs, thereby directly addressing the learning goals related to human oversight roles and assurance foundations.

**Formulae:**
While the `RandomForestClassifier` itself is a complex ensemble method, the underlying principle of classification involves assigning a class label to an input. For a given set of features $X$, the model estimates a probability distribution over the classes $P(Y | X)$. The final prediction is typically the class with the highest estimated probability.

**One-Hot Encoding:** Categorical features, like `patient_gender` or `symptom_severity`, are converted into a numerical format. For a categorical variable with $k$ unique values, one-hot encoding creates $k$ new binary features. For example, `patient_gender` with values 'Male', 'Female', 'Other' would become three new columns: `patient_gender_Male`, `patient_gender_Female`, `patient_gender_Other`, where a 1 indicates the presence of that category and 0 indicates absence.

**Random Forest Classifier:** This model builds an ensemble of decision trees. For classification, each tree in the forest predicts a class, and the class with the most votes across all trees becomes the model's final prediction.
""")

ai_model, X_train, y_train, X_test, y_test, preprocessor = train_ai_model(st.session_state.clinical_data, target_col='true_diagnosis', random_state=42)

st.write(f"\nTraining set size: {len(X_train)}")
st.write(f"Test set size: {len(X_test)}")

st.markdown("""
**Interpretation:**
A `RandomForestClassifier` has been successfully trained on the preprocessed synthetic clinical data. The data was split into training and testing sets, ensuring that the model's performance can be evaluated on unseen data. The `n_estimators` parameter in the `RandomForestClassifier` was set to 100, indicating 100 decision trees are built, and `class_weight='balanced'` was used to address potential class imbalance. This trained model now acts as our AI assistant, ready to provide diagnostic suggestions and associated confidence scores for new cases. The sizes of the training and test sets are displayed, confirming the data split proportions.
""")
```

### AI Predictions and Confidence Scores
```python
def get_ai_predictions_and_confidence(model: RandomForestClassifier, test_features_df: pd.DataFrame, positive_class: str = 'Positive') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates AI diagnostic predictions using the trained model for the test features and extracts the
    corresponding AI confidence scores, specifically the probability of the 'Positive' class.
    """
    if test_features_df.empty:
        return np.array([]), np.array([])
    predictions = model.predict(test_features_df)
    probabilities = model.predict_proba(test_features_df)
    positive_class_idx = np.where(model.classes_ == positive_class)[0][0]
    confidence_scores = probabilities[:, positive_class_idx]
    return predictions, confidence_scores

st.subheader("5.4 Utility Function: Generating AI Predictions and Confidence Scores")
st.markdown("""
**Context & Business Value:**
Beyond just providing a diagnosis, an effective AI-assisted system must convey its certainty. The `get_ai_predictions_and_confidence` function is designed to generate the AI's diagnostic predictions for the test set and, crucially, extract 'confidence scores'. These scores, representing the probability the AI assigns to its prediction, are vital for human operators. In a human-in-the-loop system, they inform human judgment, guiding when to trust the AI and when to scrutinize or override its suggestions, directly impacting system safety and efficiency.

**Formulae:**
The confidence score $C$ for a predicted class is typically given by the predicted probability of that class: 
$$ C = P(\\text{predicted class} | \\text{input features}) $$
For a binary classification problem (e.g., 'Positive' or 'Negative' diagnosis), if the model predicts 'Positive', its confidence is $P(\\text{Positive} | \\text{features})$. If it predicts 'Negative', its confidence is $1 - P(\\text{Positive} | \\text{features})$. In this simulation, we primarily use the probability of the 'Positive' class as the confidence indicator.
""")

ai_predictions, ai_confidence = get_ai_predictions_and_confidence(ai_model, X_test, positive_class='Positive')

simulation_base_df = pd.DataFrame({
    'true_diagnosis': y_test.values,
    'ai_prediction': ai_predictions,
    'ai_confidence': ai_confidence
}, index=y_test.index)

st.write("AI predictions and confidence scores generated.")
st.write("Sample of AI predictions and confidence:")
st.dataframe(simulation_base_df.head())

st.markdown("""
**Interpretation:**
The AI model has successfully generated its diagnostic predictions and corresponding confidence scores for all test cases. The `simulation_base_df` now contains the `true_diagnosis`, the `ai_prediction`, and the `ai_confidence` for each case. The `ai_confidence` represents the probability assigned by the AI to the 'Positive' class. Observing the head of this DataFrame, we can see how the AI's prediction aligns with its confidence. These scores provide an estimate of the model's certainty, which will be a key factor in our human override simulation, allowing human operators to gauge the AI's reliability for each case.
""")
```

### Human Override Mechanism Simulation
```python
def simulate_human_decision(ai_prediction: str, ai_confidence: float, true_label: str,
                                ui_explainability_enabled: bool, anomaly_highlighting_enabled: bool,
                                human_trust_threshold: float, human_expertise_level: float,
                                positive_class: str = 'Positive') -> str:
    """
    Simulates a human operator's decision process, starting with the AI's suggestion and potentially overriding it
    based on AI confidence, UI/UX features, human trust threshold, and expertise level.
    """
    human_decision = ai_prediction
    negative_class_label = 'Negative' if positive_class == 'Positive' else 'Positive'
    ai_predicted_class_confidence = ai_confidence if ai_prediction == positive_class else (1 - ai_confidence)
    is_scrutinized = False
    if ai_predicted_class_confidence < human_trust_threshold:
        is_scrutinized = True
    
    if anomaly_highlighting_enabled:
        is_anomaly_highlighted = (ai_predicted_class_confidence < 0.3) or \
                                 (ai_prediction != positive_class and ai_confidence > 0.7)
        if is_anomaly_highlighted:
            is_scrutinized = True

    if is_scrutinized:
        if ai_prediction != true_label:
            override_success_chance = human_expertise_level
            if ui_explainability_enabled:
                override_success_chance *= 0.6 
            if np.random.rand() < override_success_chance:
                human_decision = true_label
        elif ai_prediction == true_label:
            override_error_chance = (1 - human_expertise_level)
            if not ui_explainability_enabled:
                override_error_chance *= 0.2
            else:
                override_error_chance *= 0.05 
            if np.random.rand() < override_error_chance:
                human_decision = negative_class_label if ai_prediction == positive_class else positive_class
    return human_decision

st.subheader("5.5 Utility Function: Human Override Mechanism Simulation")
st.markdown("""
**Context & Business Value:**
At the heart of 'Human Oversight Roles' is the human's ability to intervene and override AI suggestions. The `simulate_human_decision` function is a crucial component that models this complex human decision-making process. It considers several factors:

-   **AI's Confidence:** How certain the AI is about its own prediction.
-   **UI/UX Features:** Whether AI explainability (e.g., reasoning behind a diagnosis) or anomaly highlighting (e.g., flagging unusual cases) are enabled.
-   **Human Trust Threshold:** An individual human's propensity to trust or question AI suggestions.
-   **Human Expertise Level:** The human operator's inherent skill in correctly identifying and correcting AI errors.

This function introduces a simulated human "error rate" or "override quality" that is dynamically influenced by these parameters. It's designed to illustrate how varying levels of UI/UX support and human capabilities impact the final decision, directly demonstrating the importance of effective UI/UX design and human factors in critical AI applications.
""")
st.info("The `simulate_human_decision` function has been defined, encapsulating human override logic.")
```

### User Interaction Controls
```python
st.subheader("5.6 User Interaction Controls: UI/UX Toggle and Human Factors Setup")
st.markdown("""
**Context & Business Value:**
Effective UI/UX design is paramount for human operators to understand AI outputs and effectively catch anomalies, thereby enhancing overall system safety and performance. This section sets up interactive controls to allow users to dynamically adjust critical parameters. These controls directly influence the `simulate_human_decision` function, enabling a hands-on exploration of how UI/UX features and human factors impact the simulation results.

-   **Show AI Explainability:** Simulates providing explanations for AI predictions, which can increase human understanding and improve override accuracy. This relates to the transparency aspect of trustworthy AI.
-   **Highlight Anomalies:** Simulates flagging cases where AI confidence might be misleading or input features are unusual, prompting human scrutiny and reducing critical errors.
-   **Human Trust Threshold:** Represents the confidence level (between 0 and 1) below which a human operator is more likely to scrutinize the AI's prediction rather than blindly accepting it. This models human psychology in interacting with automated systems.
-   **Human Expertise Level:** Represents the inherent skill or knowledge of the human operator (between 0 and 1) in identifying and correcting AI errors when they choose to intervene. This reflects the varying capabilities of human users.

These interactive elements provide a powerful way to demonstrate the concepts of 'Human Oversight Roles' and the importance of UI/UX in building assurance for critical ML systems.
""")

st.markdown("Adjust the parameters below to rerun the simulation and observe their impact:")

# Initialize session state for widgets if not already present
if 'ui_explainability_enabled' not in st.session_state:
    st.session_state.ui_explainability_enabled = True
if 'anomaly_highlighting_enabled' not in st.session_state:
    st.session_state.anomaly_highlighting_enabled = True
if 'human_trust_threshold' not in st.session_state:
    st.session_state.human_trust_threshold = 0.7
if 'human_expertise_level' not in st.session_state:
    st.session_state.human_expertise_level = 0.6

ui_explainability_enabled = st.checkbox(
    "Show AI Explainability",
    value=st.session_state.ui_explainability_enabled,
    help='When enabled, human operators are better informed and more likely to make correct overrides.',
    key='ui_explainability_enabled_checkbox' # Unique key for widget
)
anomaly_highlighting_enabled = st.checkbox(
    "Highlight Anomalies",
    value=st.session_state.anomaly_highlighting_enabled,
    help='When enabled, potential inconsistencies or low-confidence predictions are flagged, prompting human scrutiny.',
    key='anomaly_highlighting_enabled_checkbox'
)
human_trust_threshold = st.slider(
    "Human Trust Threshold:",
    min_value=0.0, max_value=1.0, value=st.session_state.human_trust_threshold, step=0.05,
    help='AI confidence score below this threshold makes human more likely to scrutinize/override.',
    key='human_trust_slider'
)
human_expertise_level = st.slider(
    "Human Expertise Level:",
    min_value=0.0, max_value=1.0, value=st.session_state.human_expertise_level, step=0.05,
    help='Higher level means human is better at correcting AI errors when they decide to override.',
    key='human_expertise_slider'
)

# Update session state values after interaction
st.session_state.ui_explainability_enabled = ui_explainability_enabled
st.session_state.anomaly_highlighting_enabled = anomaly_highlighting_enabled
st.session_state.human_trust_threshold = human_trust_threshold
st.session_state.human_expertise_level = human_expertise_level

st.markdown("""
**Interpretation:**
The interactive controls for UI/UX features and human parameters (explainability, anomaly highlighting, trust threshold, expertise level) are now defined. These widgets allow dynamic interaction with the simulation. Their values will serve as inputs to our simulation functions, allowing users to tune and observe their effects on system performance in real-time, highlighting the importance of human-centered design in AI systems.
""")
```

### Simulate Clinical Case Review and Override Decisions
```python
def run_full_simulation(
    simulation_df: pd.DataFrame,
    ui_explainability_enabled: bool,
    anomaly_highlighting_enabled: bool,
    human_trust_threshold: float,
    human_expertise_level: float
) -> pd.DataFrame:
    """
    Orchestrates the human-in-the-loop simulation loop for all test cases, iteratively calling
    simulate_human_decision for each case based on the AI's output and current UI/UX and human parameter settings.
    """
    result_df = simulation_df.copy()
    if result_df.empty:
        result_df['human_final_decision'] = pd.Series(dtype='object')
        return result_df

    human_decisions = []
    for _, row in simulation_df.iterrows():
        decision = simulate_human_decision(
            ai_prediction=row['ai_prediction'],
            ai_confidence=row['ai_confidence'],
            true_label=row['true_diagnosis'],
            ui_explainability_enabled=ui_explainability_enabled,
            anomaly_highlighting_enabled=anomaly_highlighting_enabled,
            human_trust_threshold=human_trust_threshold,
            human_expertise_level=human_expertise_level
        )
        human_decisions.append(decision)
    result_df['human_final_decision'] = human_decisions
    return result_df

st.subheader("5.7 Utility Function: Simulate Clinical Case Review and Override Decisions")
st.markdown("""
**Context & Business Value:**
This section orchestrates the full human-in-the-loop simulation, mimicking the sequential review of clinical cases. The `run_full_simulation` function takes the AI's predictions and confidence scores, and for each case, applies the logic of our `simulate_human_decision` function. This iterative process generates a `human_final_decision` for every case, reflecting the combined intelligence of the AI and the human operator, influenced by UI/UX settings and human factors.

This function is crucial for demonstrating the end-to-end flow of an AI-assisted decision system and for capturing the aggregate impact of human intervention. The output, a DataFrame containing both AI's initial suggestions and human's final decisions, forms the basis for all subsequent performance analysis and visualization, directly linking to the assessment of 'Human Oversight Roles' and system safety.
""")

# Run initial simulation with default/current widget values
initial_simulation_results_df = run_full_simulation(
    simulation_df=simulation_base_df,
    ui_explainability_enabled=st.session_state.ui_explainability_enabled,
    anomaly_highlighting_enabled=st.session_state.anomaly_highlighting_enabled,
    human_trust_threshold=st.session_state.human_trust_threshold,
    human_expertise_level=st.session_state.human_expertise_level
)
st.write("Initial clinical case review simulation completed.")
st.write("Sample of simulation results (AI vs Human decision):")
st.dataframe(initial_simulation_results_df.head())

st.markdown("""
**Interpretation:**
The initial clinical case review simulation has been executed using the default settings for UI/UX features and human parameters. The `initial_simulation_results_df` now contains the `true_diagnosis`, `ai_prediction`, `ai_confidence`, and the `human_final_decision` for each case. Observing the sample, we can see instances where the `human_final_decision` might differ from the `ai_prediction`, indicating an override. This DataFrame provides a comprehensive record of both AI's initial suggestions and the human operator's final decisions, setting the stage for quantitative performance analysis.
""")
```

### Performance Metrics Calculation
```python
def calculate_performance_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray, positive_class: str = 'Positive') -> Dict[str, float]:
    """Calculates key performance metrics for a classification system, including accuracy score,
    false positive rate (FPR), and false negative rate (FNR).
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length.")
    if len(true_labels) == 0: # Handle empty input gracefully
        return {'Accuracy': 0.0, 'False Positive Rate': 0.0, 'False Negative Rate': 0.0}

    is_true_positive = (true_labels == positive_class)
    is_predicted_positive = (predicted_labels == positive_class)

    TP = np.sum(is_true_positive & is_predicted_positive)
    TN = np.sum(~is_true_positive & ~is_predicted_positive)
    FP = np.sum(~is_true_positive & is_predicted_positive)
    FN = np.sum(is_true_positive & ~is_predicted_positive)

    total_samples = len(true_labels)
    accuracy = (TP + TN) / total_samples if total_samples > 0 else 0.0

    actual_negatives = FP + TN
    fpr = FP / actual_negatives if actual_negatives > 0 else 0.0

    actual_positives = FN + TP
    fnr = FN / actual_positives if actual_positives > 0 else 0.0

    return {
        'Accuracy': accuracy,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr
    }

st.subheader("5.8 Utility Function: Performance Metrics Calculation")
st.markdown("""
**Context & Business Value:**
To rigorously assess the impact of human oversight, we need a robust set of performance metrics. The `calculate_performance_metrics` function quantifies the effectiveness of both the AI-only system and the combined AI+Human system. By calculating Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR, often critical in clinical settings), we gain a comprehensive understanding of how human intervention alters the system's propensity for different types of errors.

This function is crucial for evaluating the 'fit-for-purpose' aspect of the AI system, particularly in safety-critical domains like clinical decision support. Minimizing False Negatives, for example, could be a primary business objective to avoid missing critical diagnoses, and this function allows us to measure progress towards that goal.

**Formulae:**

*   **Accuracy:** The proportion of correct predictions (both positive and negative) out of the total number of cases.
    $$ \\text{Accuracy} = \\frac{\\text{TP} + \\text{TN}}{\\text{TP} + \\text{TN} + \\text{FP} + \\text{FN}} $$

*   **False Positive Rate (FPR):** The proportion of actual negative cases that are incorrectly predicted as positive.
    $$ \\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}} $$

*   **False Negative Rate (FNR):** The proportion of actual positive cases that are incorrectly predicted as negative.
    $$ \\text{FNR} = \\frac{\\text{FN}}{\\text{FN} + \\text{TP}} $$

Where:
-   TP (True Positives): Cases where the true label is 'Positive' and the prediction is 'Positive'.
-   TN (True Negatives): Cases where the true label is 'Negative' and the prediction is 'Negative'.
-   FP (False Positives): Cases where the true label is 'Negative' but the prediction is 'Positive' (Type I error).
-   FN (False Negatives): Cases where the true label is 'Positive' but the prediction is 'Negative' (Type II error).
""")
st.info("The `calculate_performance_metrics` function has been defined.")
```

### Initial Performance Calculation
```python
ai_only_initial_metrics = calculate_performance_metrics(
    initial_simulation_results_df['true_diagnosis'].values,
    initial_simulation_results_df['ai_prediction'].values
)

ai_human_initial_metrics = calculate_performance_metrics(
    initial_simulation_results_df['true_diagnosis'].values,
    initial_simulation_results_df['human_final_decision'].values
)

st.subheader("5.9 Initial Performance Calculation")
st.markdown("""
**Context & Business Value:**
Before diving into interactive analysis, it's essential to establish a baseline. This section calculates and displays the initial performance metrics for both the AI-only system and the AI+Human system based on the first simulation run (using default UI/UX settings). This provides a clear quantitative snapshot of the current state, allowing us to immediately see the benefits or trade-offs of human oversight with a specific configuration.

Understanding this baseline is critical for evaluating the effectiveness of our human-in-the-loop design. For clinical decision support, for instance, we might observe how human intervention initially affects the False Negative Rate, which is often a key safety metric. This initial assessment directly supports the learning goals of understanding 'Human Oversight Roles' and their impact on system safety.
""")
st.write("--- Initial Performance Metrics ---")
st.write("AI-Only System Performance:")
for metric, value in ai_only_initial_metrics.items():
    st.write(f"  {metric}: {value:.4f}")

st.write("\nAI+Human System Performance (with default settings):")
for metric, value in ai_human_initial_metrics.items():
    st.write(f"  {metric}: {value:.4f}")
st.write("-----------------------------------")

st.markdown("""
**Interpretation:**
The initial performance metrics for both the AI-only and AI+Human systems (with default settings) have been computed and displayed. This output provides a quantitative baseline, showing the accuracy, false positive rate (FPR), and false negative rate (FNR) for each system. We can now directly compare how the introduction of human oversight, even with initial configurations, impacts these critical metrics. For example, if the FNR for the AI+Human system is lower than the AI-only system, it suggests that human intervention is effectively reducing critical missed diagnoses, aligning with the goal of improving system safety.
""")
```

### Visualizations
```python
def plot_performance_comparison(ai_only_metrics: Dict[str, Any], ai_human_metrics: Dict[str, Any]):
    """Generates a bar chart comparing AI-only vs. AI+Human system performance."""
    metrics_set = set(ai_only_metrics.keys()) | set(ai_human_metrics.keys())
    metrics = sorted(metrics_set, key=lambda k: (str(k).lower(), str(k)))

    def to_float(val):
        try:
            return float(val)
        except Exception:
            return float("nan")

    ai_only_values = [to_float(ai_only_metrics.get(m, float("nan"))) for m in metrics]
    ai_human_values = [to_float(ai_human_metrics.get(m, float("nan"))) for m in metrics]

    metrics_df = pd.DataFrame({
        'Metric': metrics * 2,
        'Value': ai_only_values + ai_human_values,
        'System': ['AI-Only'] * len(metrics) + ['AI+Human'] * len(metrics)
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', hue='System', data=metrics_df, palette='viridis')
    plt.title('Comparative System Performance (AI-Only vs. AI+Human)', fontsize=16)
    plt.xlabel('Performance Metric', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='System Type', fontsize=12, title_fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot()
    plt.close()

st.subheader("5.10 Visualization: Comparative Performance Analysis (Aggregated Comparison)")
st.markdown("""
**Context & Business Value:**
To provide an intuitive and clear understanding of the impact of human oversight, we will visualize the performance metrics using a bar chart. This `plot_performance_comparison` function will compare the Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR) between the AI-only and AI+Human systems. This visual comparison quickly conveys whether human intervention is beneficial, particularly in mitigating critical errors like false negatives in safety-critical clinical decision support.

The use of a color-blind-friendly palette and clear labels ensures that the insights are accessible and understandable to all stakeholders. This visualization directly supports the learning goal of understanding how human intervention affects system performance and safety, providing a direct link between technical metrics and business outcomes.
""")
plot_performance_comparison(ai_only_initial_metrics, ai_human_initial_metrics)
st.markdown("""
**Interpretation:**
The bar chart clearly illustrates the differences in accuracy, false positive rate, and false negative rate between the AI-only system and the system incorporating human oversight (with default settings). This visual comparison provides immediate insights into the benefits or trade-offs of human intervention. For instance, we can observe if human oversight has led to a reduction in critical errors like false negatives, which is often a key safety objective in clinical decision support. The `viridis` palette ensures the visualization is accessible and clear to a broad audience.
""")

def plot_confidence_vs_override(simulation_results_df: pd.DataFrame) -> None:
    """
    Generates a scatter plot to visualize the relationship between AI confidence scores
    and the frequency of human overrides.
    """
    df = simulation_results_df.copy()
    if df.empty:
        plot_df = pd.DataFrame(columns=['confidence_midpoint', 'Override_Frequency'])
    else:
        df['override_decision'] = (df['human_final_decision'] != df['ai_prediction'])
        bins = np.linspace(0, 1, 11)
        df['confidence_bin'] = pd.cut(df['ai_confidence'], bins=bins, include_lowest=True)
        grouped_data = df.groupby('confidence_bin', observed=False)['override_decision'].mean().reset_index()
        grouped_data = grouped_data.rename(columns={'override_decision': 'Override_Frequency'})
        grouped_data['confidence_midpoint'] = grouped_data['confidence_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
        plot_df = grouped_data.dropna(subset=['Override_Frequency']).copy()

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='confidence_midpoint',
        y='Override_Frequency',
        data=plot_df,
        s=150, alpha=0.8, color=sns.color_palette("deep")[0],
        edgecolor='black', linewidth=0.7
    )
    plt.title('AI Confidence vs. Human Override Frequency', fontsize=16)
    plt.xlabel('AI Confidence (Probability of Positive Class)', fontsize=14)
    plt.ylabel('Frequency of Human Override', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(-0.05, 1.05)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=10)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot()
    plt.close()

st.subheader("5.11 Visualization: Impact of UI/UX on Override Frequency (Relationship Plot)")
st.markdown("""
**Context & Business Value:**
Understanding when and why human operators choose to override AI suggestions is critical for optimizing human-in-the-loop systems. The `plot_confidence_vs_override` function generates a scatter plot that visualizes the relationship between the AI's confidence scores and the frequency of human overrides. This visualization helps us determine if humans are primarily intervening when the AI is less confident, or if other factors, such as perceived anomalies (potentially influenced by UI/UX features), drive their intervention patterns.

This analysis provides valuable feedback for UI/UX designers, helping them understand if their designs effectively guide human attention to high-risk or low-confidence AI predictions. It directly supports the learning goal of exploring the importance of designing effective UI/UX to help users understand model outputs and catch anomalies.
""")
plot_confidence_vs_override(initial_simulation_results_df)
st.markdown("""
**Interpretation:**
The scatter plot visualizes the interplay between AI confidence and human override decisions. Each point represents a bin of AI confidence scores, with the y-axis showing the frequency of human overrides within that confidence range. This plot helps us understand if human operators are effectively targeting low-confidence AI suggestions, or if other UI/UX cues (like anomaly highlighting) influence their intervention patterns, leading to overrides even at moderate AI confidence levels. A higher frequency of overrides at lower AI confidence would suggest effective human scrutiny where the AI is less certain.
""")

class PlottingError(Exception):
    """Custom exception for plotting-related errors."""
    pass

def plot_trend_metrics(simulation_results_df: pd.DataFrame, window_size: int = 50) -> None:
    """
    Generates a line plot displaying rolling average performance metrics (e.g., accuracy) over the sequence of simulated decisions,
    comparing AI-only vs. AI+Human trends.
    """
    required_columns = ['true_diagnosis', 'ai_prediction', 'human_final_decision']
    if not all(col in simulation_results_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in simulation_results_df.columns]
        st.error(f"DataFrame is missing required columns for accuracy calculation: {missing_cols}")
        return
    if not isinstance(window_size, int):
        st.error("window_size must be an integer.")
        return
    if window_size <= 0:
        st.error("window_size must be a positive integer (>= 1).")
        return
    if simulation_results_df.empty:
        st.warning("Cannot plot trend metrics: DataFrame is empty.")
        return

    df = simulation_results_df.copy()
    df['ai_only_correct'] = (df['true_diagnosis'] == df['ai_prediction']).astype(int)
    df['ai_human_correct'] = (df['true_diagnosis'] == df['human_final_decision']).astype(int)
    
    df['rolling_ai_accuracy'] = df['ai_only_correct'].rolling(window=window_size).mean()
    df['rolling_ai_human_accuracy'] = df['ai_human_correct'].rolling(window=window_size).mean()
    
    plt.figure(figsize=(12, 7))
    sns.set_palette("colorblind")
    
    sns.lineplot(x=df.index, y='rolling_ai_accuracy', data=df, 
                 label='AI-Only Rolling Accuracy', linewidth=2)
    sns.lineplot(x=df.index, y='rolling_ai_human_accuracy', data=df, 
                 label='AI+Human Rolling Accuracy', linewidth=2)
    
    plt.title(f'Rolling Average Performance Trends (Window Size: {window_size})', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Rolling Accuracy', fontsize=12)
    plt.legend(title='Metric', fontsize=10, title_fontsize='12', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    st.pyplot()
    plt.close()

st.subheader("5.12 Visualization: System Performance Over Time (Trend Plot)")
st.markdown("""
**Context & Business Value:**
In real-world operational settings, decision-making is a continuous process. It's crucial to monitor how system performance evolves over time to detect any drift or changes in effectiveness. The `plot_trend_metrics` function generates a line plot displaying the rolling average accuracy for both the AI-only and AI+Human systems over the sequence of simulated cases. This mimics real-time performance monitoring and helps to identify trends or fluctuations.

This visualization is vital for assessing the long-term stability and consistency of the human-in-the-loop system. It can highlight whether human intervention effectively smooths out performance variations, maintains a higher baseline, or adapts to changing conditions over time. Such insights are essential for ensuring the ongoing assurance and reliability of critical AI systems.
""")
plot_trend_metrics(initial_simulation_results_df, window_size=50)
st.markdown("""
**Interpretation:**
The trend plot displays how the rolling average accuracy changes over the sequence of simulated clinical cases for both the AI-only and AI+Human systems. The `window_size` of 50 allows for a smoothed view of performance over time. This visualization provides insights into the stability and consistency of each system during continuous operation. We can observe if human intervention helps to smooth out performance fluctuations, maintain a higher baseline accuracy, or adapt to changing conditions compared to the AI operating autonomously. This helps to understand the dynamic impact of human-in-the-loop systems on long-term performance and reliability.
""")

st.subheader("5.13 User Interaction for Parameter Tuning")
st.markdown("""
**Context & Business Value:**
To further explore the dynamic interplay between human factors and UI/UX design, we provide interactive sliders and checkboxes. These controls allow you to adjust critical parameters such as the human's `human_trust_threshold` in the AI, their `human_expertise_level`, and whether UI/UX features like `ui_explainability_enabled` or `anomaly_highlighting_enabled` are active. By changing these parameters, you can dynamically observe how they affect the simulation results.

This interactive section is crucial for hands-on learning and demonstrating the sensitivity of human-in-the-loop systems to both technological design and human factors. It directly addresses the learning goals of exploring the importance of effective UI/UX and understanding 'Human Oversight Roles' by allowing users to experiment with different scenarios and immediately see the consequences of their choices.

Each control includes inline help text to explain its purpose and guide experimentation.
""")
st.info("The interactive parameter tuning controls are active. Adjust the parameters in the sidebar to see dynamic updates.")

st.markdown("""
**Interpretation:**
The interactive parameter tuning controls are now active. The checkboxes allow toggling UI/UX features like AI explainability and anomaly highlighting, while the sliders enable adjusting the human's trust threshold and expertise level. Experiment with different settings for these parameters to dynamically observe their influence on the simulation results in the following re-analysis step. This interactive exploration will provide direct insights into how these factors contribute to the overall performance and safety of the human-in-the-loop system.
""")

st.subheader("5.14 Interactive Rerun: Rerunning Analysis with New Parameters")
st.markdown("""
**Context & Business Value:**
This section allows for a powerful interactive analysis by rerunning the entire simulation and plotting pipeline with parameters adjusted using the widgets above. Streamlit's natural rerunning behavior automatically links the widget values to the analysis, triggering a complete re-evaluation of the human-in-the-loop system whenever a parameter is changed. This real-time feedback loop is invaluable for understanding cause-and-effect relationships.

By dynamically modifying the `human_trust_threshold`, `human_expertise_level`, and UI/UX toggles, users can directly observe their impact on performance metrics and visualizations. This direct experimentation provides concrete evidence for how different design choices and human capabilities affect accuracy, false positive rates, false negative rates, and override patterns. This is a core component of evaluating uncertainty and model risk in real-world conditions and translating "fit-for-purpose" into measurable evidence.
""")

# Rerun analysis with current parameters (Streamlit's natural flow handles this)
st.markdown("---")
st.markdown("#### Dynamic Simulation Results")
st.info("Change the values in the widgets above to see dynamic updates below.")

current_simulation_results_df = run_full_simulation(
    simulation_df=simulation_base_df,
    ui_explainability_enabled=st.session_state.ui_explainability_enabled,
    anomaly_highlighting_enabled=st.session_state.anomaly_highlighting_enabled,
    human_trust_threshold=st.session_state.human_trust_threshold,
    human_expertise_level=st.session_state.human_expertise_level
)

ai_only_metrics = calculate_performance_metrics(
    current_simulation_results_df['true_diagnosis'].values,
    current_simulation_results_df['ai_prediction'].values
)
ai_human_metrics = calculate_performance_metrics(
    current_simulation_results_df['true_diagnosis'].values,
    current_simulation_results_df['human_final_decision'].values
)

st.write("\n--- Updated Performance Metrics ---")
st.write("AI-Only System Performance:")
for metric, value in ai_only_metrics.items():
    st.write(f"  {metric}: {value:.4f}")

st.write("\nAI+Human System Performance (with current settings):")
for metric, value in ai_human_metrics.items():
    st.write(f"  {metric}: {value:.4f}")
st.write("-----------------------------------")

st.write("\nUpdating Comparative Performance Plot...")
plot_performance_comparison(ai_only_metrics, ai_human_metrics)

st.write("\nUpdating AI Confidence vs. Human Override Plot...")
plot_confidence_vs_override(current_simulation_results_df)

st.write("\nUpdating Performance Trend Plot...")
plot_trend_metrics(current_simulation_results_df)

st.markdown("""
**Interpretation:**
The simulation and analysis pipeline has been made interactive. By adjusting the widgets, this section automatically re-executes the simulation, recalculates performance metrics, and updates all three visualization plots. This dynamic interactivity allows for immediate observation of how changes in human factors and UI/UX features alter the comparative performance, override patterns, and performance trends of the human-in-the-loop system. This provides a powerful, empirical demonstration of the concepts discussed in 'Assurance Foundations for Critical ML'.
""")

st.header("6. Conclusion and Key Takeaways")
st.markdown("""
This simulation has provided a practical demonstration of 'Human Oversight Roles' in AI-assisted decision-making. We've seen how human intervention can act as a crucial fallback mechanism, potentially enhancing system reliability and safety by overriding AI suggestions. The lab highlighted:

- The quantifiable impact of human oversight on overall system performance metrics like accuracy, false positive rates, and false negative rates.
- The significance of designing effective UI/UX, as features like AI explainability and anomaly highlighting can empower human operators to make more informed and accurate override decisions.
- The interplay between AI confidence, human trust, and the quality of human intervention.

Ultimately, this exercise underscores the importance of a well-designed human-in-the-loop system to ensure "fit-for-purpose" AI in critical applications like clinical decision support, aligning with the principles of assurance foundations for critical ML.

**Next Steps & Productionization Notes:**
-   **Real-world Data Integration:** While this simulation uses synthetic data, the next logical step would be to integrate real (anonymized) clinical data to validate the findings in a more realistic context.
-   **Advanced UI/UX Design:** Explore more sophisticated UI/UX elements, such as interactive dashboards that provide detailed explanations for AI predictions, to further empower human operators.
-   **Human Factors Engineering:** Conduct user studies with clinical professionals to gather feedback on the human-AI interaction, refine the override mechanisms, and optimize trust and workload.
-   **Continuous Monitoring:** Implement a continuous monitoring system for AI and human performance in live operation, tracking metrics and identifying potential drift or emerging risks.
-   **Ethical and Regulatory Compliance:** Ensure that the human-in-the-loop system adheres to relevant ethical guidelines and regulatory requirements for AI in healthcare.
""")

st.header("7. References")
st.markdown("""
[1] Unit 3: Assurance Foundations for Critical ML, 'Human Oversight Roles', [Case 2: Generative AI in Clinical Decision Support (Provided Resource)]. This section discusses roles for humans in the loop and the importance of designing UI/UX to help users understand model outputs.
""")
```
