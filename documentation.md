id: 690b6ec5adbb0296a6bfadf2_documentation
summary: Generative AI in Clinical Decision Support Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Human-in-the-Loop AI for Clinical Decision Support: A Streamlit Simulation

## 1. Introduction: Setting the Stage for Trustworthy AI
Duration: 00:05:00
In this lab, we dive into the critical domain of **Generative AI in Clinical Decision Support**, with a particular focus on the indispensable role of **Human Oversight** in ensuring system safety and performance. This interactive Streamlit application simulates a human-in-the-loop system, where you, as a developer and simulated human operator, will review AI-generated diagnostic suggestions in a synthetic clinical environment.

The core of this simulation lies in observing how your decisions to override or accept AI recommendations directly impact the system's overall accuracy, false positive rates (FPR), and false negative rates (FNR). This hands-on experience is designed to provide practical insights into:

*   <b>Assurance-Case Methodology</b>: How to systematically evaluate uncertainty and model risk in real-world conditions, translating "fit-for-purpose" into measurable evidence.
*   <b>UI/UX Design for Trustworthy AI</b>: The profound importance of effective user interface and user experience design in empowering human operators to understand complex AI model outputs, identify anomalies, and make informed decisions.
*   <b>Human Oversight Roles</b>: The dynamic interplay between human intervention and AI, and how it fundamentally shapes system safety and operational performance in critical applications.

By adjusting various simulation parameters, you will explore how human factors, AI explainability, and anomaly highlighting influence the collaborative decision-making process, ultimately leading to more robust and reliable clinical outcomes.

The overarching principle guiding our exploration can be summarized by the following relationship:
$$\text{AI-assisted System Performance} = f(\text{AI Model Performance}, \text{Human Oversight Quality}, \text{UI/UX Effectiveness})$$

Where:

*   $\text{AI Model Performance}$: The inherent accuracy and reliability of the underlying AI algorithm.
*   $\text{Human Oversight Quality}$: The human operator's ability to correctly identify and intervene in AI errors, influenced by their expertise and trust in the AI.
*   $\text{UI/UX Effectiveness}$: How well the user interface communicates AI predictions, confidence, and anomalies, thereby enabling informed human decisions.

Let's begin by understanding the architecture of our application.

### Application Architecture
The application follows a modular structure, typical for Streamlit multi-page applications, organizing functionalities into distinct Python files for clarity and maintainability.

```
.
├── app.py                      # Main Streamlit application entry point
└── application_pages/
    ├── __init__.py
    ├── page1.py                # Data generation, validation, AI model training
    ├── page2.py                # Human override simulation logic, UI/UX controls
    └── page3.py                # Performance metrics calculation, result visualization, interactive analysis
```

<aside class="positive">
This modular design enhances code readability and allows for better separation of concerns, making it easier to develop, debug, and scale the application.
</aside>

## 2. Environment Setup and Project Structure
Duration: 00:03:00

Before running the application, you'll need to set up your environment.

### 2.1 Clone the Repository (Simulated)
In a real-world scenario, you would clone the repository containing the application code. For this codelab, assume the files are already available in your working directory.

### 2.2 Install Dependencies
The application relies on several Python libraries. You would typically install them using `pip`.

```console
pip install streamlit pandas numpy scikit-learn plotly
```

### 2.3 Run the Streamlit Application
Navigate to the root directory where `app.py` is located and run the Streamlit application:

```console
streamlit run app.py
```

This command will open the application in your web browser. You'll see a sidebar for navigation and the main content area.

### 2.4 Understanding the Code Structure
The application is structured into a main `app.py` file and three sub-modules within the `application_pages` directory, each responsible for a distinct part of the simulation flow:

*   **`app.py`**: This is the entry point. It sets up the Streamlit page configuration, displays the main title and introduction, and handles navigation between the different pages (`Page 1`, `Page 2`, `Page 3`) using a `st.sidebar.selectbox`.

    ```python
    import streamlit as st

    st.set_page_config(page_title="QuLab", layout="wide")
    st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
    st.sidebar.divider()
    st.title("QuLab")
    st.divider()
    st.markdown("""...""") # Introduction text
    
    page = st.sidebar.selectbox(label="Navigation", options=["Page 1: Data & AI Model", "Page 2: Simulation Controls", "Page 3: Results & Analysis"])
    if page == "Page 1: Data & AI Model":
        from application_pages.page1 import run_page1
        run_page1()
    elif page == "Page 2: Simulation Controls":
        from application_pages.page2 import run_page2
        run_page2()
    elif page == "Page 3: Results & Analysis":
        from application_pages.page3 import run_page3
        run_page3()
    ```

*   **`application_pages/page1.py`**: Handles the generation of synthetic clinical data, performs initial data validation, and trains the AI diagnostic model.
*   **`application_pages/page2.py`**: Contains the logic for simulating human decision-making and provides interactive controls for adjusting human factors and UI/UX features.
*   **`application_pages/page3.py`**: Focuses on calculating and visualizing performance metrics for both AI-only and AI+Human systems, allowing for interactive analysis of the simulation results.

## 3. Data Generation and Validation (Page 1)
Duration: 00:10:00

Navigate to **"Page 1: Data & AI Model"** using the sidebar. This page is dedicated to preparing our simulated environment.

### 3.1 Utility Function: Generating Synthetic Clinical Data
To create a safe and controlled environment for our simulation, we start by generating a synthetic dataset of patient records. This function creates diverse patient profiles with various clinical features and a `true_diagnosis`, allowing us to experiment without privacy concerns.

```python
# From application_pages/page1.py

import pandas as pd
import numpy as np
import streamlit as st # Only for Streamlit specific functions like st.cache_data

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

    # Logic to create correlation with true_diagnosis
    p_positive = np.full(num_cases, 0.15, dtype=float)
    p_positive += 0.005 * (data['patient_age'] - 18)
    p_positive += 0.1 * (data['symptom_severity'] - 1)
    p_positive += 0.2 * data['previous_diagnosis']
    is_male = (data['patient_gender'] == 'Male')
    p_positive[is_male] += 0.05
    p_positive = np.clip(p_positive, 0.05, 0.95)

    is_positive_diagnosis = np.random.rand(num_cases) < p_positive
    data['true_diagnosis'] = np.where(is_positive_diagnosis, 'Positive', 'Negative')

    # Lab results correlated with diagnosis
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
```

This function is crucial as it provides the foundational input for the entire system, setting the stage for training the AI model and simulating human interactions. The `@st.cache_data` decorator ensures that the data generation is only run once unless its input parameters change.

You will see output similar to this:

```
Generated a dataset of 500 synthetic clinical cases.
First 5 rows of the dataset:
```

| case_id | patient_age | patient_gender | lab_result_A | lab_result_B | symptom_severity | previous_diagnosis | true_diagnosis |
||-|-|--|--||--|-|
| 0       | 52          | Female         | 7.394017     | 5.176465     | 5                | False              | Positive       |
| 1       | 50          | Female         | 5.483988     | 5.568102     | 4                | False              | Negative       |
| 2       | 36          | Male           | 9.075459     | 4.653480     | 2                | False              | Negative       |
| 3       | 58          | Female         | 10.370894    | 6.940003     | 5                | True               | Positive       |
| 4       | 74          | Male           | 10.428405    | 6.223528     | 3                | False              | Positive       |

### 3.2 Utility Function: Data Validation and Initial Exploration
Data validation is critical in clinical applications. The `perform_data_validation` function ensures our synthetic data conforms to expected standards, checking column names, data types, uniqueness of `case_id`, and absence of missing values in critical fields.

```python
# From application_pages/page1.py

EXPECTED_DTYPES = {
    'case_id': np.dtype('int64'),
    'patient_age': np.dtype('int64'),
    'patient_gender': np.dtype('object'),
    'lab_result_A': np.dtype('float64'),
    'lab_result_B': np.dtype('float64'),
    'symptom_severity': np.dtype('int64'),
    'previous_diagnosis': np.dtype('bool'),
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

    # Validation logic (as seen in the provided code)
    # ... checks for column presence, dtype, uniqueness of case_id, missing values, and displays summary stats.
    # [Code as provided in page1.py, omitted for brevity in codelab markdown]
```

<aside class="positive">
By confirming data integrity upfront, we ensure that our AI model training and human override simulations are based on a sound and consistent dataset, which is fundamental for generating trustworthy insights into assurance foundations.
</aside>

After this step, you will see a "Data Validation Report" in the Streamlit app confirming the data quality.

## 4. AI Model Training and Prediction (Page 1)
Duration: 00:10:00

Still on **"Page 1: Data & AI Model"**, we now move to establish our simulated AI assistant.

### 4.1 Utility Function: AI Model Training (Simulated)
The `train_ai_model` function builds our AI, a `RandomForestClassifier`, which predicts `true_diagnosis`. This model acts as the "AI suggestion" for human operators to review. Before training, it preprocesses categorical features using one-hot encoding and splits data into training and testing sets.

```python
# From application_pages/page1.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# ... other imports

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
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()

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
```

**Formulae:**

*   **One-Hot Encoding:** Categorical features (e.g., `patient_gender`) are converted into a numerical format. For a categorical variable with $k$ unique values, one-hot encoding creates $k$ new binary features. For example, `patient_gender` with values 'Male', 'Female', 'Other' would become three new columns: `patient_gender_Male`, `patient_gender_Female`, `patient_gender_Other`, where a 1 indicates the presence of that category and 0 indicates absence.

*   **Random Forest Classifier:** This model builds an ensemble of decision trees. For classification, each tree in the forest predicts a class, and the class with the most votes across all trees becomes the model's final prediction.

The trained AI model is critical for demonstrating how human oversight can enhance or correct AI outputs, thereby directly addressing the learning goals related to human oversight roles and assurance foundations.

### 4.2 Utility Function: Generating AI Predictions and Confidence Scores
An effective AI-assisted system must convey its certainty. The `get_ai_predictions_and_confidence` function generates the AI's diagnostic predictions and, crucially, extracts 'confidence scores'. These scores are vital for human operators, guiding when to trust the AI and when to scrutinize or override its suggestions.

```python
# From application_pages/page1.py

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
```

**Formulae:**
The confidence score $C$ for a predicted class is typically given by the predicted probability of that class: 
$$ C = P(\text{predicted class} | \text{input features}) $$ 
For a binary classification problem (e.g., 'Positive' or 'Negative' diagnosis), if the model predicts 'Positive', its confidence is $P(\text{Positive} | \text{features})$. If it predicts 'Negative', its confidence is $1 - P(\text{Positive} | \text{features})$. In this simulation, we primarily use the probability of the 'Positive' class as the confidence indicator.

You will see the sample of AI predictions and confidence scores in a dataframe, which forms the `simulation_base_df` in `st.session_state`.

## 5. Simulating Human Decision Making (Page 2)
Duration: 00:15:00

Now, navigate to **"Page 2: Simulation Controls"** using the sidebar. This is where the human-in-the-loop simulation truly comes alive.

### 5.1 Utility Function: Human Override Mechanism Simulation
The `simulate_human_decision` function is a crucial component that models the complex human decision-making process, allowing humans to intervene and override AI suggestions.

```python
# From application_pages/page2.py

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
    
    # Human scrutiny logic
    if ai_predicted_class_confidence < human_trust_threshold:
        is_scrutinized = True
    
    if anomaly_highlighting_enabled:
        # Simulate anomaly detection, e.g., low confidence for predicted, or high confidence for non-predicted
        is_anomaly_highlighted = (ai_predicted_class_confidence < 0.3) or \
                                 (ai_prediction != positive_class and ai_confidence > 0.7)
        if is_anomaly_highlighted:
            is_scrutinized = True

    # Human override logic if scrutinized
    if is_scrutinized:
        if ai_prediction != true_label: # AI made a mistake
            override_success_chance = human_expertise_level
            if ui_explainability_enabled:
                override_success_chance *= 0.6 # Explainability helps improve correction chance
            if np.random.rand() < override_success_chance:
                human_decision = true_label # Human correctly overrides
        elif ai_prediction == true_label: # AI was correct
            override_error_chance = (1 - human_expertise_level)
            if not ui_explainability_enabled:
                override_error_chance *= 0.2 # Without explainability, higher chance of incorrect override
            else:
                override_error_chance *= 0.05 # With explainability, lower chance of incorrect override
            if np.random.rand() < override_error_chance:
                # Human incorrectly overrides a correct AI prediction
                human_decision = negative_class_label if ai_prediction == positive_class else positive_class
    return human_decision
```

This function introduces a simulated human "error rate" or "override quality" that is dynamically influenced by:
*   **AI's Confidence**: How certain the AI is about its own prediction.
*   **UI/UX Features**: Whether AI explainability or anomaly highlighting are enabled.
*   **Human Trust Threshold**: An individual human's propensity to trust or question AI suggestions.
*   **Human Expertise Level**: The human operator's inherent skill in correctly identifying and correcting AI errors.

This simulation demonstrates how varying levels of UI/UX support and human capabilities impact the final decision, directly illustrating the importance of effective UI/UX design and human factors in critical AI applications.

### 5.2 User Interaction Controls: UI/UX Toggle and Human Factors Setup
Effective UI/UX design is paramount for human operators. This section provides interactive controls to dynamically adjust critical parameters, enabling a hands-on exploration of their impact.

```python
# From application_pages/page2.py

# Initialize session state for widgets if not already present
# ... (Initialization code as seen in page2.py)

ui_explainability_enabled = st.checkbox(
    "Show AI Explainability",
    value=st.session_state.ui_explainability_enabled,
    help='When enabled, human operators are better informed and more likely to make correct overrides.',
    key='ui_explainability_enabled_checkbox'
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
# ... (Update code as seen in page2.py)
```

<aside class="positive">
These interactive elements provide a powerful way to demonstrate the concepts of 'Human Oversight Roles' and the importance of UI/UX in building assurance for critical ML systems.
</aside>

### 5.3 Utility Function: Simulate Clinical Case Review and Override Decisions
This function orchestrates the full human-in-the-loop simulation, mimicking the sequential review of clinical cases. It applies the logic of our `simulate_human_decision` for each case, generating a `human_final_decision` that reflects the combined intelligence of the AI and the human operator.

```python
# From application_pages/page2.py

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
```

This function is crucial for demonstrating the end-to-end flow of an AI-assisted decision system and for capturing the aggregate impact of human intervention. You will see a sample of the simulation results, showing the AI's initial prediction versus the human's final decision.

## 6. Analyzing Simulation Results (Page 3)
Duration: 00:20:00

Now, navigate to **"Page 3: Results & Analysis"** using the sidebar. This page provides tools to quantify and visualize the impact of human oversight.

### 6.1 Utility Function: Performance Metrics Calculation
To rigorously assess the impact of human oversight, we need a robust set of performance metrics. The `calculate_performance_metrics` function quantifies the effectiveness of both the AI-only system and the combined AI+Human system.

```python
# From application_pages/page3.py

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
```

**Formulae:**

*   **Accuracy:** The proportion of correct predictions (both positive and negative) out of the total number of cases.
    $$ \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} $$

*   **False Positive Rate (FPR):** The proportion of actual negative cases that are incorrectly predicted as positive.
    $$ \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} $$

*   **False Negative Rate (FNR):** The proportion of actual positive cases that are incorrectly predicted as negative.
    $$ \text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} $$

Where:
*   TP (True Positives): Cases where the true label is 'Positive' and the prediction is 'Positive'.
*   TN (True Negatives): Cases where the true label is 'Negative' and the prediction is 'Negative'.
*   FP (False Positives): Cases where the true label is 'Negative' but the prediction is 'Positive' (Type I error).
*   FN (False Negatives): Cases where the true label is 'Positive' but the prediction is 'Negative' (Type II error).

<aside class="positive">
This function is crucial for evaluating the 'fit-for-purpose' aspect of the AI system, particularly in safety-critical domains like clinical decision support. Minimizing False Negatives, for example, could be a primary business objective.
</aside>

### 6.2 Initial Performance Calculation
Before diving into interactive analysis, a baseline is established. This section calculates and displays the initial performance metrics for both the AI-only system and the AI+Human system based on the first simulation run (using default UI/UX settings). You will see these metrics printed on the page.

### 6.3 Visualization: Comparative Performance Analysis (Aggregated Comparison)
To provide an intuitive and clear understanding of the impact of human oversight, we visualize the performance metrics using a bar chart. The `plot_performance_comparison_plotly` function compares Accuracy, FPR, and FNR between the AI-only and AI+Human systems.

```python
# From application_pages/page3.py

import plotly.graph_objects as go
import plotly.express as px
# ... other imports

def plot_performance_comparison_plotly(ai_only_metrics: Dict[str, Any], ai_human_metrics: Dict[str, Any]):
    """Generates a Plotly bar chart comparing AI-only vs. AI+Human system performance."""
    metrics_set = set(ai_only_metrics.keys()) | set(ai_human_metrics.keys())
    metrics = sorted(list(metrics_set))

    ai_only_values = [ai_only_metrics.get(m, 0.0) for m in metrics]
    ai_human_values = [ai_human_metrics.get(m, 0.0) for m in metrics]

    fig = go.Figure(data=[
        go.Bar(name='AI-Only', x=metrics, y=ai_only_values, marker_color='rgb(99,110,250)'),
        go.Bar(name='AI+Human', x=metrics, y=ai_human_values, marker_color='rgb(180,110,250)')
    ])
    fig.update_layout(
        barmode='group',
        title_text='Comparative System Performance (AI-Only vs. AI+Human)',
        xaxis_title='Performance Metric',
        yaxis_title='Metric Value',
        yaxis_range=[0, 1],
        font=dict(size=12),
        title_font_size=16,
        legend_title_text='System Type'
    )
    st.plotly_chart(fig)
```

This visual comparison quickly conveys whether human intervention is beneficial, particularly in mitigating critical errors like false negatives in safety-critical clinical decision support.

### 6.4 Visualization: Impact of UI/UX on Override Frequency (Relationship Plot)
Understanding when and why human operators choose to override AI suggestions is critical. The `plot_confidence_vs_override_plotly` function generates a scatter plot visualizing the relationship between the AI's confidence scores and the frequency of human overrides.

```python
# From application_pages/page3.py

def plot_confidence_vs_override_plotly(simulation_results_df: pd.DataFrame) -> None:
    """
    Generates a Plotly scatter plot to visualize the relationship between AI confidence scores
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

    if plot_df.empty:
        st.warning("No data to plot for AI Confidence vs. Human Override Frequency.")
        return

    fig = px.scatter(
        plot_df,
        x='confidence_midpoint',
        y='Override_Frequency',
        title='AI Confidence vs. Human Override Frequency',
        labels={'confidence_midpoint': 'AI Confidence (Probability of Positive Class)', 'Override_Frequency': 'Frequency of Human Override'},
        range_x=[0, 1],
        range_y=[-0.05, 1.05],
        color_discrete_sequence=px.colors.qualitative.Deep # deep color palette
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(font=dict(size=12), title_font_size=16)
    st.plotly_chart(fig)
```

This analysis provides valuable feedback for UI/UX designers, helping them understand if their designs effectively guide human attention to high-risk or low-confidence AI predictions.

### 6.5 Visualization: System Performance Over Time (Trend Plot)
In real-world operational settings, monitoring performance trends is crucial. The `plot_trend_metrics_plotly` function generates a line plot displaying the rolling average accuracy for both the AI-only and AI+Human systems over the sequence of simulated cases.

```python
# From application_pages/page3.py

def plot_trend_metrics_plotly(simulation_results_df: pd.DataFrame, window_size: int = 50) -> None:
    """
    Generates a Plotly line plot displaying rolling average performance metrics (e.g., accuracy) over the sequence of simulated decisions,
    comparing AI-only vs. AI+Human trends.
    """
    required_columns = ['true_diagnosis', 'ai_prediction', 'human_final_decision']
    if not all(col in simulation_results_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in simulation_results_df.columns]
        st.error(f"DataFrame is missing required columns for accuracy calculation: {missing_cols}")
        return
    if not isinstance(window_size, int) or window_size <= 0:
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
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['rolling_ai_accuracy'], mode='lines', name='AI-Only Rolling Accuracy', line=dict(color=px.colors.qualitative.Plotly[0])))
    fig.add_trace(go.Scatter(x=df.index, y=df['rolling_ai_human_accuracy'], mode='lines', name='AI+Human Rolling Accuracy', line=dict(color=px.colors.qualitative.Plotly[1])))
    
    fig.update_layout(
        title_text=f'Rolling Average Performance Trends (Window Size: {window_size})',
        xaxis_title='Simulation Step',
        yaxis_title='Rolling Accuracy',
        font=dict(size=12),
        title_font_size=16,
        legend_title_text='Metric'
    )
    st.plotly_chart(fig)
```

This visualization is vital for assessing the long-term stability and consistency of the human-in-the-loop system, identifying trends or fluctuations.

## 7. Interactive Experimentation and Dynamic Analysis (Page 3)
Duration: 00:10:00

On **"Page 3: Results & Analysis"**, you'll notice a section titled "Dynamic Simulation Results." This is where the interactive power of Streamlit comes into play.

### 7.1 Parameter Tuning in Page 2
<aside class="positive">
To further explore the dynamic interplay between human factors and UI/UX design, you need to navigate back to **"Page 2: Simulation Controls"**.
</aside>

There, you can adjust the interactive sliders and checkboxes:
*   `Show AI Explainability`
*   `Highlight Anomalies`
*   `Human Trust Threshold`
*   `Human Expertise Level`

Change these parameters and then return to **"Page 3: Results & Analysis"**.

### 7.2 Interactive Rerun: Rerunning Analysis with New Parameters
Streamlit's natural rerunning behavior automatically links the widget values from Page 2 to the analysis on Page 3. Whenever a parameter is changed on Page 2, a complete re-evaluation of the human-in-the-loop system is triggered on Page 3. This real-time feedback loop is invaluable for understanding cause-and-effect relationships.

You will see the "Updated Performance Metrics", "Updating Comparative Performance Plot...", "Updating AI Confidence vs. Human Override Plot...", and "Updating Performance Trend Plot..." sections dynamically refresh with the new results.

By dynamically modifying the human factors and UI/UX toggles, you can directly observe their impact on performance metrics and visualizations. This direct experimentation provides concrete evidence for how different design choices and human capabilities affect accuracy, false positive rates, false negative rates, and override patterns. This is a core component of evaluating uncertainty and model risk in real-world conditions and translating "fit-for-purpose" into measurable evidence.

<aside class="negative">
If you encounter a warning "No simulation results available," ensure you have first navigated to "Page 1: Data & AI Model" to generate data and train the AI model, and then to "Page 2: Simulation Controls" to run the initial simulation.
</aside>

## 8. Conclusion and Future Work
Duration: 00:05:00

This simulation has provided a practical demonstration of 'Human Oversight Roles' in AI-assisted decision-making. We've seen how human intervention can act as a crucial fallback mechanism, potentially enhancing system reliability and safety by overriding AI suggestions. The lab highlighted:

*   The quantifiable impact of human oversight on overall system performance metrics like accuracy, false positive rates, and false negative rates.
*   The significance of designing effective UI/UX, as features like AI explainability and anomaly highlighting can empower human operators to make more informed and accurate override decisions.
*   The interplay between AI confidence, human trust, and the quality of human intervention.

Ultimately, this exercise underscores the importance of a well-designed human-in-the-loop system to ensure "fit-for-purpose" AI in critical applications like clinical decision support, aligning with the principles of assurance foundations for critical ML.

### Next Steps & Productionization Notes:
*   <b>Real-world Data Integration</b>: While this simulation uses synthetic data, the next logical step would be to integrate real (anonymized) clinical data to validate the findings in a more realistic context.
*   <b>Advanced UI/UX Design</b>: Explore more sophisticated UI/UX elements, such as interactive dashboards that provide detailed explanations for AI predictions, to further empower human operators.
*   <b>Human Factors Engineering</b>: Conduct user studies with clinical professionals to gather feedback on the human-AI interaction, refine the override mechanisms, and optimize trust and workload.
*   <b>Continuous Monitoring</b>: Implement a continuous monitoring system for AI and human performance in live operation, tracking metrics and identifying potential drift or emerging risks.
*   <b>Ethical and Regulatory Compliance</b>: Ensure that the human-in-the-loop system adheres to relevant ethical guidelines and regulatory requirements for AI in healthcare.

## 9. References
*   [1] Unit 3: Assurance Foundations for Critical ML, 'Human Oversight Roles', [Case 2: Generative AI in Clinical Decision Support (Provided Resource)]. This section discusses roles for humans in the loop and the importance of designing UI/UX to help users understand model outputs.
