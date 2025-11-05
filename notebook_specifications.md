
# Technical Specification for Jupyter Notebook: Human-in-the-Loop Override Simulator

## 1. Notebook Overview

This Jupyter Notebook provides a practical simulation of 'Human Oversight Roles' within an AI-assisted decision-making system. Focusing on a synthetic clinical decision support scenario, users will act as human operators reviewing AI-generated diagnostic suggestions, making override decisions, and observing the resultant impact of human intervention on overall system safety and performance. This lab directly applies concepts from Unit 3: Assurance Foundations for Critical ML, particularly 'Human Oversight Roles' and the importance of effective UI/UX design.

#### Learning Goals
- Understand the key insights contained in the uploaded document and supporting data.
- Explain assurance-case methodology, evaluate uncertainty and model risk in real-world conditions, and translate "fit-for-purpose" into measurable evidence [1].
- Explore the importance of designing effective UI/UX to help users understand model outputs and catch anomalies, as mentioned in the provided resource [1].
- Understand 'Human Oversight Roles' and how human intervention impacts system safety and performance in critical applications.

#### Scope & Constraints
- The notebook is designed to execute end-to-end on a mid-spec laptop (8 GB RAM) in fewer than 5 minutes.
- Only open-source Python libraries available on PyPI are utilized.
- All major steps are accompanied by both code comments and brief narrative cells that describe **what** is happening and **why**.

## 2. Code Requirements

### List of Expected Libraries
-   `pandas` for data manipulation and analysis.
-   `numpy` for numerical operations.
-   `sklearn.model_selection` for splitting datasets (e.g., `train_test_split`).
-   `sklearn.ensemble.RandomForestClassifier` as the AI model for classification.
-   `sklearn.metrics` for performance evaluation (e.g., `accuracy_score`, `confusion_matrix`).
-   `matplotlib.pyplot` for generating static plots.
-   `seaborn` for enhanced data visualizations with an aesthetic appeal.
-   `ipywidgets` for creating interactive user controls (sliders, checkboxes).
-   `IPython.display` for managing widget display.

### List of Algorithms or Functions to be Implemented
1.  **`generate_synthetic_clinical_data(num_cases: int, seed: int) -> pd.DataFrame`**:
    *   Generates a DataFrame of `num_cases` synthetic patient records.
    *   Columns include: `case_id` (unique integer), `patient_age` (numeric, e.g., 20-80), `patient_gender` (categorical: 'Male', 'Female', 'Other'), `lab_result_A` (numeric, normally distributed, correlated with diagnosis), `lab_result_B` (numeric, uniform, correlated with diagnosis), `symptom_severity` (categorical: 'Mild', 'Moderate', 'Severe'), `previous_diagnosis` (categorical: 'Yes', 'No'), and `true_diagnosis` (binary: 'Positive', 'Negative').
    *   Ensures a realistic correlation between features and `true_diagnosis`.
2.  **`perform_data_validation(df: pd.DataFrame, critical_cols: list) -> None`**:
    *   Confirms expected column names and data types (e.g., `case_id` as int, `patient_age` as int, categorical as object).
    *   Asserts uniqueness of `case_id` (primary key).
    *   Asserts no missing values in `critical_cols` (e.g., `patient_age`, `lab_result_A`, `true_diagnosis`).
    *   Logs and prints summary statistics (`df.describe()`) for numeric columns.
3.  **`train_ai_model(features_df: pd.DataFrame, target_series: pd.Series, test_size: float = 0.3, random_state: int = 42) -> Tuple[sklearn.ensemble.RandomForestClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]`**:
    *   Performs one-hot encoding on categorical features in `features_df`.
    *   Splits the data into training and testing sets using `train_test_split`.
    *   Trains a `sklearn.ensemble.RandomForestClassifier` on the training data.
    *   Returns the trained model, training features/target, and test features/target.
4.  **`get_ai_predictions_and_confidence(model: sklearn.ensemble.RandomForestClassifier, test_features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`**:
    *   Generates AI diagnostic predictions (`model.predict()`) for the `test_features_df`.
    *   Extracts AI confidence scores (probability of the 'Positive' class, `model.predict_proba()`).
    *   Returns the predicted labels and confidence scores.
5.  **`simulate_human_decision(ai_prediction: str, ai_confidence: float, true_label: str, ui_explainability_enabled: bool, anomaly_highlighting_enabled: bool, human_trust_threshold: float, human_expertise_level: float) -> str`**:
    *   Simulates a human operator's decision.
    *   If `ai_confidence` is high (e.g., > `human_trust_threshold`) AND `anomaly_highlighting_enabled` is False, the human tends to accept `ai_prediction`.
    *   If `ai_confidence` is low OR `anomaly_highlighting_enabled` is True, the human's decision is more likely to be an override.
    *   If `ui_explainability_enabled` is True, the human has a higher probability of correcting an `ai_prediction` if it's initially wrong (influenced by `human_expertise_level`).
    *   Returns the final `human_final_decision` ('Positive' or 'Negative').
6.  **`calculate_performance_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, float]`**:
    *   Calculates `accuracy_score`, `false_positive_rate`, and `false_negative_rate`.
    *   `false_positive_rate` is calculated from the confusion matrix where 'Negative' is the true negative class and 'Positive' is the true positive class for medical context.
    *   Returns a dictionary of these metrics.
7.  **`run_full_simulation(data: pd.DataFrame, ai_model: sklearn.ensemble.RandomForestClassifier, encoded_features: pd.DataFrame, true_labels: pd.Series, ui_explainability_enabled: bool, anomaly_highlighting_enabled: bool, human_trust_threshold: float, human_expertise_level: float) -> pd.DataFrame`**:
    *   Orchestrates the simulation loop for all test cases.
    *   Calls `get_ai_predictions_and_confidence` and `simulate_human_decision` for each case.
    *   Returns a DataFrame containing `true_diagnosis`, `ai_prediction`, `ai_confidence`, and `human_final_decision` for each case.
8.  **`plot_performance_comparison(ai_only_metrics: Dict, ai_human_metrics: Dict) -> None`**:
    *   Generates a bar chart comparing AI-only vs. AI+Human performance metrics (Accuracy, FPR, FNR).
    *   Uses a color-blind-friendly palette.
    *   Includes clear titles, labeled axes, and legends.
    *   Saves a static PNG fallback image.
9.  **`plot_confidence_vs_override(simulation_results_df: pd.DataFrame) -> None`**:
    *   Generates a scatter plot showing AI confidence scores against the frequency or count of human overrides.
    *   May involve binning confidence scores for clearer visualization of override trends.
    *   Uses a color-blind-friendly palette.
    *   Includes clear titles, labeled axes, and legends.
    *   Saves a static PNG fallback image.
10. **`plot_trend_metrics(simulation_results_df: pd.DataFrame) -> None`**:
    *   Generates a line plot showing cumulative or rolling average metrics (e.g., accuracy or FNR) over the sequence of simulated decisions.
    *   Compares AI-only vs. AI+Human trends.
    *   Uses a color-blind-friendly palette.
    *   Includes clear titles, labeled axes, and legends.
    *   Saves a static PNG fallback image.

### Visualization Like Charts, Tables, Plots
1.  **DataFrame Display**: `df.head()`, `df.info()`, `df.describe()` for data exploration and validation.
2.  **Aggregated Comparison (Bar Chart)**: Compares Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR) for 'AI-Only' vs. 'AI+Human' systems.
    *   **Style**: `seaborn.barplot`, color-blind-friendly palette, font size $\geq$ 12pt, clear title ("Comparative System Performance"), labeled axes ("Metric", "Value"), legend ("System Type").
3.  **Relationship Plot (Scatter Plot)**: Displays `AI Confidence` vs. `Override Frequency` or `Override Count`.
    *   **Style**: `seaborn.scatterplot` (potentially with `hue` for override outcome or `size` for frequency), color-blind-friendly palette, font size $\geq$ 12pt, clear title ("AI Confidence vs. Human Override Tendency"), labeled axes ("AI Confidence", "Override Likelihood"), legend if `hue` is used.
4.  **Trend Plot (Line Plot)**: Shows `Cumulative Accuracy` or `Rolling FNR` over `Simulated Case Index` (time).
    *   **Style**: `seaborn.lineplot`, color-blind-friendly palette, font size $\geq$ 12pt, clear title ("System Performance Over Simulated Time"), labeled axes ("Simulated Case Index", "Performance Metric"), legend ("System Type").

## 3. Notebook Sections (in detail)

### 1. Notebook Overview
*   **Markdown Cell**:
    ```markdown
    # Human-in-the-Loop Override Simulator: Clinical Decision Support

    This notebook provides a practical simulation of 'Human Oversight Roles' within an AI-assisted decision-making system. Focusing on a synthetic clinical decision support scenario, users will act as human operators reviewing AI-generated diagnostic suggestions, making override decisions, and observing the resultant impact of human intervention on overall system safety and performance. This lab directly applies concepts from Unit 3: Assurance Foundations for Critical ML, particularly 'Human Oversight Roles' and the importance of effective UI/UX design.

    #### Learning Outcomes
    - Understand the key insights contained in the uploaded document and supporting data.
    - Explain assurance-case methodology, evaluate uncertainty and model risk in real-world conditions, and translate "fit-for-purpose" into measurable evidence [1].
    - Explore the importance of designing effective UI/UX to help users understand model outputs and catch anomalies, as mentioned in the provided resource [1].
    - Understand 'Human Oversight Roles' and how human intervention impacts system safety and performance in critical applications.

    #### Scope & Constraints
    - The lab must execute end-to-end on a mid-spec laptop (8 GB RAM) in fewer than 5 minutes.
    - Only open-source Python libraries from PyPI may be used.
    - All major steps include both code comments and brief narrative cells describing **what** is happening and **why**.
    ```

### 2. Library Imports
*   **Markdown Cell**:
    ```markdown
    We begin by importing all necessary Python libraries. These include `pandas` for data manipulation, `numpy` for numerical operations, `sklearn` for machine learning tasks and metrics, `matplotlib` and `seaborn` for data visualization, and `ipywidgets` for interactive elements.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import warnings
    warnings.filterwarnings('ignore') # Suppress warnings for cleaner output
    ```
*   **Code Cell (Execution)**:
    ```python
    # The import statements are executed directly above, no separate execution cell needed.
    # This cell serves as a placeholder to indicate completion of library loading.
    print("All required libraries have been successfully imported.")
    ```
*   **Markdown Cell**:
    ```markdown
    The required libraries have been successfully imported, setting up our environment for data generation, model simulation, and analysis.
    ```

### 3. Utility Functions for Synthetic Data Generation
*   **Markdown Cell**:
    ```markdown
    To simulate clinical decision-making, we need a synthetic dataset representing various patient cases. This dataset will include numeric features (e.g., age, lab results) and categorical features (e.g., gender, symptom severity). A key component will be a 'true diagnosis' label, which our simulated AI will attempt to predict.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def generate_synthetic_clinical_data(num_cases: int = 500, seed: int = 42) -> pd.DataFrame:
        """
        Generates a synthetic dataset for clinical decision support simulation.

        Args:
            num_cases (int): The number of synthetic patient cases to generate.
            seed (int): Random seed for reproducibility.

        Returns:
            pd.DataFrame: A DataFrame containing synthetic clinical case data.
        """
        np.random.seed(seed)

        # Generate patient demographics
        case_ids = np.arange(1, num_cases + 1)
        patient_ages = np.random.randint(20, 80, num_cases)
        patient_genders = np.random.choice(['Male', 'Female', 'Other'], num_cases, p=[0.48, 0.48, 0.04])

        # Generate lab results, with some correlation to diagnosis
        # Assume 'Positive' diagnosis is more likely with higher lab_result_A and lower lab_result_B
        lab_result_A = np.random.normal(loc=50, scale=15, size=num_cases)
        lab_result_B = np.random.uniform(low=0.5, high=10.0, size=num_cases)

        # Generate symptom severity
        symptom_severity = np.random.choice(['Mild', 'Moderate', 'Severe'], num_cases, p=[0.4, 0.4, 0.2])

        # Generate previous diagnosis history
        previous_diagnosis = np.random.choice(['Yes', 'No'], num_cases, p=[0.3, 0.7])

        # Generate true diagnosis based on a combination of factors
        # Make it moderately challenging for the AI
        true_diagnosis_numeric = (
            (patient_ages > 55) * 0.3 +
            (lab_result_A > 60) * 0.4 +
            (lab_result_B < 5) * 0.2 +
            (symptom_severity == 'Severe') * 0.2 +
            (previous_diagnosis == 'Yes') * 0.1 +
            np.random.normal(0, 0.1, num_cases) # Add some noise
        )
        true_diagnosis = np.where(true_diagnosis_numeric > np.percentile(true_diagnosis_numeric, 60), 'Positive', 'Negative')

        df = pd.DataFrame({
            'case_id': case_ids,
            'patient_age': patient_ages,
            'patient_gender': patient_genders,
            'lab_result_A': lab_result_A,
            'lab_result_B': lab_result_B,
            'symptom_severity': symptom_severity,
            'previous_diagnosis': previous_diagnosis,
            'true_diagnosis': true_diagnosis
        })

        return df
    ```
*   **Code Cell (Execution)**:
    ```python
    # Execution cell to define the data generation function.
    print("The `generate_synthetic_clinical_data` function has been defined.")
    ```
*   **Markdown Cell**:
    ```markdown
    The `generate_synthetic_clinical_data` function is now defined, ready to create our simulated patient records.
    ```

### 4. Generate Synthetic Clinical Case Data
*   **Markdown Cell**:
    ```markdown
    Using the utility function, we will generate a dataset of 500 synthetic clinical cases. This dataset will serve as the foundation for our AI model and human override simulation. The `true_diagnosis` column will be binary, representing the presence or absence of a specific condition.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    clinical_data = generate_synthetic_clinical_data(num_cases=500, seed=42)

    print(f"Generated a dataset of {len(clinical_data)} synthetic clinical cases.")
    print("\nFirst 5 rows of the dataset:")
    display(clinical_data.head())

    print("\nDataset Information:")
    clinical_data.info()
    ```
*   **Code Cell (Execution)**:
    ```python
    # The code to generate data and display info is executed directly above.
    print("Synthetic clinical case data has been generated and previewed.")
    ```
*   **Markdown Cell**:
    ```markdown
    A synthetic dataset of 500 clinical cases has been generated. We can observe the mix of numeric and categorical features, along with the `true_diagnosis` target variable. This dataset is designed to be lightweight (less than 5 MB) for efficient execution.
    ```

### 5. Data Validation and Initial Exploration
*   **Markdown Cell**:
    ```markdown
    Before proceeding, it's crucial to validate the generated data to ensure its quality and conformity with expectations. We will check for correct column names and data types, verify the uniqueness of the `case_id` primary key, assert no missing values in critical fields, and log summary statistics for numeric columns.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def perform_data_validation(df: pd.DataFrame, critical_cols: list) -> None:
        """
        Performs validation checks on the clinical data.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            critical_cols (list): A list of column names that must not have missing values.
        """
        print("--- Data Validation Report ---")

        # 1. Check expected column names and types
        expected_cols = {
            'case_id': np.int64,
            'patient_age': np.int64,
            'patient_gender': object,
            'lab_result_A': np.float64,
            'lab_result_B': np.float64,
            'symptom_severity': object,
            'previous_diagnosis': object,
            'true_diagnosis': object
        }
        for col, dtype in expected_cols.items():
            if col not in df.columns:
                print(f"ERROR: Missing expected column '{col}'.")
            elif not np.issubdtype(df[col].dtype, dtype):
                 print(f"WARNING: Column '{col}' has unexpected dtype {df[col].dtype}, expected {dtype}.")
            else:
                print(f"SUCCESS: Column '{col}' present with expected dtype {df[col].dtype}.")

        # 2. Check primary key uniqueness
        if not df['case_id'].is_unique:
            print("ERROR: 'case_id' column is not unique. Duplicate primary keys found.")
        else:
            print("SUCCESS: 'case_id' column is unique (primary key validated).")

        # 3. Assert no missing values in critical fields
        missing_critical = df[critical_cols].isnull().sum()
        if missing_critical.sum() > 0:
            print("ERROR: Missing values found in critical fields:")
            print(missing_critical[missing_critical > 0])
        else:
            print(f"SUCCESS: No missing values in critical fields: {', '.join(critical_cols)}.")

        # 4. Log summary statistics for numeric columns
        print("\nSummary Statistics for Numeric Columns:")
        display(df.select_dtypes(include=np.number).describe())

        print("--- End of Validation Report ---")
    ```
*   **Code Cell (Execution)**:
    ```python
    critical_fields = ['patient_age', 'lab_result_A', 'true_diagnosis']
    perform_data_validation(clinical_data, critical_fields)
    ```
*   **Markdown Cell**:
    ```markdown
    The data validation step confirms that our synthetic dataset meets the expected structural and quality standards. This ensures a reliable foundation for the subsequent AI model training and simulation.
    ```

### 6. AI Model Training (Simulated)
*   **Markdown Cell**:
    ```markdown
    Our simulated AI assistant will be represented by a classification model trained on the synthetic clinical data. This AI's role is to suggest a diagnosis based on patient features. For this purpose, we will use a `RandomForestClassifier` from `sklearn.ensemble`, a robust ensemble learning method suitable for varied data types. The goal is to simulate an AI that makes predictions with varying levels of confidence.

    We first split the data into training and testing sets to evaluate the AI's performance on unseen data.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def train_ai_model(df: pd.DataFrame, target_col: str = 'true_diagnosis', random_state: int = 42) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer]:
        """
        Preprocesses data, splits into train/test, and trains a RandomForestClassifier.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_col (str): Name of the target column.
            random_state (int): Random seed for reproducibility.

        Returns:
            Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer]:
            Trained model, processed training features, training target,
            processed test features, test target, and the preprocessor.
        """
        features = df.drop(columns=[target_col, 'case_id'])
        target = df[target_col]

        # Identify categorical and numerical columns
        categorical_features = features.select_dtypes(include='object').columns
        numerical_features = features.select_dtypes(include=np.number).columns

        # Create a preprocessor using OneHotEncoder for categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough' # Keep numerical features as is
        )

        # Apply preprocessing
        X_processed = preprocessor.fit_transform(features)
        
        # Get feature names after one-hot encoding for better interpretability
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        processed_feature_names = list(ohe_feature_names) + list(numerical_features)
        X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=features.index)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_processed_df, target, test_size=0.3, random_state=random_state, stratify=target)

        # Train RandomForestClassifier
        ai_model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        ai_model.fit(X_train, y_train)

        print("RandomForestClassifier trained successfully.")
        return ai_model, X_train, y_train, X_test, y_test, preprocessor
    ```
*   **Code Cell (Execution)**:
    ```python
    ai_model, X_train, y_train, X_test, y_test, preprocessor = train_ai_model(clinical_data, target_col='true_diagnosis')

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    ```
*   **Markdown Cell**:
    ```markdown
    A `RandomForestClassifier` has been trained on the synthetic clinical data. This model will now act as our AI assistant, providing diagnostic suggestions and associated confidence scores for new cases.
    ```

### 7. Generate AI Predictions and Confidence Scores
*   **Markdown Cell**:
    ```markdown
    After training, the AI model will generate predictions for the test set. Crucially, it will also output 'confidence scores' – the probability it assigns to each diagnosis. These scores are vital in a human-in-the-loop system, as they inform human operators about the AI's certainty, influencing override decisions [1].

    The confidence score $C$ for a predicted class is typically given by the predicted probability:
    $$ C = P(\text{predicted class} | \text{input features}) $$
    For a binary classification problem (e.g., 'Positive' or 'Negative' diagnosis), if the model predicts 'Positive', its confidence is $P(\text{Positive} | \text{features})$. If it predicts 'Negative', its confidence is $P(\text{Negative} | \text{features})$. We will use the probability of the 'Positive' class as the primary confidence indicator.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def get_ai_predictions_and_confidence(model: RandomForestClassifier, test_features_df: pd.DataFrame, positive_class: str = 'Positive') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates AI predictions and extracts confidence scores.

        Args:
            model (RandomForestClassifier): The trained AI model.
            test_features_df (pd.DataFrame): The processed test features.
            positive_class (str): The label of the positive class.

        Returns:
            Tuple[np.ndarray, np.ndarray]: AI predicted labels and confidence scores.
        """
        ai_predictions = model.predict(test_features_df)
        ai_probabilities = model.predict_proba(test_features_df)

        # Get the index of the positive class
        positive_class_idx = np.where(model.classes_ == positive_class)[0][0]
        ai_confidence = ai_probabilities[:, positive_class_idx]

        return ai_predictions, ai_confidence
    ```
*   **Code Cell (Execution)**:
    ```python
    ai_predictions, ai_confidence = get_ai_predictions_and_confidence(ai_model, X_test)

    # Combine AI results with true labels for simulation
    simulation_base_df = pd.DataFrame({
        'true_diagnosis': y_test.values,
        'ai_prediction': ai_predictions,
        'ai_confidence': ai_confidence
    }, index=y_test.index)

    print("AI predictions and confidence scores generated.")
    print("\nSample of AI predictions and confidence:")
    display(simulation_base_df.head())
    ```
*   **Markdown Cell**:
    ```markdown
    The AI model has generated its diagnostic predictions and corresponding confidence scores for all test cases. These scores provide an estimate of the model's certainty, which will be a key factor in our human override simulation.
    ```

### 8. Human Override Mechanism (Setup)
*   **Markdown Cell**:
    ```markdown
    The core of this simulation is the 'Human Override Mechanism'. In real-world critical systems, humans act as fallback triggers and provide manual overrides when AI suggestions are deemed unsafe or incorrect. This function simulates a human operator's decision process, considering the AI's confidence, the presence of UI/UX features, and the human's intrinsic trust threshold and expertise level.

    The human decision logic can be simplified as follows:
    - If AI confidence is very high AND no anomalies are highlighted, the human accepts the AI's suggestion.
    - If AI confidence is low OR anomalies are highlighted, the human might override based on a `human_trust_threshold` and their `human_expertise_level`.
    - If `ui_explainability_enabled` is true, the human is more likely to correctly override an incorrect AI prediction.

    This function introduces a simulated human "error rate" or "override quality" dependent on the human's expertise and the clarity of the UI.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def simulate_human_decision(ai_prediction: str, ai_confidence: float, true_label: str,
                                ui_explainability_enabled: bool, anomaly_highlighting_enabled: bool,
                                human_trust_threshold: float, human_expertise_level: float,
                                positive_class: str = 'Positive') -> str:
        """
        Simulates a human's decision based on AI output and UI/UX factors.

        Args:
            ai_prediction (str): The AI's diagnostic prediction.
            ai_confidence (float): The AI's confidence in its prediction (probability of positive class).
            true_label (str): The actual correct diagnosis.
            ui_explainability_enabled (bool): Whether AI explainability features are enabled.
            anomaly_highlighting_enabled (bool): Whether anomaly highlighting is enabled.
            human_trust_threshold (float): The confidence threshold below which humans are more likely to scrutinize.
            human_expertise_level (float): A measure of human skill in correcting AI errors (0.0 to 1.0).
            positive_class (str): The label for the positive class.

        Returns:
            str: The human's final diagnostic decision.
        """
        human_decision = ai_prediction # Start by assuming human accepts AI

        # AI Confidence considered low if below threshold for its *predicted* class
        # If AI predicted 'Negative', confidence of 'Negative' is 1 - ai_confidence (of 'Positive')
        ai_predicted_class_confidence = ai_confidence if ai_prediction == positive_class else (1 - ai_confidence)
        
        low_confidence_flag = (ai_predicted_class_confidence < human_trust_threshold)
        anomaly_flag = anomaly_highlighting_enabled and (
            (ai_prediction == positive_class and ai_confidence < 0.3) or
            (ai_prediction != positive_class and ai_confidence > 0.7) # AI is confident in Negative, but Positive prob is high
        ) # Simplified anomaly: AI is confident but perhaps in the 'wrong' way or is uncertain

        # Human is more likely to override if confidence is low or anomaly is flagged
        if low_confidence_flag or anomaly_flag:
            # Human scrutiny increases
            
            # The chance of human overriding *correctly* depends on expertise and UI/UX
            # This simulates human intelligence and ability to spot errors
            if ai_prediction != true_label: # If AI made a mistake
                override_success_chance = human_expertise_level * (0.6 if ui_explainability_enabled else 0.3)
                if np.random.rand() < override_success_chance:
                    human_decision = true_label # Human correctly overrides
                else:
                    human_decision = ai_prediction # Human fails to correct or makes another error
            else: # If AI was correct, human might still override incorrectly if too suspicious
                override_error_chance = (1 - human_expertise_level) * (0.2 if not ui_explainability_enabled else 0.1)
                if np.random.rand() < override_error_chance:
                    human_decision = 'Positive' if ai_prediction == 'Negative' else 'Negative' # Human makes an incorrect override
        
        return human_decision
    ```
*   **Code Cell (Execution)**:
    ```python
    # Execution cell to define the human decision simulation function.
    print("The `simulate_human_decision` function has been defined, encapsulating human override logic.")
    ```
*   **Markdown Cell**:
    ```markdown
    The `simulate_human_decision` function is now defined, embodying the logic for how human operators interact with AI suggestions, influenced by various factors. This forms the basis for our human-in-the-loop simulation.
    ```

### 9. User Interaction Controls (UI/UX Toggle)
*   **Markdown Cell**:
    ```markdown
    Effective UI/UX design is crucial for human operators to understand AI outputs and catch anomalies, enhancing overall system safety [1]. We will introduce interactive controls using `ipywidgets` to allow users to toggle different UI/UX features. These toggles will influence how the `simulate_human_decision` function behaves, demonstrating their impact.

    - **Show AI Explainability**: Simulates providing explanations for AI predictions, potentially increasing human understanding and override accuracy.
    - **Highlight Anomalies**: Simulates flagging cases where AI confidence might be misleading or features are unusual, prompting human scrutiny.
    - **Human Trust Threshold**: Represents the confidence level above which a human is more likely to trust the AI's prediction without deep scrutiny ($0 \leq \text{Threshold} \leq 1$).
    - **Human Expertise Level**: Represents the inherent skill of the human operator in identifying and correcting AI errors ($0 \leq \text{Level} \leq 1$).
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    # Placeholder for ipywidgets setup. These widgets will be defined and displayed in a later interactive cell.
    # This cell just ensures the variables are known.
    ui_explainability_toggle = widgets.Checkbox(
        value=True,
        description='Show AI Explainability',
        disabled=False,
        indent=False,
        tooltip='When enabled, human operators are better informed and more likely to make correct overrides.'
    )

    anomaly_highlighting_toggle = widgets.Checkbox(
        value=True,
        description='Highlight Anomalies',
        disabled=False,
        indent=False,
        tooltip='When enabled, potential inconsistencies or low-confidence predictions are flagged, prompting human scrutiny.'
    )

    human_trust_slider = widgets.FloatSlider(
        value=0.7,
        min=0.0,
        max=1.0,
        step=0.05,
        description='Human Trust Threshold:',
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        tooltip='AI confidence score below this threshold makes human more likely to scrutinize/override.'
    )

    human_expertise_slider = widgets.FloatSlider(
        value=0.6,
        min=0.0,
        max=1.0,
        step=0.05,
        description='Human Expertise Level:',
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        tooltip='Higher level means human is better at correcting AI errors when they decide to override.'
    )

    print("Interactive widget definitions prepared.")
    ```
*   **Code Cell (Execution)**:
    ```python
    # The interactive widgets will be displayed in section 16, linked to the re-run function.
    print("UI/UX controls are defined and will be displayed in a subsequent section.")
    ```
*   **Markdown Cell**:
    ```markdown
    Interactive controls for UI/UX features and human parameters are now defined. These will be displayed in a later section to allow dynamic interaction.
    ```

### 10. Simulate Clinical Case Review and Override Decisions
*   **Markdown Cell**:
    ```markdown
    Now we simulate the sequential review of clinical cases. For each case, the AI provides a prediction and confidence. The human operator then makes a decision (accept or override) based on the AI's output and the current UI/UX settings and their personal parameters. We will record both the AI's original decision and the human's final decision for each case.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def run_full_simulation(
        simulation_df: pd.DataFrame,
        ui_explainability_enabled: bool,
        anomaly_highlighting_enabled: bool,
        human_trust_threshold: float,
        human_expertise_level: float
    ) -> pd.DataFrame:
        """
        Runs the full human-in-the-loop simulation over all test cases.

        Args:
            simulation_df (pd.DataFrame): DataFrame with true_diagnosis, ai_prediction, ai_confidence.
            ui_explainability_enabled (bool): Whether AI explainability is on.
            anomaly_highlighting_enabled (bool): Whether anomaly highlighting is on.
            human_trust_threshold (float): Human's trust threshold for AI confidence.
            human_expertise_level (float): Human's skill level.

        Returns:
            pd.DataFrame: Original simulation_df with added 'human_final_decision' column.
        """
        human_decisions = []
        for idx, row in simulation_df.iterrows():
            human_decision = simulate_human_decision(
                ai_prediction=row['ai_prediction'],
                ai_confidence=row['ai_confidence'],
                true_label=row['true_diagnosis'],
                ui_explainability_enabled=ui_explainability_enabled,
                anomaly_highlighting_enabled=anomaly_highlighting_enabled,
                human_trust_threshold=human_trust_threshold,
                human_expertise_level=human_expertise_level
            )
            human_decisions.append(human_decision)
        
        simulation_df_results = simulation_df.copy()
        simulation_df_results['human_final_decision'] = human_decisions
        return simulation_df_results
    ```
*   **Code Cell (Execution)**:
    ```python
    # For initial run, use default widget values
    initial_simulation_results_df = run_full_simulation(
        simulation_df=simulation_base_df,
        ui_explainability_enabled=ui_explainability_toggle.value,
        anomaly_highlighting_enabled=anomaly_highlighting_toggle.value,
        human_trust_threshold=human_trust_slider.value,
        human_expertise_level=human_expertise_slider.value
    )

    print("Initial clinical case review simulation completed.")
    print("\nSample of simulation results (AI vs Human decision):")
    display(initial_simulation_results_df.head())
    ```
*   **Markdown Cell**:
    ```markdown
    The initial clinical case review simulation has been executed. We now have a record of both the AI's initial suggestions and the human operator's final decisions for each case, incorporating the effects of UI/UX features and human parameters set to their default values.
    ```

### 11. Performance Metrics Calculation
*   **Markdown Cell**:
    ```markdown
    To assess the effectiveness of human oversight, we need to compare the performance of the AI-only system with the combined AI+Human system. Key metrics for safety-critical applications include:

    - **Accuracy**: The proportion of correct predictions.
    - **False Positive Rate (FPR)**: The proportion of healthy individuals incorrectly diagnosed as having the condition (Type I error). For binary classification, where 'Positive' is the target condition:
      $$ \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} $$
    - **False Negative Rate (FNR)**: The proportion of individuals with the condition incorrectly diagnosed as healthy (Type II error).
      $$ \text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} $$
    Where FP is False Positives, TN is True Negatives, FN is False Negatives, and TP is True Positives. A low FNR is often critical in clinical diagnosis to avoid missing actual cases.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def calculate_performance_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray, positive_class: str = 'Positive') -> Dict[str, float]:
        """
        Calculates accuracy, false positive rate, and false negative rate.

        Args:
            true_labels (np.ndarray): Array of true labels.
            predicted_labels (np.ndarray): Array of predicted labels.
            positive_class (str): The label of the positive class.

        Returns:
            Dict[str, float]: A dictionary containing accuracy, FPR, and FNR.
        """
        acc = accuracy_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels, labels=[positive_class, 'Negative'])

        # cm is [[TP, FN], [FP, TN]] where rows are true, cols are predicted for labels=[Positive, Negative]
        TP, FN, FP, TN = cm.ravel()

        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

        return {
            'Accuracy': acc,
            'False Positive Rate': fpr,
            'False Negative Rate': fnr
        }
    ```
*   **Code Cell (Execution)**:
    ```python
    # Execution cell to define the performance calculation function.
    print("The `calculate_performance_metrics` function has been defined for evaluating system performance.")
    ```
*   **Markdown Cell**:
    ```markdown
    The function to calculate accuracy, false positive rate, and false negative rate has been defined. This will allow us to quantitatively assess the performance of both the AI-only and AI+Human systems.
    ```

### 12. Calculate Initial Performance
*   **Markdown Cell**:
    ```markdown
    Using the defined metrics function, we now calculate the initial performance metrics for both the AI-only system and the AI+Human system based on the initial simulation run (with default UI/UX settings). This provides a baseline for comparison.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    ai_only_initial_metrics = calculate_performance_metrics(
        initial_simulation_results_df['true_diagnosis'],
        initial_simulation_results_df['ai_prediction']
    )

    ai_human_initial_metrics = calculate_performance_metrics(
        initial_simulation_results_df['true_diagnosis'],
        initial_simulation_results_df['human_final_decision']
    )

    print("--- Initial Performance Metrics ---")
    print("AI-Only System Performance:")
    for metric, value in ai_only_initial_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nAI+Human System Performance (with default settings):")
    for metric, value in ai_human_initial_metrics.items():
        print(f"  {metric}: {value:.4f}")
    ```
*   **Code Cell (Execution)**:
    ```python
    # The metrics calculation and printing are executed directly above.
    print("\nInitial performance metrics for both AI-only and AI+Human systems have been computed.")
    ```
*   **Markdown Cell**:
    ```markdown
    The initial performance metrics for both AI-only and AI+Human systems have been calculated and displayed. These numbers provide a quantitative snapshot of how human intervention (with default UI/UX settings) impacts system performance.
    ```

### 13. Comparative Performance Analysis (Aggregated Comparison)
*   **Markdown Cell**:
    ```markdown
    To visually compare the AI-only system against the AI+Human system, we will generate a bar chart. This visualization will highlight how human oversight influences key performance metrics like accuracy, false positive rate, and false negative rate, especially in the context of safety-critical decisions. A color-blind-friendly palette will be used, and axis labels will be clear.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def plot_performance_comparison(ai_only_metrics: Dict, ai_human_metrics: Dict) -> None:
        """
        Generates a bar chart comparing AI-only vs. AI+Human system performance.

        Args:
            ai_only_metrics (Dict): Dictionary of performance metrics for AI-only.
            ai_human_metrics (Dict): Dictionary of performance metrics for AI+Human.
        """
        metrics_df = pd.DataFrame({
            'Metric': list(ai_only_metrics.keys()) * 2,
            'Value': list(ai_only_metrics.values()) + list(ai_human_metrics.values()),
            'System': ['AI-Only'] * len(ai_only_metrics) + ['AI+Human'] * len(ai_human_metrics)
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Value', hue='System', data=metrics_df, palette='viridis') # viridis is generally colorblind-friendly
        plt.title('Comparative System Performance (AI-Only vs. AI+Human)', fontsize=16)
        plt.xlabel('Performance Metric', fontsize=14)
        plt.ylabel('Metric Value', fontsize=14)
        plt.ylim(0, 1) # Metrics are usually between 0 and 1
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='System Type', fontsize=12, title_fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300) # Static fallback
        plt.show()
    ```
*   **Code Cell (Execution)**:
    ```python
    plot_performance_comparison(ai_only_initial_metrics, ai_human_initial_metrics)
    ```
*   **Markdown Cell**:
    ```markdown
    The bar chart clearly illustrates the differences in accuracy, false positive rate, and false negative rate between the AI-only system and the system incorporating human oversight. This visual comparison provides immediate insights into the benefits or trade-offs of human intervention, showing how human oversight can reduce critical errors like false negatives in this simulation.
    ```

### 14. Impact of UI/UX on Override Frequency (Relationship Plot)
*   **Markdown Cell**:
    ```markdown
    Understanding the relationship between AI confidence and human override frequency is crucial for designing effective interfaces. A scatter plot will show how often humans override AI suggestions across different confidence levels. This helps reveal if humans primarily intervene when the AI is less confident or if other factors (like perceived anomalies influenced by UI/UX) play a role.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def plot_confidence_vs_override(simulation_results_df: pd.DataFrame) -> None:
        """
        Generates a scatter plot showing AI confidence vs. human override frequency.

        Args:
            simulation_results_df (pd.DataFrame): DataFrame with simulation results.
        """
        plot_df = simulation_results_df.copy()
        plot_df['override_decision'] = (plot_df['ai_prediction'] != plot_df['human_final_decision'])

        # Bin AI confidence to show frequency
        plot_df['confidence_bin'] = pd.cut(plot_df['ai_confidence'], bins=np.arange(0, 1.01, 0.1), right=False, include_lowest=True, labels=False)
        
        # Calculate override frequency per bin
        override_summary = plot_df.groupby('confidence_bin')['override_decision'].value_counts(normalize=True).unstack(fill_value=0)
        override_summary = override_summary.rename(columns={True: 'Override_Frequency', False: 'Accept_Frequency'})
        override_summary['Mid_Confidence'] = override_summary.index * 0.1 + 0.05 # Mid-point of bin

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Mid_Confidence', y='Override_Frequency', data=override_summary, s=150, color='#1b9e77', alpha=0.8) # A colorblind-friendly green
        plt.title('AI Confidence vs. Human Override Frequency', fontsize=16)
        plt.xlabel('AI Confidence (Probability of Positive Class)', fontsize=14)
        plt.ylabel('Frequency of Human Override', fontsize=14)
        plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('confidence_vs_override.png', dpi=300) # Static fallback
        plt.show()
    ```
*   **Code Cell (Execution)**:
    ```python
    plot_confidence_vs_override(initial_simulation_results_df)
    ```
*   **Markdown Cell**:
    ```markdown
    The scatter plot visualizes the interplay between AI confidence and human override decisions. This helps us understand if human operators are effectively targeting low-confidence AI suggestions or if other UI/UX cues (like anomaly highlighting) drive their intervention patterns, leading to more overrides even at moderate AI confidence.
    ```

### 15. System Performance Over Time (Trend Plot)
*   **Markdown Cell**:
    ```markdown
    In a continuous decision-making process, it's important to monitor how system performance evolves over time. A line plot can show the cumulative or rolling accuracy/error rates as more clinical cases are processed. This mimics real-time operational monitoring and can highlight potential drift or changes in performance.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    def plot_trend_metrics(simulation_results_df: pd.DataFrame, window_size: int = 50) -> None:
        """
        Generates a line plot showing rolling performance metrics over time.

        Args:
            simulation_results_df (pd.DataFrame): DataFrame with simulation results.
            window_size (int): The window size for rolling calculations.
        """
        plot_df = simulation_results_df.copy()
        
        # Calculate rolling metrics
        plot_df['ai_only_correct'] = (plot_df['true_diagnosis'] == plot_df['ai_prediction']).astype(int)
        plot_df['ai_human_correct'] = (plot_df['true_diagnosis'] == plot_df['human_final_decision']).astype(int)

        plot_df['rolling_ai_accuracy'] = plot_df['ai_only_correct'].rolling(window=window_size).mean()
        plot_df['rolling_ai_human_accuracy'] = plot_df['ai_human_correct'].rolling(window=window_size).mean()

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=plot_df.index, y='rolling_ai_accuracy', data=plot_df, label='AI-Only Rolling Accuracy', color='#d95f02') # A colorblind-friendly orange
        sns.lineplot(x=plot_df.index, y='rolling_ai_human_accuracy', data=plot_df, label='AI+Human Rolling Accuracy', color='#7570b3') # A colorblind-friendly purple
        
        plt.title(f'Rolling Accuracy Over Simulated Cases (Window Size: {window_size})', fontsize=16)
        plt.xlabel('Simulated Case Index', fontsize=14)
        plt.ylabel('Rolling Accuracy', fontsize=14)
        plt.ylim(0.5, 1.0)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='System Type', fontsize=12, title_fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('performance_trend.png', dpi=300) # Static fallback
        plt.show()
    ```
*   **Code Cell (Execution)**:
    ```python
    plot_trend_metrics(initial_simulation_results_df, window_size=50)
    ```
*   **Markdown Cell**:
    ```markdown
    The trend plot displays how the rolling accuracy metrics change over the sequence of simulated clinical cases. This provides insights into the stability and consistency of both AI-only and AI+Human systems during continuous operation, demonstrating how human intervention can potentially smooth out performance fluctuations or maintain a higher baseline.
    ```

### 16. User Interaction for Parameter Tuning
*   **Markdown Cell**:
    ```markdown
    To further explore the impact of human factors and UI/UX design, we provide interactive sliders and checkboxes. These controls allow you to adjust critical parameters such as the human's trust in the AI, their expertise level, and whether UI/UX features like AI explainability or anomaly highlighting are enabled. By changing these, you can dynamically observe how they affect the simulation results.

    Each control includes inline help text to explain its purpose.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    # Widgets are already defined in section 9.
    # We will now display them for user interaction.
    print("Adjust the parameters below to rerun the simulation and observe their impact:")
    display(ui_explainability_toggle, anomaly_highlighting_toggle, human_trust_slider, human_expertise_slider)
    ```
*   **Code Cell (Execution)**:
    ```python
    # The widgets are displayed directly above.
    print("Interactive parameter tuning controls are now displayed.")
    ```
*   **Markdown Cell**:
    ```markdown
    The interactive parameter tuning controls are now active. Experiment with different settings for human trust, expertise, and UI/UX features to dynamically observe their influence on the simulation results in the following re-analysis step.
    ```

### 17. Rerun Analysis with New Parameters
*   **Markdown Cell**:
    ```markdown
    This section allows you to re-run the entire simulation and analysis pipeline with the parameters adjusted using the interactive widgets above. By modifying the `human_trust_threshold`, `human_expertise_level`, and UI/UX toggles, you can directly observe their impact on performance metrics and visualizations.
    ```
*   **Code Cell (Function Implementation)**:
    ```python
    @widgets.interact(
        ui_explainability_enabled=ui_explainability_toggle,
        anomaly_highlighting_enabled=anomaly_highlighting_toggle,
        human_trust_threshold=human_trust_slider,
        human_expertise_level=human_expertise_slider
    )
    def rerun_analysis_with_new_params(
        ui_explainability_enabled: bool,
        anomaly_highlighting_enabled: bool,
        human_trust_threshold: float,
        human_expertise_level: float
    ):
        """
        Reruns the full simulation and analysis with updated parameters.
        """
        with clear_output(wait=True):
            print("Rerunning simulation and analysis with new parameters...")
            
            # Run simulation
            current_simulation_results_df = run_full_simulation(
                simulation_df=simulation_base_df,
                ui_explainability_enabled=ui_explainability_enabled,
                anomaly_highlighting_enabled=anomaly_highlighting_enabled,
                human_trust_threshold=human_trust_threshold,
                human_expertise_level=human_expertise_level
            )

            # Calculate metrics
            ai_only_metrics = calculate_performance_metrics(
                current_simulation_results_df['true_diagnosis'],
                current_simulation_results_df['ai_prediction']
            )
            ai_human_metrics = calculate_performance_metrics(
                current_simulation_results_df['true_diagnosis'],
                current_simulation_results_df['human_final_decision']
            )

            print("\n--- Updated Performance Metrics ---")
            print("AI-Only System Performance:")
            for metric, value in ai_only_metrics.items():
                print(f"  {metric}: {value:.4f}")

            print("\nAI+Human System Performance (with current settings):")
            for metric, value in ai_human_metrics.items():
                print(f"  {metric}: {value:.4f}")
            print("-----------------------------------")

            # Plot results
            print("\nUpdating Comparative Performance Plot...")
            plot_performance_comparison(ai_only_metrics, ai_human_metrics)
            
            print("\nUpdating AI Confidence vs. Human Override Plot...")
            plot_confidence_vs_override(current_simulation_results_df)
            
            print("\nUpdating Performance Trend Plot...")
            plot_trend_metrics(current_simulation_results_df)
            
            print("\nAnalysis complete with new parameters. Scroll up to see updated plots.")

    ```
*   **Code Cell (Execution)**:
    ```python
    # The `@widgets.interact` decorator above automatically creates the execution environment.
    # No separate execution cell is needed here.
    print("Interaction is now active. Change the values in the widgets above to see dynamic updates below.")
    ```
*   **Markdown Cell**:
    ```markdown
    The simulation and analysis have been re-run with the selected parameters. Observe how changing the UI/UX features or human parameters alters the comparative performance, override patterns, and performance trends. This interactivity demonstrates the sensitivity of human-in-the-loop systems to both technological design and human factors.
    ```

### 18. Conclusion and Key Takeaways
*   **Markdown Cell**:
    ```markdown
    This simulation has provided a practical demonstration of 'Human Oversight Roles' in AI-assisted decision-making. We've seen how human intervention can act as a crucial fallback mechanism, potentially enhancing system reliability and safety by overriding AI suggestions. The lab highlighted:

    - The quantifiable impact of human oversight on overall system performance metrics like accuracy, false positive rates, and false negative rates.
    - The significance of designing effective UI/UX, as features like AI explainability and anomaly highlighting can empower human operators to make more informed and accurate override decisions.
    - The interplay between AI confidence, human trust, and the quality of human intervention.

    Ultimately, this exercise underscores the importance of a well-designed human-in-the-loop system to ensure "fit-for-purpose" AI in critical applications like clinical decision support, aligning with the principles of assurance foundations for critical ML.
    ```

### 19. References
*   **Markdown Cell**:
    ```markdown
    ## References

    [1] Unit 3: Assurance Foundations for Critical ML, 'Human Oversight Roles', [Case 2: Generative AI in Clinical Decision Support (Provided Resource)]. This section discusses roles for humans in the loop and the importance of designing UI/UX to help users understand model outputs.
    ```
