import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def run_page1():
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

    EXPECTED_DTYPES = {
        'case_id': np.dtype('int64'),
        'patient_age': np.dtype('int64'),
        'patient_gender': np.dtype('object'),
        'lab_result_A': np.dtype('float64'),
        'lab_result_B': np.dtype('float64'),
        'symptom_severity': np.dtype('int64'), # Changed to int64 from object based on generation
        'previous_diagnosis': np.dtype('bool'), # Changed to bool from object based on generation
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
            st.success(f"SUCCESS: No missing values in critical fields: {critical_cols[0]}, {critical_cols[1]}, {critical_cols[2]}.")

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
    st.session_state.ai_model = ai_model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.preprocessor = preprocessor

    st.write(f"\nTraining set size: {len(X_train)}")
    st.write(f"Test set size: {len(X_test)}")

    st.markdown("""
    **Interpretation:**
    A `RandomForestClassifier` has been successfully trained on the preprocessed synthetic clinical data. The data was split into training and testing sets, ensuring that the model's performance can be evaluated on unseen data. The `n_estimators` parameter in the `RandomForestClassifier` was set to 100, indicating 100 decision trees are built, and `class_weight='balanced'` was used to address potential class imbalance. This trained model now acts as our AI assistant, ready to provide diagnostic suggestions and associated confidence scores for new cases. The sizes of the training and test sets are displayed, confirming the data split proportions.
    """)

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

    ai_predictions, ai_confidence = get_ai_predictions_and_confidence(st.session_state.ai_model, st.session_state.X_test, positive_class='Positive')

    simulation_base_df = pd.DataFrame({
        'true_diagnosis': st.session_state.y_test.values,
        'ai_prediction': ai_predictions,
        'ai_confidence': ai_confidence
    }, index=st.session_state.y_test.index)
    st.session_state.simulation_base_df = simulation_base_df

    st.write("AI predictions and confidence scores generated.")
    st.write("Sample of AI predictions and confidence:")
    st.dataframe(simulation_base_df.head())

    st.markdown("""
    **Interpretation:**
    The AI model has successfully generated its diagnostic predictions and corresponding confidence scores for all test cases. The `simulation_base_df` now contains the `true_diagnosis`, the `ai_prediction`, and the `ai_confidence` for each case. Observing the head of this DataFrame, we can see how the AI's prediction aligns with its confidence. These scores provide an estimate of the model's certainty, which will be a key factor in our human override simulation, allowing human operators to gauge the AI's reliability for each case.
    """)