import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple

def run_page2():
    st.title("Simulation Controls")

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


    st.subheader("5.6 User Interaction Controls: UI/UX Toggle and Human Factors Setup")
    st.markdown("""
    **Context & Business Value:**
    Effective UI/UX design is paramount for human operators to understand AI outputs and effectively catch anomalies, thereby enhancing overall system safety and performance. This section sets up interactive controls to allow users to dynamically adjust critical parameters. These controls directly influence the `simulate_human_decision` function, enabling a hands-on exploration of how UI/UX features and human factors impact the simulation results.

    -   **Show AI Explainability:** Simulates providing explanations for AI predictions, which can increase human understanding and improve override accuracy. This relates to the transparency aspect of trustworthy AI.
    -   **Highlight Anomalies:** Simulates flagging cases where AI confidence might be misleading or input features are unusual, prompting human scrutiny and reducing critical errors.
    -   **Human Trust Threshold:** Represents the confidence level (between 0 and 1) below which a human operator is more likely to scrutinize the AI's prediction rather than blindly accepting it. This models human psychology in interacting with automated systems.
    -   **Human Expertise Level:** Represents the inherent skill or knowledge of the human operator (between 0 and 1) in identifying and correcting AI errors when they choose to intervene.

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
    if 'simulation_base_df' in st.session_state:
        st.session_state.initial_simulation_results_df = run_full_simulation(
            simulation_df=st.session_state.simulation_base_df,
            ui_explainability_enabled=st.session_state.ui_explainability_enabled,
            anomaly_highlighting_enabled=st.session_state.anomaly_highlighting_enabled,
            human_trust_threshold=st.session_state.human_trust_threshold,
            human_expertise_level=st.session_state.human_expertise_level
        )
    else:
        st.warning("Simulation base data is not available. Please navigate to Page 1 to generate data and train the AI model.")
        st.session_state.initial_simulation_results_df = pd.DataFrame(columns=['true_diagnosis', 'ai_prediction', 'ai_confidence', 'human_final_decision'])


    st.write("Initial clinical case review simulation completed.")
    st.write("Sample of simulation results (AI vs Human decision):")
    st.dataframe(st.session_state.initial_simulation_results_df.head())

    st.markdown("""
    **Interpretation:**
    The initial clinical case review simulation has been executed using the default settings for UI/UX features and human parameters. The `initial_simulation_results_df` now contains the `true_diagnosis`, `ai_prediction`, `ai_confidence`, and the `human_final_decision` for each case. Observing the sample, we can see instances where the `human_final_decision` might differ from the `ai_prediction`, indicating an override. This DataFrame provides a comprehensive record of both AI's initial suggestions and the human operator's final decisions, setting the stage for quantitative performance analysis.
    """)