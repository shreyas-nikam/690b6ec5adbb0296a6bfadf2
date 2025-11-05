import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

def run_page3():
    st.title("Simulation Results & Analysis")

    st.subheader("5.8 Utility Function: Performance Metrics Calculation")
    st.markdown("""
    **Context & Business Value:**
    To rigorously assess the impact of human oversight, we need a robust set of performance metrics. The `calculate_performance_metrics` function quantifies the effectiveness of both the AI-only system and the combined AI+Human system. By calculating Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR, often critical in clinical settings), we gain a comprehensive understanding of how human intervention alters the system's propensity for different types of errors.

    This function is crucial for evaluating the 'fit-for-purpose' aspect of the AI system, particularly in safety-critical domains like clinical decision support. Minimizing False Negatives, for example, could be a primary business objective to avoid missing critical diagnoses, and this function allows us to measure progress towards that goal.

    **Formulae:**

    *   **Accuracy:** The proportion of correct predictions (both positive and negative) out of the total number of cases.
        $$ \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} $$

    *   **False Positive Rate (FPR):** The proportion of actual negative cases that are incorrectly predicted as positive.
        $$ \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} $$

    *   **False Negative Rate (FNR):** The proportion of actual positive cases that are incorrectly predicted as negative.
        $$ \text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} $$

    Where:
    -   TP (True Positives): Cases where the true label is 'Positive' and the prediction is 'Positive'.
    -   TN (True Negatives): Cases where the true label is 'Negative' and the prediction is 'Negative'.
    -   FP (False Positives): Cases where the true label is 'Negative' but the prediction is 'Positive' (Type I error).
    -   FN (False Negatives): Cases where the true label is 'Positive' but the prediction is 'Negative' (Type II error).
    """)
    st.info("The `calculate_performance_metrics` function has been defined.")

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

    ai_only_initial_metrics = calculate_performance_metrics(
        st.session_state.initial_simulation_results_df['true_diagnosis'].values,
        st.session_state.initial_simulation_results_df['ai_prediction'].values
    )

    ai_human_initial_metrics = calculate_performance_metrics(
        st.session_state.initial_simulation_results_df['true_diagnosis'].values,
        st.session_state.initial_simulation_results_df['human_final_decision'].values
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

    def plot_performance_comparison(ai_only_metrics: Dict[str, Any], ai_human_metrics: Dict[str, Any]):
        """Generates a bar chart comparing AI-only vs. AI+Human system performance using Plotly."""
        metrics_set = set(ai_only_metrics.keys()) | set(ai_human_metrics.keys())
        metrics = sorted(list(metrics_set))

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

        fig = px.bar(metrics_df, x='Metric', y='Value', color='System',
                     barmode='group', title='Comparative System Performance (AI-Only vs. AI+Human)',
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(yaxis_range=[0, 1], font=dict(size=12))
        fig.update_xaxes(title_text='Performance Metric')
        fig.update_yaxes(title_text='Metric Value')
        st.plotly_chart(fig)

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
        and the frequency of human overrides using Plotly.
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

        fig = px.scatter(plot_df, x='confidence_midpoint', y='Override_Frequency',
                         title='AI Confidence vs. Human Override Frequency',
                         labels={'confidence_midpoint': 'AI Confidence (Probability of Positive Class)',
                                 'Override_Frequency': 'Frequency of Human Override'},
                         color_discrete_sequence=[px.colors.qualitative.Deep[0]])
        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(xaxis_range=[0, 1], yaxis_range=[-0.05, 1.05], font=dict(size=12))
        st.plotly_chart(fig)

    st.subheader("5.11 Visualization: Impact of UI/UX on Override Frequency (Relationship Plot)")
    st.markdown("""
    **Context & Business Value:**
    Understanding when and why human operators choose to override AI suggestions is critical for optimizing human-in-the-loop systems. The `plot_confidence_vs_override` function generates a scatter plot that visualizes the relationship between the AI's confidence scores and the frequency of human overrides. This visualization helps us determine if humans are primarily intervening when the AI is less confident, or if other factors, such as perceived anomalies (potentially influenced by UI/UX features), drive their intervention patterns.

    This analysis provides valuable feedback for UI/UX designers, helping them understand if their designs effectively guide human attention to high-risk or low-confidence AI predictions. It directly supports the learning goal of exploring the importance of designing effective UI/UX to help users understand model outputs and catch anomalies.
    """)
    plot_confidence_vs_override(st.session_state.initial_simulation_results_df)
    st.markdown("""
    **Interpretation:**
    The scatter plot visualizes the interplay between AI confidence and human override decisions. Each point represents a bin of AI confidence scores, with the y-axis showing the frequency of human overrides within that confidence range. This plot helps us understand if human operators are effectively targeting low-confidence AI suggestions, or if other UI/UX cues (like anomaly highlighting) influence their intervention patterns, leading to overrides even at moderate AI confidence levels. A higher frequency of overrides at lower AI confidence would suggest effective human scrutiny where the AI is less certain.
    """)

    def plot_trend_metrics(simulation_results_df: pd.DataFrame, window_size: int = 50) -> None:
        """
        Generates a line plot displaying rolling average performance metrics (e.g., accuracy) over the sequence of simulated decisions,
        comparing AI-only vs. AI+Human trends using Plotly.
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
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['rolling_ai_accuracy'], mode='lines',
                                 name='AI-Only Rolling Accuracy', line=dict(width=2, color=px.colors.qualitative.Plotly[0])))
        fig.add_trace(go.Scatter(x=df.index, y=df['rolling_ai_human_accuracy'], mode='lines',
                                 name='AI+Human Rolling Accuracy', line=dict(width=2, color=px.colors.qualitative.Plotly[1])))
        
        fig.update_layout(title_text=f'Rolling Average Performance Trends (Window Size: {window_size})',
                          xaxis_title='Simulation Step', yaxis_title='Rolling Accuracy',
                          font=dict(size=12))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_layout(legend_title_text='Metric')
        st.plotly_chart(fig)

    st.subheader("5.12 Visualization: System Performance Over Time (Trend Plot)")
    st.markdown("""
    **Context & Business Value:**
    In real-world operational settings, decision-making is a continuous process. It's crucial to monitor how system performance evolves over time to detect any drift or changes in effectiveness. The `plot_trend_metrics` function generates a line plot displaying the rolling average accuracy for both the AI-only and AI+Human systems over the sequence of simulated cases. This mimics real-time performance monitoring and helps to identify trends or fluctuations.

    This visualization is vital for assessing the long-term stability and consistency of the human-in-the-loop system. It can highlight whether human intervention effectively smooths out performance variations, maintains a higher baseline, or adapts to changing conditions over time. Such insights are essential for ensuring the ongoing assurance and reliability of critical AI systems.
    """)
    plot_trend_metrics(st.session_state.initial_simulation_results_df, window_size=50)
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
    st.info("Change the values in the widgets in the sidebar to see dynamic updates below.")
    
    # Import simulate_human_decision from page2 for rerun
    from application_pages.page2 import simulate_human_decision

    def run_full_simulation_page3(
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


    current_simulation_results_df = run_full_simulation_page3(
        simulation_df=st.session_state.simulation_base_df,
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
