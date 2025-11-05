import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
# Streamlit specific warning for pyplot is now handled by using Plotly

def run_page3():
    st.title("Simulation Results & Interactive Analysis")

    # --- Utility functions for metrics and plotting ---
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

    def plot_performance_comparison_plotly(ai_only_metrics: Dict[str, Any], ai_human_metrics: Dict[str, Any]):
        """Generates a Plotly bar chart comparing AI-only vs. AI+Human system performance."""
        metrics_set = set(ai_only_metrics.keys()) | set(ai_human_metrics.keys())
        metrics = sorted(list(metrics_set))

        ai_only_values = [ai_only_metrics.get(m, 0.0) for m in metrics]
        ai_human_values = [ai_human_metrics.get(m, 0.0) for m in metrics]

        fig = go.Figure(data=[
            go.Bar(name='AI-Only', x=metrics, y=ai_only_values, marker_color='rgb(99,110,250)'), # Viridis-like
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

    # --- Main content of page3 ---
    if 'current_simulation_results_df' not in st.session_state or st.session_state.current_simulation_results_df.empty:
        st.warning("No simulation results available. Please navigate to 'Page 1: Data & AI Model' to generate data and train the AI model, and then to 'Page 2: Simulation Controls' to run the simulation.")
        return

    st.subheader("5.8 Performance Metrics Calculation")
    st.markdown("""
    **Context & Business Value:**
    To rigorously assess the impact of human oversight, we need a robust set of performance metrics. The `calculate_performance_metrics` function quantifies the effectiveness of both the AI-only system and the combined AI+Human system. By calculating Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR, often critical in clinical settings), we gain a comprehensive understanding of how human intervention alters the system\'s propensity for different types of errors.

    This function is crucial for evaluating the \'fit-for-purpose\' aspect of the AI system, particularly in safety-critical domains like clinical decision support. Minimizing False Negatives, for example, could be a primary business objective to avoid missing critical diagnoses, and this function allows us to measure progress towards that goal.

    **Formulae:**

    *   **Accuracy:** The proportion of correct predictions (both positive and negative) out of the total number of cases.
        $$ \\text{Accuracy} = \\frac{\\text{TP} + \\text{TN}}{\\text{TP} + \\text{TN} + \\text{FP} + \\text{FN}} $$

    *   **False Positive Rate (FPR):** The proportion of actual negative cases that are incorrectly predicted as positive.
        $$ \\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}} $$

    *   **False Negative Rate (FNR):** The proportion of actual positive cases that are incorrectly predicted as negative.
        $$ \\text{FNR} = \\frac{\\text{FN}}{\\text{FN} + \\text{TP}} $$

    Where:
    -   TP (True Positives): Cases where the true label is \'Positive\' and the prediction is \'Positive\'.
    -   TN (True Negatives): Cases where the true label is \'Negative\' and the prediction is \'Negative\'.
    -   FP (False Positives): Cases where the true label is \'Negative\' but the prediction is \'Positive\' (Type I error).
    -   FN (False Negatives): Cases where the true label is \'Positive\' but the prediction is \'Negative\' (Type II error).
    """)
    st.info("The `calculate_performance_metrics` function has been defined.")

    st.subheader("5.9 Initial Performance Calculation")
    st.markdown("""
    **Context & Business Value:**
    Before diving into interactive analysis, it\'s essential to establish a baseline. This section calculates and displays the initial performance metrics for both the AI-only system and the AI+Human system based on the first simulation run (using default UI/UX settings). This provides a clear quantitative snapshot of the current state, allowing us to immediately see the benefits or trade-offs of human oversight with a specific configuration.

    Understanding this baseline is critical for evaluating the effectiveness of our human-in-the-loop design. For clinical decision support, for instance, we might observe how human intervention initially affects the False Negative Rate, which is often a key safety metric. This initial assessment directly supports the learning goals of understanding \'Human Oversight Roles\' and their impact on system safety.
    """)

    current_simulation_results_df = st.session_state.current_simulation_results_df

    ai_only_metrics = calculate_performance_metrics(
        current_simulation_results_df['true_diagnosis'].values,
        current_simulation_results_df['ai_prediction'].values
    )

    ai_human_metrics = calculate_performance_metrics(
        current_simulation_results_df['true_diagnosis'].values,
        current_simulation_results_df['human_final_decision'].values
    )

    st.write("--- Current Performance Metrics ---")
    st.write("AI-Only System Performance:")
    for metric, value in ai_only_metrics.items():
        st.write(f"  {metric}: {value:.4f}")

    st.write("\nAI+Human System Performance (with current settings):")
    for metric, value in ai_human_metrics.items():
        st.write(f"  {metric}: {value:.4f}")
    st.write("-----------------------------------")

    st.markdown("""
    **Interpretation:**
    The current performance metrics for both the AI-only and AI+Human systems (with the latest settings from the simulation controls) have been computed and displayed. This output provides a quantitative baseline, showing the accuracy, false positive rate (FPR), and false negative rate (FNR) for each system. We can now directly compare how the introduction of human oversight, with the current configurations, impacts these critical metrics. For example, if the FNR for the AI+Human system is lower than the AI-only system, it suggests that human intervention is effectively reducing critical missed diagnoses, aligning with the goal of improving system safety.
    """)

    st.subheader("5.10 Visualization: Comparative Performance Analysis (Aggregated Comparison)")
    st.markdown("""
    **Context & Business Value:**
    To provide an intuitive and clear understanding of the impact of human oversight, we will visualize the performance metrics using a bar chart. This `plot_performance_comparison` function will compare the Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR) between the AI-only and AI+Human systems. This visual comparison quickly conveys whether human intervention is beneficial, particularly in mitigating critical errors like false negatives in safety-critical clinical decision support.

    The use of a color-blind-friendly palette and clear labels ensures that the insights are accessible and understandable to all stakeholders. This visualization directly supports the learning goal of understanding how human intervention affects system performance and safety, providing a direct link between technical metrics and business outcomes.
    """)
    plot_performance_comparison_plotly(ai_only_metrics, ai_human_metrics)
    st.markdown("""
    **Interpretation:**
    The bar chart clearly illustrates the differences in accuracy, false positive rate, and false negative rate between the AI-only system and the system incorporating human oversight (with current settings). This visual comparison provides immediate insights into the benefits or trade-offs of human intervention. For instance, we can observe if human oversight has led to a reduction in critical errors like false negatives, which is often a key safety objective in clinical decision support. The chosen color palette ensures the visualization is accessible and clear to a broad audience.
    """)

    st.subheader("5.11 Visualization: Impact of UI/UX on Override Frequency (Relationship Plot)")
    st.markdown("""
    **Context & Business Value:**
    Understanding when and why human operators choose to override AI suggestions is critical for optimizing human-in-the-loop systems. The `plot_confidence_vs_override` function generates a scatter plot that visualizes the relationship between the AI\'s confidence scores and the frequency of human overrides. This visualization helps us determine if humans are primarily intervening when the AI is less confident, or if other factors, such as perceived anomalies (potentially influenced by UI/UX features), drive their intervention patterns.

    This analysis provides valuable feedback for UI/UX designers, helping them understand if their designs effectively guide human attention to high-risk or low-confidence AI predictions. It directly supports the learning goal of exploring the importance of designing effective UI/UX to help users understand model outputs and catch anomalies.
    """)
    plot_confidence_vs_override_plotly(current_simulation_results_df)
    st.markdown("""
    **Interpretation:**
    The scatter plot visualizes the interplay between AI confidence and human override decisions. Each point represents a bin of AI confidence scores, with the y-axis showing the frequency of human overrides within that confidence range. This plot helps us understand if human operators are effectively targeting low-confidence AI suggestions, or if other UI/UX cues (like anomaly highlighting) influence their intervention patterns, leading to overrides even at moderate AI confidence levels. A higher frequency of overrides at lower AI confidence would suggest effective human scrutiny where the AI is less certain.
    """)

    st.subheader("5.12 Visualization: System Performance Over Time (Trend Plot)")
    st.markdown("""
    **Context & Business Value:**
    In real-world operational settings, decision-making is a continuous process. It\'s crucial to monitor how system performance evolves over time to detect any drift or changes in effectiveness. The `plot_trend_metrics` function generates a line plot displaying the rolling average accuracy for both the AI-only and AI+Human systems over the sequence of simulated cases. This mimics real-time performance monitoring and helps to identify trends or fluctuations.

    This visualization is vital for assessing the long-term stability and consistency of the human-in-the-loop system. It can highlight whether human intervention effectively smooths out performance variations, maintains a higher baseline, or adapts to changing conditions over time. Such insights are essential for ensuring the ongoing assurance and reliability of critical AI systems.
    """)
    plot_trend_metrics_plotly(current_simulation_results_df, window_size=50)
    st.markdown("""
    **Interpretation:**
    The trend plot displays how the rolling average accuracy changes over the sequence of simulated clinical cases for both the AI-only and AI+Human systems. The `window_size` of 50 allows for a smoothed view of performance over time. This visualization provides insights into the stability and consistency of each system during continuous operation. We can observe if human intervention helps to smooth out performance fluctuations, maintain a higher baseline accuracy, or adapt to changing conditions compared to the AI operating autonomously. This helps to understand the dynamic impact of human-in-the-loop systems on long-term performance and reliability.
    """)

    st.header("6. Conclusion and Key Takeaways")
    st.markdown("""
    This simulation has provided a practical demonstration of \'Human Oversight Roles\' in AI-assisted decision-making. We\'ve seen how human intervention can act as a crucial fallback mechanism, potentially enhancing system reliability and safety by overriding AI suggestions. The lab highlighted:

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
    [1] Unit 3: Assurance Foundations for Critical ML, \'Human Oversight Roles\', [Case 2: Generative AI in Clinical Decision Support (Provided Resource)]. This section discusses roles for humans in the loop and the importance of designing UI/UX to help users understand model outputs.
    """)