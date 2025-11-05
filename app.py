import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we explore the fascinating domain of "Generative AI in Clinical Decision Support" by simulating a Human-in-the-Loop Override system. This application allows users to step into the role of human operators, reviewing AI-generated diagnostic suggestions in a synthetic clinical context.

Our primary goal is to understand the interplay between AI predictions, human judgment, and various UI/UX features. Users will be able to:

-   **Interact with AI Suggestions**: Observe AI-generated diagnoses and associated confidence scores.
-   **Make Override Decisions**: Simulate human intervention based on AI's confidence and other contextual cues.
-   **Analyze Impact**: See how human overrides influence overall system performance, including accuracy, false positive rates, and false negative rates.
-   **Explore UI/UX Features**: Toggle features like "AI Explainability" and "Anomaly Highlighting" to understand their effect on human decision-making.
-   **Adjust Human Factors**: Modify "Human Trust Threshold" and "Human Expertise Level" to simulate different human operator profiles.

Through this interactive experience, we aim to provide key insights into how human oversight roles, coupled with effective UI/UX design, are critical for building robust and trustworthy AI systems in safety-critical applications like clinical decision support.

Formulae used for performance metrics:

*   **Accuracy:** $\\text{Accuracy} = \\frac{\\text{TP} + \\text{TN}}{\\text{TP} + \\text{TN} + \\text{FP} + \\text{FN}} $

*   **False Positive Rate (FPR):** $\\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}} $

*   **False Negative Rate (FNR):** $\\text{FNR} = \\frac{\\text{FN}}{\\text{FN} + \\text{TP}} $

Where:
-   TP (True Positives): Cases where the true label is 'Positive' and the prediction is 'Positive'.
-   TN (True Negatives): Cases where the true label is 'Negative' and the prediction is 'Negative'.
-   FP (False Positives): Cases where the true label is 'Negative' but the prediction is 'Positive' (Type I error).
-   FN (False Negatives): Cases where the true label is 'Positive' but the prediction is 'Negative' (Type II error).
""")
# Your code starts here
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
# Your code ends