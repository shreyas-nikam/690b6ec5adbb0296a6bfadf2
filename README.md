# QuLab: Human-in-the-Loop Override Simulator for Clinical Decision Support

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## 1. Project Title and Description

This Streamlit application, "QuLab," serves as an interactive lab project designed to explore the critical domain of **Generative AI in Clinical Decision Support**, with a particular focus on the pivotal role of **Human Oversight** in enhancing safety and performance.

Through a simulated clinical environment, users assume the role of a human operator, reviewing AI-generated diagnostic suggestions. The application allows you to make decisions to override or accept AI recommendations, directly observing the impact of these choices on the system's overall accuracy, false positive rates, and false negative rates.

The lab aims to provide hands-on experience and insights into:

*   **Assurance-Case Methodology**: How to evaluate uncertainty and model risk in real-world conditions, translating "fit-for-purpose" into measurable evidence.
*   **UI/UX Design for Trustworthy AI**: The importance of effective user interface and user experience design in helping operators understand complex AI model outputs and identify anomalies.
*   **Human Oversight Roles**: The dynamic interplay between human intervention and AI, and how it shapes system safety and operational performance in critical applications.

By adjusting various parameters, you can explore how human factors (e.g., trust, expertise), AI explainability, and anomaly highlighting influence the collaborative decision-making process, ultimately leading to more robust and reliable clinical outcomes.

The core relationship explored is:

$$
\text{AI-assisted System Performance} = f(\text{AI Model Performance}, \text{Human Oversight Quality}, \text{UI/UX Effectiveness})
$$

## 2. Features

*   **Synthetic Clinical Data Generation**: Automatically generates a configurable synthetic dataset of patient records for a binary classification task.
*   **Data Validation**: Performs checks on generated data for column presence, data types, uniqueness, and missing values.
*   **AI Model Training**: Trains a `RandomForestClassifier` on the synthetic data to simulate AI diagnostic suggestions.
*   **AI Prediction & Confidence**: Generates AI predictions and associated confidence scores for test cases.
*   **Interactive Human Override Simulation**: Models human decision-making, allowing users to define parameters such as:
    *   **AI Explainability**: Toggle feature simulating AI providing reasoning for its predictions.
    *   **Anomaly Highlighting**: Toggle feature flagging unusual or low-confidence AI predictions.
    *   **Human Trust Threshold**: Slider to adjust the human operator's propensity to scrutinize AI.
    *   **Human Expertise Level**: Slider to adjust the human's skill in correcting AI errors.
*   **Performance Metrics Calculation**: Computes key classification metrics (Accuracy, False Positive Rate, False Negative Rate) for both AI-only and AI+Human systems.
*   **Dynamic Visualizations**: Presents interactive Plotly charts to compare system performance:
    *   Bar chart comparing aggregated AI-only vs. AI+Human metrics.
    *   Scatter plot showing AI Confidence vs. Human Override Frequency.
    *   Line plot displaying rolling average performance trends over time.
*   **Real-time Parameter Tuning**: All controls are interactive, providing immediate feedback on how changing human factors and UI/UX features impact simulation results and visualizations.
*   **Modular Design**: Application logic is organized into separate pages for clarity and maintainability.

## 3. Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/quolab-generative-ai-clinical-support.git
    cd quolab-generative-ai-clinical-support
    ```
    *(Note: Replace `https://github.com/your-username/quolab-generative-ai-clinical-support.git` with the actual repository URL if different.)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is not provided, you can create it with `pip freeze > requirements.txt` after manually installing the core libraries, or simply install them directly: `pip install streamlit pandas numpy scikit-learn plotly`)*

## 4. Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

2.  **Navigate through the pages**:
    *   Use the **sidebar navigation** to move between:
        *   **Page 1: Data & AI Model**: Initializes synthetic clinical data, performs data validation, and trains the AI diagnostic model. This page sets up the foundational data and AI for the simulation.
        *   **Page 2: Simulation Controls**: Contains the interactive widgets (`Show AI Explainability`, `Highlight Anomalies`, `Human Trust Threshold`, `Human Expertise Level`) that allow you to configure the human-in-the-loop simulation.
        *   **Page 3: Results & Analysis**: Displays the performance metrics and visualizations, dynamically updating as you adjust parameters on "Page 2".

3.  **Interact with the simulation**:
    *   Start on **Page 1** to ensure data and AI are initialized.
    *   Move to **Page 2** and adjust the sliders and checkboxes. Observe how these changes affect the `human_final_decision` logic.
    *   Switch to **Page 3** to see the immediate impact of your chosen parameters on overall system accuracy, error rates, and human override patterns through updated charts and metrics. Experiment with different combinations to understand their effects.

## 5. Project Structure

The project is organized into a modular structure to keep the code clean and manageable:

```
.
├── app.py                      # Main Streamlit application entry point and navigation logic
├── requirements.txt            # List of Python dependencies
└── application_pages/          # Directory containing individual Streamlit page implementations
    ├── __init__.py             # Makes application_pages a Python package
    ├── page1.py                # Handles data generation, validation, and AI model training
    ├── page2.py                # Implements human override logic and UI/UX controls
    └── page3.py                # Calculates performance metrics and generates visualizations
```

## 6. Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/)
*   **Programming Language**: Python
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (for `RandomForestClassifier`, `OneHotEncoder`, `ColumnTransformer`, `train_test_split`)
*   **Visualization**: [Plotly](https://plotly.com/python/) (for interactive charts: `plotly.graph_objects`, `plotly.express`)

## 7. Contributing

This project is primarily a lab exercise. However, contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to standard Python best practices and is well-commented.

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 9. Contact

For any questions or further information, please refer to the QuantUniversity resources or contact the course instructors.

---

_This README was generated based on the provided Streamlit application code and documentation for a lab project on Generative AI in Clinical Decision Support._
