from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.predict import load_model, predict_message

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Spam Message Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PROJECT PATHS
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

PERFORMANCE_CHART = OUTPUT_DIR / "threshold_performance.png"
ERROR_CHART = OUTPUT_DIR / "threshold_errors.png"
RESULTS_FILE = OUTPUT_DIR / "threshold_results.csv"

# CUSTOM CSS
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1120;
            color: #e5e7eb;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        .hero {
            padding: 2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #172554, #111827);
            border: 1px solid #263449;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            color: #f8fafc;
            font-size: 2.4rem;
            margin-bottom: 0.5rem;
        }

        .hero p {
            color: #cbd5e1;
            font-size: 1rem;
            margin-bottom: 0;
        }

        .section-heading {
            color: #f8fafc;
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 1rem;
        }

        div[data-testid="stMetric"] {
            background-color: #111827;
            border: 1px solid #263449;
            padding: 1rem;
            border-radius: 12px;
        }

        div[data-testid="stMetricLabel"] {
            color: #cbd5e1;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        .stButton > button {
            width: 100%;
            border-radius: 10px;
            min-height: 3rem;
            font-weight: 600;
        }

        div[data-testid="stSidebar"] {
            background-color: #080d18;
        }

        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# LOAD TRAINED MODEL
@st.cache_resource
def get_model():
    return load_model()

# HEADER
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ AI Spam Message Detector</h1>
        <p>
            Analyze suspicious messages, explore spam probabilities,
            and understand how detection thresholds affect classification.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# SIDEBAR
with st.sidebar:
    st.title("🛡️ Spam Detector")

    st.markdown("---")

    st.subheader("Navigation")

    page = st.radio(
        "Choose a section",
        [
            "Message Detector",
            "Threshold Dashboard",
            "About the Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.subheader("Detection Settings")

    threshold = st.slider(
        "Spam Detection Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        format="%.2f",
        help=(
            "Messages with spam probabilities at or above this value "
            "are classified as spam."
        ),
    )

    st.metric(
        label="Current Threshold",
        value=f"{threshold:.0%}",
    )

    st.caption(
        "A lower threshold may detect more spam but can flag "
        "more legitimate messages."
    )

    st.caption(
        "A higher threshold may reduce false positives but can "
        "allow more spam messages to pass undetected."
    )

# MESSAGE DETECTOR
if page == "Message Detector":

    st.markdown(
        '<div class="section-heading">Analyze a Message</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Enter a message below to estimate its spam probability."
    )

    message = st.text_area(
        "Message to analyze",
        placeholder=(
            "Example: Congratulations! You have won a free prize. "
            "Click here to claim your reward."
        ),
        height=150,
        max_chars=5000,
    )

    analyze_button = st.button(
        "🔍 Analyze Message",
        type="primary",
        use_container_width=True,
    )

    if analyze_button:

        if not message.strip():

            st.warning("Please enter a message before analyzing it.")

        else:

            try:
                with st.spinner("Analyzing your message..."):

                    model = get_model()

                    result = predict_message(
                        message,
                        model,
                        threshold=threshold,
                    )

                prediction = str(result["prediction"]).lower()

                spam_probability = float(
                    result["spam_probability"]
                )

                ham_probability = float(
                    result["ham_probability"]
                )

                st.markdown("---")

                st.subheader("Analysis Results")

                # Classification result
                if prediction == "spam":

                    st.error(
                        "🚨 Spam Detected",
                        icon="⚠️",
                    )

                    st.write(
                        "The estimated spam probability meets or "
                        "exceeds your selected threshold."
                    )

                else:

                    st.success(
                        "✅ Classified as Legitimate",
                        icon="✅",
                    )

                    st.write(
                        "The estimated spam probability is below "
                        "your selected threshold."
                    )

                # Probability metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Spam Probability",
                        f"{spam_probability:.2%}",
                    )

                with col2:
                    st.metric(
                        "Legitimate Probability",
                        f"{ham_probability:.2%}",
                    )

                with col3:
                    st.metric(
                        "Detection Threshold",
                        f"{threshold:.0%}",
                    )

                # Probability bar
                st.subheader("Spam Probability")

                st.progress(
                    max(
                        0.0,
                        min(1.0, spam_probability),
                    )
                )

                st.caption(
                    f"The model estimates a "
                    f"{spam_probability:.2%} probability of spam."
                )

                # Probability chart
                probability_chart = go.Figure(
                    data=[
                        go.Pie(
                            labels=[
                                "Spam",
                                "Legitimate",
                            ],
                            values=[
                                spam_probability,
                                ham_probability,
                            ],
                            hole=0.65,
                            textinfo="label+percent",
                            hovertemplate=(
                                "%{label}<br>"
                                "Probability: %{value:.2%}"
                                "<extra></extra>"
                            ),
                        )
                    ]
                )

                probability_chart.update_layout(
                    title="Message Classification Probabilities",
                    height=350,
                    margin=dict(
                        t=60,
                        b=20,
                        l=20,
                        r=20,
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        color="#e5e7eb",
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.1,
                        xanchor="center",
                        x=0.5,
                    ),
                )

                st.plotly_chart(
                    probability_chart,
                    use_container_width=True,
                )

                st.info(
                    "This result is a model prediction, not a guarantee "
                    "that the message is safe or malicious. "
                    "Do not rely on this tool alone to make security decisions."
                )

            except FileNotFoundError as error:

                st.error(
                    "The trained model could not be found. "
                    "Please train the model first."
                )

                st.code(str(error))

            except Exception as error:

                st.error(
                    "Something went wrong while analyzing the message."
                )

                st.code(str(error))

# THRESHOLD EVALUATION DASHBOARD
elif page == "Threshold Dashboard":

    st.markdown(
        '<div class="section-heading">Threshold Evaluation Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Compare model performance across different thresholds "
        "and explore the trade-off between false positives and missed spam."
    )

    if RESULTS_FILE.exists():

        try:
            results_df = pd.read_csv(RESULTS_FILE)

            # Remove accidental whitespace from column names
            results_df.columns = (
                results_df.columns.astype(str).str.strip()
            )

            required_columns = [
                "Threshold",
                "Precision",
                "Recall",
                "F1-score",
                "False Positives",
                "Missed Spam",
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in results_df.columns
            ]

            if missing_columns:

                st.error(
                    "The evaluation CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:

                # Convert percentages to numeric values when needed
                for column in [
                    "Precision",
                    "Recall",
                    "F1-score",
                ]:

                    results_df[column] = pd.to_numeric(
                        results_df[column]
                        .astype(str)
                        .str.replace("%", "", regex=False),
                        errors="coerce",
                    )

                results_df["Threshold"] = pd.to_numeric(
                    results_df["Threshold"],
                    errors="coerce",
                )

                for column in [
                    "False Positives",
                    "Missed Spam",
                ]:

                    results_df[column] = pd.to_numeric(
                        results_df[column],
                        errors="coerce",
                    )

                results_df = results_df.dropna(
                    subset=required_columns
                )

                # Convert proportion values to percentages
                # if the CSV stores them as decimals.
                for column in [
                    "Precision",
                    "Recall",
                    "F1-score",
                ]:

                    if (
                        not results_df.empty
                        and results_df[column].max() <= 1
                    ):
                        results_df[column] = (
                            results_df[column] * 100
                        )

                if results_df.empty:

                    st.warning(
                        "No valid evaluation results were found "
                        "in the CSV file."
                    )

                else:

                    # Summary metrics
                    best_f1_row = results_df.loc[
                        results_df["F1-score"].idxmax()
                    ]

                    lowest_fp_row = results_df.loc[
                        results_df["False Positives"].idxmin()
                    ]

                    highest_recall_row = results_df.loc[
                        results_df["Recall"].idxmax()
                    ]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Highest F1-score",
                            f"{best_f1_row['F1-score']:.2f}%",
                            help="Highest F1-score in this evaluation.",
                        )

                    with col2:
                        st.metric(
                            "Fewest False Positives",
                            f"{int(lowest_fp_row['False Positives'])}",
                            help=(
                                "Minimum false positives in this "
                                "evaluation."
                            ),
                        )

                    with col3:
                        st.metric(
                            "Highest Recall",
                            f"{highest_recall_row['Recall']:.2f}%",
                            help=(
                                "Highest recall in this evaluation."
                            ),
                        )

                    st.caption(
                        "These figures describe your manually written "
                        "evaluation set, not expected real-world performance."
                    )

                    st.markdown("---")

                    # Performance chart
                    st.subheader(
                        "1. Model Performance Across Thresholds"
                    )

                    if PERFORMANCE_CHART.exists():

                        st.image(
                            str(PERFORMANCE_CHART),
                            use_container_width=True,
                        )

                    else:

                        st.info(
                            "The saved performance chart was not found. "
                            "Run python src/plot_threshold_results.py "
                            "to generate it."
                        )

                    # Error chart
                    st.subheader(
                        "2. False Positives vs Missed Spam"
                    )

                    if ERROR_CHART.exists():

                        st.image(
                            str(ERROR_CHART),
                            use_container_width=True,
                        )

                    else:

                        st.info(
                            "The saved error chart was not found. "
                            "Run python src/plot_threshold_results.py "
                            "to generate it."
                        )

                    # Interactive Plotly chart
                    st.subheader(
                        "3. Interactive Threshold Comparison"
                    )

                    comparison_chart = go.Figure()

                    comparison_chart.add_trace(
                        go.Scatter(
                            x=results_df["Threshold"],
                            y=results_df["Precision"],
                            mode="lines+markers",
                            name="Precision",
                        )
                    )

                    comparison_chart.add_trace(
                        go.Scatter(
                            x=results_df["Threshold"],
                            y=results_df["Recall"],
                            mode="lines+markers",
                            name="Recall",
                        )
                    )

                    comparison_chart.add_trace(
                        go.Scatter(
                            x=results_df["Threshold"],
                            y=results_df["F1-score"],
                            mode="lines+markers",
                            name="F1-score",
                        )
                    )

                    comparison_chart.update_layout(
                        xaxis_title="Spam Probability Threshold",
                        yaxis_title="Score (%)",
                        yaxis=dict(
                            range=[0, 105],
                        ),
                        height=450,
                        hovermode="x unified",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(
                            color="#e5e7eb",
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0,
                        ),
                    )

                    st.plotly_chart(
                        comparison_chart,
                        use_container_width=True,
                    )

                    # Evaluation table
                    st.subheader(
                        "4. Threshold Evaluation Results"
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        label="Download Evaluation Results (CSV)",
                        data=results_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="threshold_results.csv",
                        mime="text/csv",
                    )

                    # Explanation
                    with st.expander(
                        "How should I interpret these results?"
                    ):

                        st.markdown(
                            """
                            - **Precision:** Of the messages flagged as
                              spam, the proportion that were actually
                              labelled spam in this test set.

                            - **Recall:** Of the messages labelled spam
                              in this test set, the proportion detected
                              by the model.

                            - **F1-score:** A combined measure of
                              precision and recall.

                            - **False positives:** Legitimate messages
                              incorrectly flagged as spam.

                            - **Missed spam:** Spam messages incorrectly
                              classified as legitimate.

                            A lower threshold generally flags more
                            messages as spam. A higher threshold generally
                            requires stronger model confidence before
                            a message is classified as spam.
                            """
                        )

        except Exception as error:

            st.error(
                "Could not read the evaluation results."
            )

            st.code(str(error))

    else:

        st.warning(
            "Evaluation results were not found."
        )

        st.write(
            "Run the following command in your project terminal:"
        )

        st.code(
            "python src/evaluate_thresholds.py",
            language="bash",
        )

# ABOUT THE PROJECT
elif page == "About the Project":

    st.markdown(
        '<div class="section-heading">About This Project</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        The AI Spam Message Detector is a machine learning project
        that classifies text messages as spam or legitimate.
        It also allows users to adjust the spam probability threshold
        and examine how this changes classification results.
        """
    )

    st.subheader("Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            - **Python** — application and model code
            - **Scikit-learn** — machine learning pipeline
            - **TF-IDF** — text feature extraction
            """
        )

    with col2:
        st.markdown(
            """
            - **Multinomial Naive Bayes** — message classification
            - **Streamlit** — interactive web application
            - **Plotly and Matplotlib** — charts and visualizations
            """
        )

    st.subheader("Threshold Evaluation")

    st.write(
        """
        The project evaluates multiple probability thresholds using
        a manually written set of legitimate and phishing-style
        messages. It compares precision, recall, F1-score,
        false positives, and missed spam.
        """
    )

    st.warning(
        """
        **Limitations:** The model was trained on a general SMS spam
        dataset, and the additional threshold evaluation uses only
        40 manually written messages. The phishing-style examples
        are not a professionally verified dataset. These results
        do not establish reliable real-world phishing detection
        performance.
        """
    )

    st.subheader("Future Improvements")

    st.markdown(
        """
        - Evaluate the model using a larger, independently labelled dataset.
        - Include more examples of legitimate security alerts and phishing messages.
        - Compare additional text classification models.
        - Improve detection of phishing messages without unnecessarily
          flagging legitimate messages.
        """
    )

# FOOTER
st.markdown(
    """
    <div class="footer">
        AI Spam Message Detector · Machine Learning Project
        <br>
        Predictions are estimates and should not replace security checks.
    </div>
    """,
    unsafe_allow_html=True,
)