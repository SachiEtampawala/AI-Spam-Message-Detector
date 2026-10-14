import streamlit as st

from src.predict import load_model, predict_message

# 1. Page configuration
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# 2. Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #101827;
        color: #F1F5F9;
    }

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: #60A5FA;
    }

    .subtitle {
        text-align: center;
        color: #CBD5E1;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Application heading
st.markdown(
    '<div class="main-title">AI Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze messages using machine learning'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Enter a message and adjust the detection threshold "
    "to explore the trade-off between detecting spam "
    "and avoiding false alarms."
)

# 4. Load the trained model
try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "The trained model was not found. "
        "Run: python src/train_model.py"
    )
    st.stop()

# 5. Threshold control
threshold = st.slider(
    "Spam detection threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help=(
        "A lower threshold flags more messages as spam. "
        "A higher threshold requires stronger model evidence."
    )
)

st.caption(
    f"Current threshold: {threshold:.0%}. "
    "A message is classified as spam when its estimated "
    "spam probability is greater than or equal to this value."
)

# 6. Message input
message = st.text_area(
    "Enter your message",
    placeholder="Paste your SMS or security alert here...",
    height=150
)

# 7. Analyze the message
if st.button(
    "Analyze Message",
    use_container_width=True
):

    if not message.strip():

        st.warning(
            "Please enter a message first."
        )

    else:

        result = predict_message(
            message,
            model,
            threshold=threshold
        )

        prediction = result["prediction"]

        spam_probability = result["spam_probability"]
        ham_probability = result["ham_probability"]

        st.subheader("Analysis Result")

        if prediction == "spam":

            st.error(
                "⚠️ The model classified this message as SPAM."
            )

        else:

            st.success(
                "✅ The model classified this message as "
                "HAM (legitimate)."
            )

        # Probability metrics.
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam probability",
                f"{spam_probability:.1%}"
            )

        with col2:
            st.metric(
                "Legitimate probability",
                f"{ham_probability:.1%}"
            )

        # Show the selected threshold.
        st.write(
            f"**Classification threshold:** {threshold:.0%}"
        )

        # Show the difference between probability and threshold.
        if spam_probability >= threshold:

            st.write(
                "The spam probability meets or exceeds "
                "the selected threshold."
            )

        else:

            st.write(
                "The spam probability is below "
                "the selected threshold."
            )

        st.caption(
            "This model was trained on an SMS spam dataset. "
            "It has not been validated as a standalone phishing "
            "or security-alert detection system. Never rely on "
            "its result alone to decide whether a message is safe."
        )

# 8. Footer
st.divider()

st.caption(
    "AI Spam Detector | Python, Scikit-learn and Streamlit"
)