import streamlit as st

from src.predict import load_model, predict_message

# Page configuration
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# Custom styling
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

    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Application heading
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
    "Enter an SMS message below to check whether "
    "our machine learning model classifies it as spam."
)

# Load the trained model
try:
    model, vectorizer = load_model()

except FileNotFoundError:
    st.error(
        "The trained model was not found. "
        "Please run: python src/train_model.py"
    )
    st.stop()

# Message input
message = st.text_area(
    "Enter your message",
    placeholder="Paste your SMS message here...",
    height=150
)

# Analyze button
if st.button("Analyze Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        result = predict_message(
            message,
            model,
            vectorizer
        )

        prediction = result["prediction"]

        spam_probability = result["spam_probability"]
        ham_probability = result["ham_probability"]

        st.subheader("Analysis Result")

        if prediction == "spam":
            st.error("⚠️ The model classified this message as SPAM.")

        else:
            st.success(
                "✅ The model classified this message as HAM "
                "(legitimate)."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam probability",
                f"{spam_probability:.1%}"
            )

        with col2:
            st.metric(
                "Ham probability",
                f"{ham_probability:.1%}"
            )

        st.caption(
            "These are model-estimated probabilities, not "
            "proof that a message is malicious or safe."
        )

# Footer
st.divider()

st.caption(
    "AI Spam Detector | Built with Python, "
    "Scikit-learn and Streamlit"
)