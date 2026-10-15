from pathlib import Path
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import load_model, predict_message

st.set_page_config(
    page_title="SignalLens | Spam Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "outputs" / "threshold_results.csv"


def load_css():
    css_file = ROOT / "styles" / "style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.warning("Could not find styles/style.css")


load_css()


@st.cache_resource
def get_model():
    return load_model()


def phrase_signals(message):
    text = message.lower()
    patterns = [
        ("Urgent language", r"\b(urgent|immediately|act now|limited time|final notice|expires today)\b"),
        ("Prize or money offer", r"\b(won|winner|prize|free gift|cash reward|claim your|lottery|bonus)\b"),
        ("Password or PIN request", r"\b(password|pin|passcode|one[- ]time password|otp|login details|verify your account)\b"),
        ("Payment or account warning", r"\b(suspended|blocked|unpaid|payment failed|delivery fee|account compromised|pay now)\b"),
        ("Contains a link", r"https?://|www\.|\b(bit\.ly|tinyurl\.com|t\.co)"),
        ("Asks you to reply or call", r"\b(call this number|reply now|text back|contact us|call immediately)\b"),
    ]
    return [name for name, pattern in patterns if re.search(pattern, text)]


def analyse(message, threshold, model):
    result = predict_message(message, model, threshold=threshold)
    spam = float(result.get("spam_probability", 0.0))
    ham = float(result.get("ham_probability", 1.0 - spam))
    label = str(result["prediction"]).lower()
    signals = phrase_signals(message)
    if label == "spam":
        explanation = "The model's spam score is at or above your selected threshold. Review the message carefully and verify the sender independently."
    else:
        explanation = "The model's spam score is below your selected threshold. This does not guarantee that the message is safe."
    if signals:
        explanation += " The text also contains these patterns: " + ", ".join(signals).lower() + "."
    else:
        explanation += " The quick phrase check did not find any of its selected warning patterns."
    return {"label": label, "spam": spam, "ham": ham, "signals": signals, "explanation": explanation}


def heading(kicker, title, description=None):
    st.markdown(f'<div class="eyebrow">{kicker}</div><h2 class="page-title">{title}</h2>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<p class="page-description">{description}</p>', unsafe_allow_html=True)


def score_chart(spam, ham, threshold):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[spam * 100], y=["Spam"], orientation="h", marker_color="#d85f55", text=[f"{spam:.1%}"], textposition="inside", name="Spam"))
    fig.add_trace(go.Bar(x=[ham * 100], y=["Legitimate"], orientation="h", marker_color="#58a985", text=[f"{ham:.1%}"], textposition="inside", name="Legitimate"))
    fig.update_layout(
        barmode="group", height=175, margin=dict(l=4, r=8, t=8, b=8),
        xaxis=dict(range=[0, 100], ticksuffix="%", title=None, gridcolor="#eceee8"),
        yaxis=dict(title=None), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#262b28"),
    )
    return fig


with st.sidebar:
    st.markdown('<div class="brand-mark">◈ <span>SignalLens</span></div>', unsafe_allow_html=True)
    st.caption("A simple message spam checker")
    st.markdown("---")
    page = st.radio("Menu", ["Scan Message", "Batch Analysis", "Model Performance", "About"], label_visibility="visible")
    st.markdown("---")
    threshold = st.slider("Spam threshold", 0.10, 0.90, 0.50, 0.05, format="%.0%%")
    st.caption("A message at or above this score is labelled spam.")

st.markdown('<div class="top-note">MESSAGE SECURITY <span>•</span> MACHINE LEARNING</div>', unsafe_allow_html=True)
st.title("Check a message.")
st.markdown('<p class="intro">Paste a text message and get a simple spam risk estimate.</p>', unsafe_allow_html=True)

if page == "Scan Message":
    heading("01 / QUICK CHECK", "Scan a message", "Paste the message below. You do not need to include personal details.")
    message = st.text_area("Message", key="scan_message", height=145, placeholder="Paste your message here…")
    sample_a, sample_b, blank = st.columns([1, 1, 3])
    with sample_a:
        if st.button("Try a spam example"):
            st.session_state["scan_message"] = "Congratulations! You have won a free prize. Click https://example.com to claim your reward now."
            st.rerun()
    with sample_b:
        if st.button("Try a normal example"):
            st.session_state["scan_message"] = "Hi, are we still meeting at the cafe tomorrow afternoon?"
            st.rerun()
    if st.button("Analyse message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please paste a message first.")
        else:
            try:
                with st.spinner("Checking the message…"):
                    result = analyse(message.strip(), threshold, get_model())
                st.session_state["last_scan"] = {"message": message.strip(), "threshold": threshold, **result}
                record = {"Message": message.strip()[:160], "Result": result["label"].upper(), "Spam score": f"{result['spam']:.1%}"}
                st.session_state["history"] = ([record] + st.session_state.get("history", []))[:8]
            except Exception as error:
                st.error("The message could not be analysed. Check that your trained model is available.")
                st.code(str(error))

    result = st.session_state.get("last_scan")
    if result:
        st.markdown("---")
        heading("YOUR RESULT", "Analysis result")
        is_spam = result["label"] == "spam"
        near = not is_spam and result["spam"] >= result["threshold"] - 0.10
        if is_spam:
            status, title, style = "SPAM WARNING", "Possible spam detected", "danger"
        elif near:
            status, title, style = "CHECK CAREFULLY", "Close to the decision threshold", "caution"
        else:
            status, title, style = "NOT FLAGGED", "No spam detected by the model", "safe"
        left, right = st.columns([0.9, 1.1], gap="large")
        with left:
            st.markdown(f'<div class="result-card {style}"><div class="result-status">{status}</div><div class="result-score">{result["spam"]:.1%}</div><div class="result-label">Estimated spam score</div><h3>{title}</h3><p>{"Be careful with links, payments, and requests for personal information." if is_spam else "A low score does not guarantee that the message is safe."}</p></div>', unsafe_allow_html=True)
            st.progress(min(max(result["spam"], 0.0), 1.0), text=f"Spam score · {result['spam']:.1%}")
        with right:
            st.markdown('<div class="simple-card"><h3>Score comparison</h3>', unsafe_allow_html=True)
            st.plotly_chart(score_chart(result["spam"], result["ham"], result["threshold"]), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="simple-card explanation-card"><h3>Why this result?</h3>', unsafe_allow_html=True)
        st.write(result["explanation"])
        if result["signals"]:
            st.markdown("**Patterns found**")
            st.write(" · ".join(result["signals"]))
        st.caption("This is a machine-learning estimate with a simple phrase check, not proof that a message is safe or malicious.")
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("Show scanned message"):
            st.write(result["message"])
    history = st.session_state.get("history", [])
    if history:
        with st.expander("Recent scans"):
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

elif page == "Batch Analysis":
    heading("02 / MULTIPLE MESSAGES", "Batch analysis", "Enter one message per line to check several messages together.")
    batch = st.text_area("Messages", height=190, placeholder="Your parcel will arrive tomorrow.\nURGENT: Verify your password at https://example.com", key="batch_input")
    if st.button("Analyse messages", type="primary", use_container_width=True):
        messages = [line.strip() for line in batch.splitlines() if line.strip()]
        if not messages:
            st.warning("Enter at least one message.")
        elif len(messages) > 100:
            st.warning("Please enter 100 messages or fewer.")
        else:
            try:
                model = get_model()
                rows = []
                progress = st.progress(0, text="Starting analysis…")
                for i, text in enumerate(messages, 1):
                    item = analyse(text, threshold, model)
                    rows.append({"Message": text[:240], "Result": item["label"].upper(), "Spam score (%)": round(item["spam"] * 100, 2), "Legitimate score (%)": round(item["ham"] * 100, 2), "Patterns found": ", ".join(item["signals"])})
                    progress.progress(i / len(messages), text=f"Checked {i} of {len(messages)}")
                progress.empty()
                st.session_state["batch_results"] = pd.DataFrame(rows)
            except Exception as error:
                st.error("Batch analysis failed. Check that your trained model is available.")
                st.code(str(error))
    df = st.session_state.get("batch_results")
    if df is not None:
        a, b, c = st.columns(3)
        a.metric("Messages checked", len(df))
        b.metric("Flagged as spam", int((df["Result"] == "SPAM").sum()))
        c.metric("Not flagged", int((df["Result"] != "SPAM").sum()))
        st.markdown("### Results")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV report", df.to_csv(index=False).encode("utf-8"), "signallens_batch_report.csv", "text/csv", use_container_width=True)
        fig = go.Figure(go.Bar(x=list(range(1, len(df) + 1)), y=df["Spam score (%)"], marker_color=["#d85f55" if x == "SPAM" else "#58a985" for x in df["Result"]]))
        fig.update_layout(title="Spam score by message", xaxis_title="Message number", yaxis_title="Spam score (%)", yaxis=dict(range=[0, 100]), height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

elif page == "Model Performance":
    heading("03 / MODEL CHECK", "Model performance", "Explore how the decision threshold changes the model's results.")
    if not RESULTS_FILE.exists():
        st.info("Evaluation data has not been created yet. Run this command from the project folder:")
        st.code("python src/evaluate_thresholds.py", language="bash")
    else:
        try:
            df = pd.read_csv(RESULTS_FILE)
            df.columns = df.columns.astype(str).str.strip()
            needed = ["Threshold", "Precision", "Recall", "F1-score", "False Positives", "Missed Spam"]
            if not all(col in df.columns for col in needed):
                st.error("The evaluation CSV is missing expected columns.")
            else:
                for col in needed:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False), errors="coerce")
                df = df.dropna(subset=needed)
                for col in ["Precision", "Recall", "F1-score"]:
                    if not df.empty and df[col].max() <= 1:
                        df[col] *= 100
                if df.empty:
                    st.warning("No valid evaluation rows were found.")
                else:
                    a, b, c = st.columns(3)
                    a.metric("Best F1 score", f"{df['F1-score'].max():.1f}%")
                    b.metric("Highest recall", f"{df['Recall'].max():.1f}%")
                    c.metric("Fewest false alarms", f"{int(df['False Positives'].min())}")
                    fig = go.Figure()
                    for col, color in [("Precision", "#6b82d9"), ("Recall", "#58a985"), ("F1-score", "#d98b55")]:
                        fig.add_trace(go.Scatter(x=df["Threshold"], y=df[col], mode="lines+markers", name=col, line=dict(color=color, width=3)))
                    fig.update_layout(title="Scores by threshold", xaxis_title="Threshold", yaxis_title="Score (%)", yaxis=dict(range=[0, 105]), height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button("Download evaluation CSV", df.to_csv(index=False).encode("utf-8"), "threshold_results.csv", "text/csv")
                    st.caption("These results describe the evaluation dataset only. They do not guarantee real-world phishing detection performance.")
        except Exception as error:
            st.error("Could not read the evaluation file.")
            st.code(str(error))

else:
    heading("04 / PROJECT INFO", "About SignalLens", "A small machine-learning project for exploring suspicious text messages.")
    st.markdown('<div class="simple-card"><h3>How it works</h3><p>The text is converted into TF-IDF features and classified with a Multinomial Naive Bayes model. SignalLens displays the model score and checks for a few visible warning phrases.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="simple-card"><h3>Technology</h3><p>Python · Streamlit · pandas · scikit-learn · Plotly</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="simple-card"><h3>Important limitations</h3><p>The model can make mistakes. A message marked as legitimate may still be a scam, and a legitimate message may be flagged. Do not share passwords, PINs, or one-time codes. Verify unexpected requests through an official app, website, or known phone number.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">SIGNALLENS · A SIMPLE SPAM MESSAGE CHECKER</div>', unsafe_allow_html=True)
