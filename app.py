from pathlib import Path
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import load_model, predict_message

# PAGE SETUP
st.set_page_config(
    page_title="SignalLens | Message Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_FILE = OUTPUT_DIR / "threshold_results.csv"

# A distinctive warm-paper + ink + mint visual identity.
def load_css():
    """Load the app's visual styles from styles/style.css."""
    css_path = PROJECT_DIR / "styles" / "style.css"
    try:
        css = css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning("The stylesheet was not found. Create styles/style.css to apply the custom design.")
        return
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)

load_css()

@st.cache_resource
def get_model():
    return load_model()


def safe_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def get_signals(message: str):
    """Transparent phrase-based hints; these are not model explanations."""
    text = message.lower()
    checks = [
        ("Urgency or pressure", r"\b(urgent|immediately|act now|limited time|within 24 hours|final notice|expires today)\b", "Pushes the reader to act quickly instead of checking the request."),
        ("Money or prize offer", r"\b(won|winner|prize|free gift|cash reward|claim your|lottery|bonus)\b", "Unexpected rewards can be used to encourage a quick response."),
        ("Credential or PIN request", r"\b(password|pin|passcode|one[- ]time password|otp|login details|verify your account)\b", "Requests involving credentials or verification codes deserve extra caution."),
        ("Payment or account threat", r"\b(suspended|blocked|unpaid|payment failed|delivery fee|account compromised|card compromised|pay now)\b", "Uses an account, card, delivery, or payment problem to prompt action."),
        ("Link or web address", r"https?://|www\.|\b(bit\.ly|tinyurl\.com|t\.co)/?", "Contains a web address. Check the real domain independently before opening it."),
        ("Contact or callback request", r"\b(call this number|reply now|text back|contact us|call immediately)\b", "Encourages a reply or callback that may need independent verification."),
    ]
    found = []
    for title, pattern, explanation in checks:
        if re.search(pattern, text):
            found.append((title, explanation))
    if not found:
        found.append(("No listed phrase pattern found", "This quick scan did not match its selected phrase patterns. That does not prove the message is safe."))
    return found

def make_explanation(prediction, spam_probability, threshold, signals):
    gap = spam_probability - threshold
    if prediction == "spam":
        opening = "The model placed this message above your selected spam threshold."
        if gap >= 0.25:
            strength = "The score is comfortably above the decision line."
        else:
            strength = "The score is relatively close to the decision line, so the result is more threshold-sensitive."
    else:
        opening = "The model placed this message below your selected spam threshold."
        if abs(gap) <= 0.10:
            strength = "The score is close to the decision line; a small threshold change could change the label."
        else:
            strength = "The score is below the current decision line, but a low score is not proof of safety."
    signal_text = " ".join(title.lower() for title, _ in signals if title != "No listed phrase pattern found")
    if signal_text:
        signal_line = f"The phrase scan also noticed: {signal_text}."
    else:
        signal_line = "The phrase scan did not find any of its selected warning patterns."
    return f"{opening} {strength} {signal_line}"

def probability_chart(spam_probability, ham_probability, threshold):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[spam_probability * 100], y=["Spam likelihood"], orientation="h",
        name="Spam likelihood", marker_color="#e58b56", text=[f"{spam_probability:.1%}"],
        textposition="inside", hovertemplate="Spam likelihood: %{x:.2f}%<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=[ham_probability * 100], y=["Legitimate likelihood"], orientation="h",
        name="Legitimate likelihood", marker_color="#55b98b", text=[f"{ham_probability:.1%}"],
        textposition="inside", hovertemplate="Legitimate likelihood: %{x:.2f}%<extra></extra>"
    ))
    fig.add_vline(x=threshold * 100, line_dash="dash", line_color="#18211f", line_width=2,
                  annotation_text=f"Decision threshold {threshold:.0%}", annotation_position="top")
    fig.update_layout(
        barmode="overlay", height=230, margin=dict(l=8, r=15, t=48, b=18),
        xaxis=dict(range=[0, 100], title="Model score (%)", ticksuffix="%", gridcolor="#e3e6df"),
        yaxis=dict(title="", autorange="reversed"), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#18211f"),
    )
    return fig

def analyze_one(message, threshold, model):
    result = predict_message(message, model, threshold=threshold)
    prediction = str(result["prediction"]).lower()
    spam_probability = safe_float(result.get("spam_probability"))
    ham_probability = safe_float(result.get("ham_probability"), 1.0 - spam_probability)
    signals = get_signals(message)
    return {
        "prediction": prediction,
        "spam_probability": spam_probability,
        "ham_probability": ham_probability,
        "signals": signals,
        "explanation": make_explanation(prediction, spam_probability, threshold, signals),
    }

# APP SHELL
with st.sidebar:
    st.markdown('<div class="brand-lockup">◈ SignalLens</div>', unsafe_allow_html=True)
    st.markdown("### Message intelligence")
    st.caption("A small machine-learning lab for suspicious text messages.")
    st.markdown("---")
    page = st.radio(
        "Workspace",
        ["Scan Studio", "Batch Analyzer", "Model Observatory", "Project Notes"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**Decision settings**")
    threshold = st.slider(
        "Spam decision line", min_value=0.10, max_value=0.90,
        value=0.50, step=0.05, format="%.0%%",
        help="A message is labelled spam when its model score meets or exceeds this value.",
    )
    st.metric("Current decision line", f"{threshold:.0%}")
    st.caption("Lower = catch more possible spam, but flag more legitimate messages. Higher = fewer alerts, but more spam may pass.")
    st.markdown("---")
    st.caption("MODEL NOTE · Scores are estimates, not proof that a message is safe or malicious.")

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">TEXT SECURITY / MACHINE LEARNING LAB</div>
      <h1>Read between<br>the messages.</h1>
      <p>Inspect a message, see the model's estimated spam score, and review transparent warning signals before deciding what to do next.</p>
      <span class="hero-tag">TF-IDF + Naive Bayes · Interactive analysis</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# SCAN STUDIO
if page == "Scan Studio":
    st.markdown('<div class="eyebrow">01 / Single message</div><div class="section-title">Scan Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Paste the message exactly as received. Avoid adding passwords, private codes, or personal information you do not need to analyze.</div>', unsafe_allow_html=True)

    sample_col1, sample_col2, sample_col3 = st.columns([1, 1, 2])
    with sample_col1:
        if st.button("Try a prize message", use_container_width=True):
            st.session_state["scan_message"] = "Congratulations! You have won a free prize. Click https://example.com to claim your reward today."
    with sample_col2:
        if st.button("Try a delivery alert", use_container_width=True):
            st.session_state["scan_message"] = "Your parcel is on the way. The courier will deliver it tomorrow afternoon."
    with sample_col3:
        st.caption("Sample text is provided for demonstration. Real messages can be more subtle.")

    message = st.text_area(
        "Message content", key="scan_message", height=180, max_chars=5000,
        placeholder="Paste a text message here…",
        help="The message is processed by the local model configured in this project.",
    )
    run_scan = st.button("Run message analysis  ↗", type="primary", use_container_width=True)

    if run_scan:
        if not message.strip():
            st.warning("Paste a message first, then run the analysis.")
        else:
            try:
                with st.status("Inspecting message features…", expanded=True) as status:
                    st.write("Loading the trained text-classification pipeline")
                    model = get_model()
                    st.write("Estimating spam and legitimate-message scores")
                    result = analyze_one(message.strip(), threshold, model)
                    st.write("Preparing the explanation and phrase signals")
                    status.update(label="Analysis complete", state="complete", expanded=False)

                st.session_state["last_scan"] = {
                    "message": message.strip(), "threshold": threshold, **result
                }
                record = {
                    "message": message.strip()[:160],
                    "prediction": result["prediction"].upper(),
                    "spam_probability": result["spam_probability"],
                    "threshold": threshold,
                }
                history = st.session_state.get("scan_history", [])
                st.session_state["scan_history"] = [record] + history[:9]
            except FileNotFoundError as error:
                st.error("The trained model was not found. Train the model before launching this app.")
                st.code(str(error))
            except Exception as error:
                st.error("The message could not be analyzed. Check the terminal for details.")
                st.code(str(error))

    last = st.session_state.get("last_scan")
    if last:
        st.markdown("---")
        st.markdown('<div class="eyebrow">Analysis report</div><div class="section-title">What the model found</div>', unsafe_allow_html=True)
        spam_p = last["spam_probability"]
        ham_p = last["ham_probability"]
        pred = last["prediction"]
        if pred == "spam":
            pill = '<span class="status-pill status-danger">⚑ Flag for review</span>'
            result_title = "Possible spam"
            result_copy = "The model score crossed your selected decision line. Verify the sender independently before interacting."
        elif spam_p >= last["threshold"] - 0.10:
            pill = '<span class="status-pill status-caution">◉ Near decision line</span>'
            result_title = "Lower score, still worth checking"
            result_copy = "The message is below the current threshold but close enough that a different setting could change the label."
        else:
            pill = '<span class="status-pill status-safe">✓ Below decision line</span>'
            result_title = "Not flagged by this model"
            result_copy = "The model score is below your selected decision line. This does not verify the sender or guarantee safety."

        left, right = st.columns([1.02, 1.35], gap="large")
        with left:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown(pill, unsafe_allow_html=True)
            st.markdown(f'<div class="big-score">{spam_p:.1%}</div>', unsafe_allow_html=True)
            st.markdown("**Estimated spam score**", unsafe_allow_html=True)
            st.markdown(f'<div class="subtle">{result_title}</div>', unsafe_allow_html=True)
            st.markdown('<div class="scanline"><span></span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="subtle">{result_copy}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")
            a, b = st.columns(2)
            a.metric("Legitimate score", f"{ham_p:.1%}")
            b.metric("Decision line", f"{last['threshold']:.0%}")
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Score comparison**")
            st.caption("The dashed line marks the threshold used for the label.")
            st.plotly_chart(probability_chart(spam_p, ham_p, last["threshold"]), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Explanation")
        st.markdown(f'<div class="panel">{last["explanation"]}<br><br><span class="subtle">This explanation combines the model score with a small, visible phrase-matching checklist. It is not generated by a large language model and may miss context, sarcasm, or new scam patterns.</span></div>', unsafe_allow_html=True)

        st.markdown("### Signals noticed in the text")
        sig_col, sig_info = st.columns([1.2, 1])
        with sig_col:
            for title, detail in last["signals"]:
                is_none = title.startswith("No listed")
                dot_class = "good" if is_none else ""
                st.markdown(f'<div class="signal"><span class="signal-dot {dot_class}"></span><div><div class="signal-title">{title}</div><div class="signal-copy">{detail}</div></div></div>', unsafe_allow_html=True)
        with sig_info:
            st.markdown('<div class="panel"><div class="eyebrow">Next best step</div><h4>Verify out of band</h4><p class="subtle">Do not use the phone number or link inside a suspicious message to verify it. Open the organisation’s official app or website yourself, or contact a known official number.</p><p class="subtle">Never share an OTP, PIN, or password in response to a message.</p></div>', unsafe_allow_html=True)

    history = st.session_state.get("scan_history", [])
    if history:
        with st.expander("Recent scans in this session"):
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
            st.caption("History is held in this browser session only; it is not a permanent audit log.")

# BATCH ANALYZER
elif page == "Batch Analyzer":
    st.markdown('<div class="eyebrow">02 / Multiple messages</div><div class="section-title">Batch Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Compare several messages in one run. Enter one message per line. This is useful for small experiments and does not replace testing on a representative labelled dataset.</div>', unsafe_allow_html=True)
    batch_text = st.text_area("Messages to compare", height=230, placeholder="Your package will arrive tomorrow.\nURGENT: Verify your account password now at https://example.com\nThanks for your help earlier!", key="batch_text")
    if st.button("Analyze message set", type="primary", use_container_width=True):
        messages = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not messages:
            st.warning("Enter at least one message, with each message on its own line.")
        elif len(messages) > 100:
            st.warning("Please analyze 100 messages or fewer per batch.")
        else:
            try:
                model = get_model()
                rows = []
                progress = st.progress(0, text="Preparing batch analysis…")
                for index, text in enumerate(messages, start=1):
                    result = analyze_one(text, threshold, model)
                    rows.append({
                        "Message": text[:240],
                        "Label": result["prediction"].upper(),
                        "Spam score (%)": round(result["spam_probability"] * 100, 2),
                        "Legitimate score (%)": round(result["ham_probability"] * 100, 2),
                        "Phrase signals": ", ".join(title for title, _ in result["signals"]),
                    })
                    progress.progress(index / len(messages), text=f"Analyzed {index} of {len(messages)} messages")
                progress.empty()
                df = pd.DataFrame(rows)
                st.session_state["batch_results"] = df
            except Exception as error:
                st.error("Batch analysis failed. Check that the trained model exists.")
                st.code(str(error))

    batch_df = st.session_state.get("batch_results")
    if batch_df is not None:
        st.markdown("---")
        total = len(batch_df)
        flagged = int((batch_df["Label"] == "SPAM").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages scanned", total)
        c2.metric("Flagged by model", flagged)
        c3.metric("Below decision line", total - flagged)
        st.markdown("### Results table")
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        st.download_button("Download batch report (CSV)", data=batch_df.to_csv(index=False).encode("utf-8"), file_name="signallens_batch_report.csv", mime="text/csv", use_container_width=True)
        chart = go.Figure(go.Bar(
            x=batch_df["Message"].str.slice(0, 35), y=batch_df["Spam score (%)"],
            marker_color=["#e58b56" if label == "SPAM" else "#55b98b" for label in batch_df["Label"]],
            text=batch_df["Spam score (%)"].map(lambda value: f"{value:.1f}%"), textposition="outside",
            hovertemplate="%{x}<br>Spam score: %{y:.2f}%<extra></extra>",
        ))
        chart.add_hline(y=threshold * 100, line_dash="dash", line_color="#18211f", annotation_text="Current decision line")
        chart.update_layout(title="Spam score by message", xaxis_title="Message (shortened)", yaxis_title="Spam score (%)", yaxis=dict(range=[0, 110]), height=380, margin=dict(t=55, b=70), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope, sans-serif", color="#18211f"))
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

# MODEL OBSERVATORY
elif page == "Model Observatory":
    st.markdown('<div class="eyebrow">03 / Evaluation</div><div class="section-title">Model Observatory</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Explore how changing the decision threshold affects the trade-off between finding spam and avoiding false alarms.</div>', unsafe_allow_html=True)
    if RESULTS_FILE.exists():
        try:
            df = pd.read_csv(RESULTS_FILE)
            df.columns = df.columns.astype(str).str.strip()
            required = ["Threshold", "Precision", "Recall", "F1-score", "False Positives", "Missed Spam"]
            if not all(column in df.columns for column in required):
                st.error("The threshold results file does not contain the expected columns.")
            else:
                for column in ["Threshold", "Precision", "Recall", "F1-score", "False Positives", "Missed Spam"]:
                    df[column] = pd.to_numeric(df[column].astype(str).str.replace("%", "", regex=False), errors="coerce")
                df = df.dropna(subset=required)
                for column in ["Precision", "Recall", "F1-score"]:
                    if not df.empty and df[column].max() <= 1:
                        df[column] *= 100
                if df.empty:
                    st.warning("No valid rows were found in the threshold results file.")
                else:
                    best_f1 = df.loc[df["F1-score"].idxmax()]
                    best_recall = df.loc[df["Recall"].idxmax()]
                    fewest_fp = df.loc[df["False Positives"].idxmin()]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Best F1 in this test", f"{best_f1['F1-score']:.1f}%", f"Threshold {best_f1['Threshold']:.2f}")
                    m2.metric("Highest recall", f"{best_recall['Recall']:.1f}%", f"Threshold {best_recall['Threshold']:.2f}")
                    m3.metric("Fewest false positives", f"{int(fewest_fp['False Positives'])}", f"Threshold {fewest_fp['Threshold']:.2f}")
                    st.caption("These metrics describe the small manually written evaluation set only. They are not reliable estimates of real-world phishing performance.")

                    left, right = st.columns(2, gap="large")
                    with left:
                        fig = go.Figure()
                        for column, color in [("Precision", "#e58b56"), ("Recall", "#55b98b"), ("F1-score", "#607be8")]:
                            fig.add_trace(go.Scatter(x=df["Threshold"], y=df[column], mode="lines+markers", name=column, line=dict(width=3, color=color), marker=dict(size=7)))
                        fig.update_layout(title="Quality metrics by threshold", xaxis_title="Threshold", yaxis_title="Score (%)", yaxis=dict(range=[0, 105]), height=390, hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope, sans-serif", color="#18211f"), legend=dict(orientation="h", y=1.15))
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    with right:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(x=df["Threshold"], y=df["False Positives"], name="False positives", marker_color="#e58b56"))
                        fig2.add_trace(go.Bar(x=df["Threshold"], y=df["Missed Spam"], name="Missed spam", marker_color="#607be8"))
                        fig2.update_layout(title="Error counts by threshold", barmode="group", xaxis_title="Threshold", yaxis_title="Number of messages", height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope, sans-serif", color="#18211f"), legend=dict(orientation="h", y=1.15))
                        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("### Full evaluation table")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button("Export evaluation results", data=df.to_csv(index=False).encode("utf-8"), file_name="threshold_results.csv", mime="text/csv")
                    with st.expander("How to read these metrics"):
                        st.markdown("- **Precision:** How many messages flagged as spam were spam in this test set.\n- **Recall:** How many of the test-set spam messages were detected.\n- **F1-score:** A combined measure of precision and recall.\n- **False positives:** Legitimate messages incorrectly flagged as spam.\n- **Missed spam:** Spam messages incorrectly labelled legitimate.\n\nA lower threshold usually catches more spam but creates more false alarms. A higher threshold usually reduces false alarms but can miss more spam.")
        except Exception as error:
            st.error("Could not load threshold evaluation results.")
            st.code(str(error))
    else:
        st.info("Evaluation results are not available yet. Run this command from the project root:")
        st.code("python src/evaluate_thresholds.py", language="bash")

# PROJECT NOTES
elif page == "Project Notes":
    st.markdown('<div class="eyebrow">04 / Documentation</div><div class="section-title">Project Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">A compact technical overview of what this application does—and what it does not claim to do.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="panel"><div class="eyebrow">Pipeline</div><h3>From text to score</h3><p class="subtle">The text-classification pipeline converts a message into TF-IDF features and uses a Multinomial Naive Bayes classifier to estimate whether it resembles spam in the training data.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="eyebrow">Interface</div><h3>Interactive analysis</h3><p class="subtle">SignalLens presents a score comparison, adjustable decision line, batch report, threshold charts, and a transparent phrase checklist.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="eyebrow">Explanation design</div><h3>Evidence, not magic</h3><p class="subtle">The explanation combines model scores with simple, visible phrase patterns. It is not an external generative-AI call, and it cannot explain every decision made by the classifier.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="eyebrow">Safety boundary</div><h3>Never treat a low score as proof</h3><p class="subtle">The model was trained on a general SMS spam dataset. A suspicious message can be missed, and a legitimate message can be flagged. Confirm unexpected requests through a trusted channel.</p></div>', unsafe_allow_html=True)

    st.markdown("### Technology stack")
    tech1, tech2, tech3, tech4 = st.columns(4)
    tech1.metric("Language", "Python")
    tech2.metric("ML", "Scikit-learn")
    tech3.metric("Interface", "Streamlit")
    tech4.metric("Charts", "Plotly")

    st.markdown("### Before publishing")
    st.markdown("- Keep the dataset source and licence details in the README.\n- Do not describe the phrase checklist as a real generative AI explanation.\n- Mention the size and limitations of the manually written evaluation set.\n- Do not claim the app can reliably detect all phishing messages.\n- Test normal, suspicious, empty, long, and multi-line inputs before deployment.")

st.markdown('<div class="footer">SIGNALLENS · MESSAGE INTELLIGENCE LAB<br>Model scores are estimates. Verify sensitive requests through official channels.</div>', unsafe_allow_html=True)