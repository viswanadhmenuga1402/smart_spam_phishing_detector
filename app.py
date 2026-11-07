import streamlit as st
import pandas as pd
from pathlib import Path
from src.model import load_dataset, train_model, predict_message, evaluate_model
from src.security import inspect_risk
from src.history import save_scan, read_history, clear_history

st.set_page_config(page_title="Smart Spam & Phishing Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Smart Spam & Phishing Detector")
st.caption("A beginner-friendly Python + Machine Learning internship project")

@st.cache_resource
def get_model():
    data = load_dataset(Path(__file__).resolve().parent / "data" / "messages.csv")
    return train_model(data)

try:
    model, vectorizer, X_test, y_test = get_model()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Project overview")
    st.write("**ML:** TF-IDF + Multinomial Naive Bayes")
    st.write("**Python:** functions, modules, exceptions, CSV, OOP-friendly design")
    st.write("**Security:** separate rule-based risk indicators")
    st.warning("This demo is educational. Do not use it as the sole basis for security decisions.")

tab_scan, tab_metrics, tab_history, tab_about = st.tabs(
    ["🔎 Scan message", "📊 Model evaluation", "🧾 Scan history", "ℹ️ About"]
)

with tab_scan:
    st.subheader("Check an email or message")
    message = st.text_area(
        "Paste message text",
        placeholder="Example: Congratulations! You have won a prize. Click the link to claim it.",
        height=170
    )
    if st.button("Analyze message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message before scanning.")
        else:
            try:
                label, score = predict_message(message, model, vectorizer)
                risk_level, flags = inspect_risk(message)
                c1, c2, c3 = st.columns(3)
                c1.metric("ML classification", label)
                c2.metric("Model score for predicted class", f"{score:.1%}")
                c3.metric("Rule-based risk level", risk_level)
                if flags:
                    st.write("**Risk indicators found:**")
                    for flag in flags:
                        st.write(f"- {flag}")
                else:
                    st.info("No configured rule-based risk indicators were found.")
                st.caption("The model score is not a guarantee that the message is safe or malicious.")
                save_scan(message, label, score, risk_level, flags)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

with tab_metrics:
    st.subheader("Evaluation on a held-out test split")
    metrics, matrix = evaluate_model(model, vectorizer, X_test, y_test)
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("Precision (spam)", f"{metrics['precision']:.1%}")
    c.metric("Recall (spam)", f"{metrics['recall']:.1%}")
    d.metric("F1-score (spam)", f"{metrics['f1_score']:.1%}")
    st.write("**Confusion matrix** (rows = actual, columns = predicted; Ham then Spam)")
    st.dataframe(pd.DataFrame(matrix, index=["Actual Ham", "Actual Spam"],
                              columns=["Predicted Ham", "Predicted Spam"]))
    st.caption("Metrics depend on the small example dataset. Use a larger, representative dataset before making performance claims.")

with tab_history:
    st.subheader("Previous scans")
    history = read_history()
    if history.empty:
        st.info("No scans saved yet. Analyze a message to create history.")
    else:
        st.dataframe(history, use_container_width=True)
        st.download_button("Download history CSV", history.to_csv(index=False), "scan_history.csv", "text/csv")
        if st.button("Clear scan history"):
            clear_history()
            st.rerun()

with tab_about:
    st.subheader("How it works")
    st.markdown("""
    1. **TF-IDF** converts message text into numeric features.
    2. **Multinomial Naive Bayes** learns word patterns from labeled examples.
    3. The model predicts **Spam** or **Ham** (legitimate message).
    4. A separate rule-based checker highlights configured indicators such as urgent requests,
       credential requests, suspicious link patterns, and prize claims.
    5. The application records scans in a CSV history file.

    The rule checker is a simple demonstration, not a full phishing detector. It does not visit links,
    verify sender identities, or prove that a message is safe.
    """)
