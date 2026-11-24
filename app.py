from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "English": ROOT / "models" / "fake_news_pipeline.joblib",
    "हिन्दी": ROOT / "models" / "hindi_fake_news_pipeline.joblib",
}
METRICS_PATHS = {
    "English": ROOT / "models" / "metrics.json",
    "हिन्दी": ROOT / "models" / "hindi_metrics.json",
}

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eaf2ff 0%, #f8fbff 52%, #e8f7f1 100%);
        background-attachment: fixed;
    }
    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stForm"] {
        padding: 1.4rem;
        border: 1px solid #dce5f1;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
    }
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 10px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


language = st.selectbox("Language / भाषा", list(MODEL_PATHS))
model_path = MODEL_PATHS[language]
metrics_path = METRICS_PATHS[language]

st.title("Fake News Detection")
st.write(
    "Enter the article topic and a short brief, then check the model's prediction."
    if language == "English"
    else "खबर का विषय और उसका छोटा विवरण डालें, फिर मॉडल का अनुमान देखें।"
)

if language == "English":
    st.markdown(
        "Dataset: [Open WELFake news articles on Hugging Face]"
        "(https://huggingface.co/datasets/davanstrien/WELFake)"
    )
else:
    st.markdown(
        "डेटासेट: [Hindi fake और authentic खबरें देखने के लिए खोलें]"
        "(https://github.com/BSAkash/Hindi-Fake-News-Detection-Dataset)"
    )
    st.warning(
        "Hindi dataset के FAKE examples अधिकतर गलत दावों पर fact-check लेख हैं। "
        "मॉडल का label खबर की सच्चाई की पुष्टि नहीं करता।"
    )
if language == "English":
    st.info("The model predicts dataset patterns; it does not verify facts or check live sources.")

if not model_path.exists():
    st.error("Trained model नहीं मिला। Project folder में यह चलाएँ:")
    st.code("python train_model.py", language="powershell")
    st.stop()

model = load_model(str(model_path))
topic_key = "topic_hindi" if language == "हिन्दी" else "topic_english"
brief_key = "brief_hindi" if language == "हिन्दी" else "brief_english"
result_key = "analysis_result"
st.session_state.setdefault(topic_key, "")
st.session_state.setdefault(brief_key, "")
st.session_state.setdefault(result_key, None)

topic_label = "Article topic" if language == "English" else "खबर का विषय"
brief_label = "Brief of the article" if language == "English" else "खबर का संक्षिप्त विवरण"
with st.form(f"article_form_{language}"):
    st.text_input(
        topic_label,
        key=topic_key,
        placeholder="Example: Local school announces new opening hours"
        if language == "English"
        else "उदाहरण: शहर के स्कूलों के नए समय की घोषणा",
    )
    st.text_area(
        brief_label,
        key=brief_key,
        height=180,
        placeholder="Write 2–3 sentences describing the news."
        if language == "English"
        else "खबर को 2–3 वाक्यों में लिखें।",
    )
    submitted = st.form_submit_button(
        "Check news" if language == "English" else "खबर जांचें",
        type="primary",
    )

if submitted:
    content = "\n\n".join(
        part.strip()
        for part in (st.session_state[topic_key], st.session_state[brief_key])
        if part.strip()
    )
    if not st.session_state[topic_key].strip() or not st.session_state[brief_key].strip():
        st.session_state[result_key] = None
        st.error(
            "Enter both the article topic and its brief."
            if language == "English"
            else "खबर का विषय और उसका संक्षिप्त विवरण, दोनों भरें।"
        )
    else:
        prediction = int(model.predict([content])[0])
        probabilities = model.predict_proba([content])[0]
        st.session_state[result_key] = {
            "language": language,
            "prediction": prediction,
            "confidence": float(probabilities[prediction]),
        }

result = st.session_state[result_key]
if result and result["language"] == language:
    if result["prediction"] == 1:
        label = "Likely REAL / संभवतः सही"
    else:
        label = "Likely FAKE / संभवतः गलत"
    st.subheader(label)
    st.metric(
        "Model confidence / मॉडल का अनुमान",
        f"{result['confidence']:.1%}",
    )
    st.caption(
        "यह मॉडल का अनुमान है—खबर की सच्चाई की पुष्टि नहीं।"
        if language == "हिन्दी"
        else "This is a model prediction, not independent fact verification."
    )

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with st.expander("Show model evaluation results / मूल्यांकन देखें"):
        st.subheader("Test metrics")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
        metric_columns[1].metric("Fake precision", f"{metrics['precision_fake']:.1%}")
        metric_columns[2].metric("Fake recall", f"{metrics['recall_fake']:.1%}")
        metric_columns[3].metric("Fake F1", f"{metrics['f1_fake']:.1%}")
        st.write(
            f"ROC AUC: {metrics['roc_auc']:.3f} · "
            f"Train articles: {metrics['train_rows']:,} · "
            f"Test articles: {metrics['test_rows']:,}"
        )
        st.write("Confusion matrix (rows = actual, columns = predicted):")
        confusion = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual fake", "Actual real"],
            columns=["Predicted fake", "Predicted real"],
        )
        st.dataframe(confusion)
        st.caption(
            "These scores are from a held-out split of this dataset, not a guarantee "
            "of accuracy on current news."
        )
