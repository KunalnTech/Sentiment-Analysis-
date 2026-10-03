import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

st.set_page_config(page_title="Sentiment Analyser", page_icon="🔍", layout="centered")

st.title("🔍 Product Review Sentiment Analyser")
st.markdown("Fine-tuned **BERT** model trained on 40,000 Amazon product reviews (tested on 10,000).")
st.divider()

MODEL_PATH = "./model"
LABELS = {0: "Negative 😞", 1: "Positive 😊"}
COLORS = {0: "#e74c3c", 1: "#2ecc71"}

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        st.success("Fine-tuned model loaded!")
    except Exception:
        st.warning("Fine-tuned model not found in ./model. Predictions below come from an untrained base BERT and are not meaningful. Run train.py first.")
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        mdl = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    mdl.eval()
    return tok, mdl

tok, mdl = load_model()

def predict(text):
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze()
    label = int(torch.argmax(probs))
    return label, probs[label].item()

review = st.text_area("Paste a product review below:", placeholder="e.g. This laptop is amazing!", height=150)

if st.button("Analyse", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analysing…"):
            label, confidence = predict(review)
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Sentiment")
            st.markdown(f"<h2 style='color:{COLORS[label]}'>{LABELS[label]}</h2>", unsafe_allow_html=True)
        with col_b:
            st.markdown("### Confidence")
            st.metric(label="", value=f"{confidence * 100:.1f}%")
            st.progress(confidence)

st.divider()
st.markdown("<small>BERT fine-tuned on Amazon Polarity · Built by Kunaljit Das</small>", unsafe_allow_html=True)
