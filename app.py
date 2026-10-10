import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="DocClassify AI", page_icon="📄", layout="centered")

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 900px;}
.hero {background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
       padding: 2rem 2rem 1.5rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.1rem; color: white; padding: 0;}
.hero p {margin: .4rem 0 0; opacity: .92; font-size: 1.02rem; color: white;}
.chip {display: inline-block; padding: .18rem .75rem; border-radius: 999px;
       background: rgba(255,255,255,.2); font-size: .78rem; margin: .8rem .35rem 0 0; color: white;}
.result {border: 1px solid rgba(124,131,255,.5); background: rgba(124,131,255,.12);
         border-radius: 16px; padding: 1.1rem 1.3rem; margin: .8rem 0;}
.result .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .7;}
.result .cat {font-size: 1.7rem; font-weight: 700;}
</style>
<div class="hero">
  <h1>📄 DocClassify AI</h1>
  <p>Paste any text and the model tells you which topic it belongs to, with confidence scores and the words behind the decision.</p>
  <span class="chip">TF-IDF</span><span class="chip">Linear SVM</span>
  <span class="chip">6 topics</span><span class="chip">91% test accuracy</span>
</div>
""",
    unsafe_allow_html=True,
)

META = {
    "comp.graphics": ("🖼️", "Computer Graphics"),
    "rec.autos": ("🚗", "Cars and Automobiles"),
    "rec.sport.hockey": ("🏒", "Hockey"),
    "sci.med": ("🩺", "Medicine"),
    "sci.space": ("🚀", "Space"),
    "talk.politics.guns": ("🏛️", "Politics: Guns"),
}
EXAMPLES = {
    "🚀 Space": "NASA launched a new satellite to study the moon and Mars",
    "🏒 Hockey": "The goalie made an amazing save in overtime of the playoff game",
    "🖼️ Graphics": "My new graphics card renders 3D scenes and images much faster",
    "🩺 Medicine": "The doctor prescribed antibiotics for the infection and fever",
    "🚗 Cars": "This car has a powerful engine and great fuel mileage",
    "🏛️ Politics": "The debate over gun control laws continues in congress",
}
CLASSES = ["comp.graphics", "rec.autos", "rec.sport.hockey", "sci.med", "sci.space", "talk.politics.guns"]
CM = np.array([[181, 1, 3, 0, 3, 2], [5, 170, 2, 4, 1, 5], [1, 3, 183, 4, 0, 3],
               [2, 6, 0, 175, 6, 3], [9, 3, 3, 6, 168, 2], [2, 5, 2, 7, 8, 153]])


@st.cache_resource
def load_model():
    return joblib.load("doc_classifier.joblib")


def clean_text(text):
    tokens = re.findall(r"[a-z]{2,}", text.lower())
    return " ".join(t for t in tokens if t not in ENGLISH_STOP_WORDS)


def set_example(txt):
    st.session_state["doc_text"] = txt


bundle = load_model()
tfidf, model = bundle["tfidf"], bundle["model"]
if "doc_text" not in st.session_state:
    st.session_state["doc_text"] = ""

tab1, tab2, tab3 = st.tabs(["🔍 Classify", "📊 Model performance", "ℹ️ About"])

with tab1:
    st.caption("Try an example:")
    cols = st.columns(3)
    for i, (label, txt) in enumerate(EXAMPLES.items()):
        cols[i % 3].button(label, on_click=set_example, args=(txt,), key=f"ex_{i}")
    st.text_area("Your document", key="doc_text", height=150,
                 placeholder="Paste a paragraph about space, cars, hockey, medicine, graphics or gun politics...")

    if st.button("Classify document", type="primary", key="classify"):
        cleaned = clean_text(st.session_state["doc_text"])
        if not cleaned:
            st.warning("Please enter some text with real words in it.")
        else:
            vec = tfidf.transform([cleaned])
            scores = model.decision_function(vec)[0]
            probs = np.exp(scores - scores.max())
            probs = probs / probs.sum()
            order = np.argsort(probs)[::-1]
            top = order[0]
            emoji, nice = META.get(model.classes_[top], ("📄", model.classes_[top]))
            st.markdown(
                f'<div class="result"><div class="label">Predicted category</div>'
                f'<div class="cat">{emoji} {nice}</div></div>',
                unsafe_allow_html=True,
            )
            for i in order[:3]:
                e, n = META.get(model.classes_[i], ("📄", model.classes_[i]))
                st.write(f"{e} {n}: **{probs[i] * 100:.1f}%**")
                st.progress(float(probs[i]))
            contrib = vec.multiply(model.coef_[top]).tocsr()
            if contrib.nnz:
                pos = np.argsort(contrib.data)[::-1][:8]
                pos = [p for p in pos if contrib.data[p] > 0]
                words = tfidf.get_feature_names_out()[contrib.indices[pos]]
                if len(words):
                    st.info("Words pushing toward this category: " + ", ".join(words))
            st.caption("Confidence is a softmax over the SVM scores. Treat it as a relative ranking, not a true probability.")

with tab2:
    support = CM.sum(axis=1)
    prec, rec = np.diag(CM) / CM.sum(axis=0), np.diag(CM) / support
    f1 = 2 * prec * rec / (prec + rec)
    w = support / CM.sum()
    acc = np.trace(CM) / CM.sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc * 100:.1f}%")
    c2.metric("Weighted F1", f"{(f1 * w).sum():.4f}")
    c3.metric("Test documents", f"{CM.sum():,}")
    c4.metric("Categories", "6")
    st.subheader("Model comparison")
    comp = pd.DataFrame(
        {"Accuracy": [0.9045, 0.8974, 0.9089, acc, 0.8930], "Precision": [0.9058, 0.8977, 0.9090, (prec * w).sum(), np.nan],
         "Recall": [0.9045, 0.8974, 0.9089, (rec * w).sum(), 0.8930], "F1-Score": [0.9044, 0.8974, 0.9088, (f1 * w).sum(), 0.8930]},
        index=["Naive Bayes", "Logistic Regression", "Linear SVM", "Linear SVM (tuned, C=0.5)", "MiniLM embeddings + LR"]).round(4)
    st.dataframe(comp)
    st.caption("The three TF-IDF models are within about 1 point of each other, inside the margin of error for 1,131 test documents. The embedding baseline scored about 1.8 points lower.")
    st.subheader("Per-category results")
    st.dataframe(pd.DataFrame({"Precision": prec, "Recall": rec, "F1-Score": f1, "Support": support},
                              index=CLASSES).round(3))
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(CM, index=[f"actual: {c}" for c in CLASSES], columns=CLASSES)
                 .style.background_gradient(cmap="Blues"))

with tab3:
    st.markdown(
        """
**How it works**
1. Text is lowercased, tokenized and stripped of stop words
2. TF-IDF converts it into numeric features (unigrams and bigrams)
3. A Linear SVM, tuned with GridSearchCV, predicts one of 6 topics

**Limitations**
- Trained on 1990s Usenet posts, so modern slang and new topics are out of scope
- Only 6 categories: anything else gets forced into the closest one
- Short or vague text gives low-confidence results

Built by **Kaushikee Karmakar** | SparkIIT AI Internship Project 1
"""
    )
