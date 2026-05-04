import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="LexDomain — Next-Word Prediction",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --border: #2a2a3a;
    --accent-news: #00d4ff;
    --accent-lit: #ff6b6b;
    --text: #e8e8f0;
    --text-dim: #6b6b80;
    --text-dimmer: #3a3a50;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 3rem 4rem !important;
    max-width: 1400px !important;
}

.hero {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 2.5rem;
    margin-bottom: 2.5rem;
}
.hero-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    color: var(--text);
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
}
.hero-title span { color: var(--accent-news); }
.hero-sub {
    font-size: 0.95rem;
    color: var(--text-dim);
    font-weight: 300;
    max-width: 420px;
    line-height: 1.6;
}
.hero-right {
    display: flex;
    gap: 2.5rem;
    align-items: flex-end;
    padding-bottom: 0.5rem;
}
.stat-block { text-align: right; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.model-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.model-card {
    background: var(--surface);
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.model-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-news { background: var(--accent-news); box-shadow: 0 0 8px var(--accent-news); }
.dot-lit  { background: var(--accent-lit);  box-shadow: 0 0 8px var(--accent-lit); }
.model-name { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.9rem; color: var(--text); }
.model-desc { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--text-dim); }

[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-news) !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.2) !important;
}

[data-testid="stButton"] > button {
    background: var(--accent-news) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    background: #fff !important;
    transform: translateY(-1px) !important;
}

.result-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}
.result-header {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.result-body { padding: 1.5rem; }
.completion-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.3rem;
    font-weight: 300;
    line-height: 1.6;
    color: var(--text-dim);
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.gen-news { color: var(--accent-news) !important; font-weight: 500; }
.gen-lit  { color: var(--accent-lit)  !important; font-weight: 500; }

.step-row { margin-bottom: 1rem; }
.step-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }
.step-num { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text-dimmer); letter-spacing: 0.1em; }
.step-word { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: var(--text); font-weight: 500; }

.token-bar-wrap { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.25rem; }
.token-word {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    width: 110px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.token-word.chosen { color: var(--text); font-weight: 500; }
.token-bar-outer { flex: 1; height: 4px; background: var(--surface2); border-radius: 2px; overflow: hidden; }
.token-bar-inner { height: 100%; border-radius: 2px; }
.token-pct { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text-dimmer); width: 38px; text-align: right; flex-shrink: 0; }

.arch-strip {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
}
.arch-layer { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--text-dim); }
.arch-sublabel { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--text-dimmer); letter-spacing: 0.1em; }
.arch-arrow { font-size: 0.8rem; color: var(--text-dimmer); }

label, [data-testid="stWidgetLabel"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)

MAX_SEQ_LEN = 20

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    import tensorflow as tf
    missing = [f for f in [
        "news_model.keras", "news_model_tokenizer.pickle",
        "gutenberg_model.keras", "gutenberg_model_tokenizer.pickle"
    ] if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    nm = tf.keras.models.load_model("news_model.keras")
    gm = tf.keras.models.load_model("gutenberg_model.keras")
    with open("news_model_tokenizer.pickle", "rb") as f: nt = pickle.load(f)
    with open("gutenberg_model_tokenizer.pickle", "rb") as f: gt = pickle.load(f)
    return nm, nt, gm, gt, []

def predict_next_words(seed_text, model, tokenizer, n_words, top_k=5):
    import tensorflow as tf
    idx2word = {v: k for k, v in tokenizer.word_index.items()}
    text, steps = seed_text.lower().strip(), []
    for _ in range(n_words):
        seq    = tokenizer.texts_to_sequences([text])[0][-MAX_SEQ_LEN:]
        padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=MAX_SEQ_LEN, padding="pre")
        probs  = model.predict(padded, verbose=0)[0]
        top_ids = probs.argsort()[-top_k:][::-1]
        top_preds = [(idx2word.get(i, "⟨oov⟩"), float(probs[i])) for i in top_ids]
        steps.append(top_preds)
        text += " " + top_preds[0][0]
    return text, steps

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-left">
    <div class="hero-tag">◈ NLP · BiLSTM · Attention · GloVe</div>
    <div class="hero-title">Lex<span>Domain</span></div>
    <div class="hero-sub">Domain-specific next-word prediction. Two models, two worlds — news vs. literature.</div>
  </div>
  <div class="hero-right">
    <div class="stat-block"><div class="stat-num">3.8M</div><div class="stat-label">News tokens</div></div>
    <div class="stat-block"><div class="stat-num">2.2M</div><div class="stat-label">Lit tokens</div></div>
    <div class="stat-block"><div class="stat-num">2.3M</div><div class="stat-label">Parameters</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="model-grid">
  <div class="model-card">
    <div class="model-dot dot-news"></div>
    <div><div class="model-name">News Model</div><div class="model-desc">AG News · 120k articles · formal register</div></div>
  </div>
  <div class="model-card">
    <div class="model-dot dot-lit"></div>
    <div><div class="model-name">Literature Model</div><div class="model-desc">Gutenberg · 18 novels · narrative prose</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

news_model, news_tok, gut_model, gut_tok, missing = load_models()
if missing:
    st.error("Missing model files: " + ", ".join(missing))
    st.info("Train models in Google Colab using `Next_Word_Prediction_Models.ipynb` and place the `.keras` and `.pickle` files here.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
with c1:
    seed = st.text_input("SEED TEXT", value="the government announced", placeholder="Enter seed text…")
with c2:
    n_words = st.number_input("WORDS", min_value=1, max_value=20, value=6)
with c3:
    top_k = st.number_input("TOP-K", min_value=2, max_value=8, value=4)
with c4:
    st.write("")  # spacer to align button
    predict_btn = st.button("Run →", type="primary", use_container_width=True)
    
# ── Results ───────────────────────────────────────────────────────────────────
if predict_btn:
    if not seed.strip():
        st.warning("Enter a seed phrase.")
        st.stop()

    with st.spinner(""):
        nr, ns = predict_next_words(seed, news_model, news_tok, n_words, top_k)
        gr, gs = predict_next_words(seed, gut_model,  gut_tok,  n_words, top_k)

    seed_clean = seed.lower().strip()
    news_gen   = nr[len(seed_clean):].strip()
    gut_gen    = gr[len(seed_clean):].strip()

    left, right = st.columns(2, gap="medium")

    def render_panel(col, label, accent_color, bar_color, seed_text, generated, steps):
        with col:
            html = f"""
            <div class="result-panel">
            <div class="result-header">
                <span class="result-label" style="color:{accent_color}">◈ {label}</span>
            </div>
            <div class="result-body">
                <div class="completion-text">
                {seed_text} <span style="color:{accent_color};font-weight:500">{generated}</span>
                </div>"""

            for i, preds in enumerate(steps):
                best = preds[0][0]
                max_prob = preds[0][1] if preds[0][1] > 0 else 1
                html += f"""
                <div class="step-row">
                <div class="step-header">
                    <span class="step-num">STEP {i+1}</span>
                    <span class="step-word">→ {best}</span>
                </div>"""
                for word, prob in preds:
                    pct = (prob / max_prob) * 100
                    is_chosen = word == best
                    word_style = f"color:#e8e8f0;font-weight:500" if is_chosen else "color:#6b6b80"
                    bar_opacity = "1" if is_chosen else "0.3"
                    html += f"""
                <div class="token-bar-wrap">
                    <span class="token-word" style="{word_style}">{word}</span>
                    <div class="token-bar-outer">
                    <div class="token-bar-inner" style="width:{pct:.1f}%;background:{bar_color};opacity:{bar_opacity}"></div>
                    </div>
                    <span class="token-pct">{prob:.1%}</span>
                </div>"""
                html += "</div>"

            html += "</div></div>"
            st.markdown(html, unsafe_allow_html=True)

    render_panel(left,  "News Model",       "#00d4ff", "#00d4ff", seed_clean, news_gen, ns)
    render_panel(right, "Literature Model", "#ff6b6b", "#ff6b6b", seed_clean, gut_gen,  gs)

# ── Architecture strip ────────────────────────────────────────────────────────
st.markdown("""
<div class="arch-strip">
  <div><div class="arch-layer">GloVe 100d</div><div class="arch-sublabel">EMBEDDINGS</div></div>
  <div class="arch-arrow">→</div>
  <div><div class="arch-layer">BiLSTM ×64</div><div class="arch-sublabel">ENCODER</div></div>
  <div class="arch-arrow">→</div>
  <div><div class="arch-layer">Self-Attention</div><div class="arch-sublabel">CONTEXT</div></div>
  <div class="arch-arrow">→</div>
  <div><div class="arch-layer">Avg Pooling</div><div class="arch-sublabel">AGGREGATE</div></div>
  <div class="arch-arrow">→</div>
  <div><div class="arch-layer">Dropout 0.5</div><div class="arch-sublabel">REGULARISE</div></div>
  <div class="arch-arrow">→</div>
  <div><div class="arch-layer">Dense 10k</div><div class="arch-sublabel">OUTPUT</div></div>
</div>
""", unsafe_allow_html=True)