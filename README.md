# Next-Word Prediction: Domain-Specific Language Models

Two next-word prediction models trained on different text domains — **news articles** and **classic literature** — to explore how domain-specific language patterns affect word prediction.

---

## Demo

```
Seed: "the government announced"

  [News     ] the government announced the new policy plan for
               step 1: the (34.2%), a (12.1%), its (8.7%)
               step 2: new (28.4%), proposed (11.2%), federal (9.3%)

  [Gutenberg] the government announced that it would not be
               step 1: that (41.2%), what (9.8%), he (7.1%)
               step 2: it (22.3%), the (18.4%), she (11.2%)
```

---

## Models

| Model | Dataset | Domain | Vocab |
|---|---|---|---|
| `news_model` | [AG News](https://huggingface.co/datasets/ag_news) | Modern news writing | 10,000 |
| `gutenberg_model` | [NLTK Gutenberg](https://www.nltk.org/book/ch02.html) | Classic literature (18 books) | 10,000 |

---

## Architecture

Each model uses the same **BiLSTM + Self-Attention** stack with pre-trained GloVe embeddings:

```
Input (seq_len=20)
    │
    ▼
GloVe Embedding (100d, fine-tuned)
    │
    ▼
Bidirectional LSTM (64 units each direction → 128 total)
    │
    ▼
Self-Attention
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout (0.5)
    │
    ▼
Dense (vocab_size=10,000, softmax)
```

- **GloVe 100d** — pre-trained word embeddings to initialise with real-world semantics
- **Bidirectional LSTM** — processes the input sequence in both directions for richer context
- **Self-Attention** — focuses on the most relevant tokens in the context window
- **Dropout** — regularisation to reduce overfitting

---

## Project Structure

```
├── Next_Word_Prediction_Models.ipynb   # Training notebook (run in Google Colab)
├── app.py                              # Streamlit demo app
├── requirements.txt                    # Python dependencies for the app
└── README.md
```

> **Note:** The trained model files (`.keras`) and tokenizer files (`.pickle`) are not included due to file size. Run the notebook to generate them, then place them in the root directory before running the app.

---

## Quickstart

### 1. Train the models (Google Colab recommended)

Open `Next_Word_Prediction_Models.ipynb` in [Google Colab](https://colab.research.google.com/), enable a **T4 GPU** (`Runtime → Change runtime type → T4 GPU`), and run all cells. Download the four output files:

```
news_model.keras
news_model_tokenizer.pickle
gutenberg_model.keras
gutenberg_model_tokenizer.pickle
```

### 2. Run the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Requirements

```
tensorflow>=2.15
tensorflow-datasets
nltk
streamlit
numpy
matplotlib
```

---

## Key Design Decisions

**Why two domains?**
Different text domains have distinct vocabulary distributions and sentence structures. News text is concise and factual; literary text is narrative and expressive. Training on both lets us compare how domain-specific patterns are captured by the same architecture.

**Why BiLSTM + Attention?**
A BiLSTM reads the context window in both directions, which helps the model pick up on longer-range dependencies. The self-attention layer then re-weights the LSTM outputs to focus on the most informative positions — a lightweight alternative to a full Transformer for this task size.

**Why GloVe over random initialisation?**
Starting from pre-trained embeddings gives the model a head start with real-world word semantics, which reduces the amount of data and training time needed to reach reasonable performance.

**Speed optimisations in training**
- Native TF tensor slicing instead of `tf.py_function` (avoids Python GIL overhead per batch)
- GloVe 100d instead of 300d — 3x smaller matrix, negligible quality loss for this task
- Vocab size 10k — the output Dense layer is the main compute bottleneck; halving from 20k roughly halves its cost
- Batch size 256 for better GPU utilisation

---

## Results

Training converges in around 5–8 epochs on both corpora. The models learn domain-specific patterns clearly — the news model favours formal, factual continuations while the Gutenberg model leans toward narrative and dialogue structures.
