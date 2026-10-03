# Sentiment Analysis on Product Reviews 🔍

Fine-tuned BERT (bert-base-uncased) on 40,000 Amazon Polarity reviews to classify sentiment as positive or negative. Includes an interactive Streamlit web app.

## Results
Evaluated on 10,000 held-out reviews from the Amazon Polarity test set.
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 93.2%  |
| Precision | 93.5%  |
| Recall    | 93.9%  |
| F1-Score  | 93.2%  |

## Project Structure

```
sentiment-analysis/
├── train.py          # Fine-tuning script (BERT + HuggingFace Trainer API)
├── app.py            # Streamlit inference app
├── requirements.txt  # Dependencies
└── model/            # Saved fine-tuned model (generated after training)
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This downloads the `amazon_polarity` dataset, fine-tunes BERT for 3 epochs, and saves the model to `./model/`.

> GPU recommended. On CPU, reduce `BATCH_SIZE` to 8 in `train.py`.

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## Tech Stack

- **Model:** `bert-base-uncased` (HuggingFace Transformers)
- **Dataset:** Amazon Polarity (via HuggingFace Datasets)
- **Training:** HuggingFace `Trainer` API with fp16 support
- **Frontend:** Streamlit
- **Libraries:** PyTorch, Scikit-learn, NumPy

## How It Works

1. Raw review text is tokenized using `BertTokenizer` (max 128 tokens)
2. Token embeddings pass through BERT's 12 transformer layers
3. The `[CLS]` token representation is fed into a linear classification head
4. Softmax gives probability scores for Negative / Positive classes

## Author

**Kunaljit Das** — B.Tech CSE, The Assam Kaziranga University  
[LinkedIn](https://linkedin.com/in/kunaljit-das) · [GitHub](https://github.com/kunaljitdas)
