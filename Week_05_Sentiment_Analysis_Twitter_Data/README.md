# Week 05 — Sentiment Analysis on Twitter Data

Intermediate NLP project using Sentiment140, GloVe 100d embeddings, an LSTM, a TF-IDF + LogisticRegression baseline, ROC/AUC evaluation, and a 30-day sentiment trend.

## GitHub name
`Week_05_Sentiment_Analysis_Twitter_Data`

## GitHub description (under 200 characters)
NLP sentiment analysis using Sentiment140, GloVe 100d embeddings, an LSTM model, TF-IDF baseline, ROC/AUC evaluation, and a 30-day sentiment trend dashboard.

## Requirements
**Python 3.12 recommended.** Install:
```bash
py -3.12 -m pip install -r requirements.txt
```

## Dataset
The curriculum specifies Sentiment140 (about 1.6M tweets) and loading 50,000 tweets. Place the CSV at:
`data/sentiment140.csv`

Expected columns:
`label, id, date, flag, user, text`

Labels are mapped as required by the execution steps:
- 0 = negative
- 4 = positive
- 0/4 -> 0/1

## GloVe
Download GloVe 100d and place:
`embeddings/glove.6B.100d.txt`

## Preprocessing
Removes URLs, @mentions, hashtag symbols and HTML entities; keeps emoji characters; cleans whitespace and text case.

## Tokenization
Keras Tokenizer:
- vocabulary size: 10,000
- maximum sequence length: 100
- padded/truncated sequences

## LSTM
```text
Embedding(GloVe, trainable=False)
        ↓
SpatialDropout1D(0.2)
        ↓
LSTM(128)
        ↓
Dense(64, relu)
        ↓
Dense(1, sigmoid)
```
Train for 10 epochs.

## Evaluation
Compares the LSTM with TF-IDF + LogisticRegression, plots ROC curves, prints AUC, and saves confidently wrong predictions.

## Trend Dashboard
Simulates a 30-day tweet stream from dataset dates, plots daily average positive sentiment, and saves example tweets for high/low sentiment spikes.

## Structure
```text
Week_05_Sentiment_Analysis_Twitter_Data/
├── sentiment_analysis.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── embeddings/
├── models/
└── outputs/
```

## Curriculum checklist
- Sentiment140
- 50,000 tweets
- Label distribution and 0/4 -> 0/1
- Regex cleaning
- Emoji preservation
- Tokenizer vocab_size=10000
- max_length=100
- GloVe 100d
- Frozen embedding
- SpatialDropout1D(0.2)
- LSTM(128)
- Dense(64, relu)
- Dense(1, sigmoid)
- 10 epochs
- TF-IDF + LogisticRegression baseline
- ROC and AUC
- Confidently wrong predictions
- 30-day sentiment trend

## Source alignment
The Week 05 overview mentions Positive/Negative/Neutral, but its detailed execution steps specify Sentiment140 labels 0 and 4, relabeling to 0/1, and a sigmoid output. This repository follows those detailed execution steps. 
