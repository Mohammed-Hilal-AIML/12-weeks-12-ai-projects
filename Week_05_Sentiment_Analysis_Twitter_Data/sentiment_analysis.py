from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

BASE=Path(__file__).parent
DATA=BASE/"data/sentiment140.csv"
GLOVE=BASE/"embeddings/glove.6B.100d.txt"
OUT=BASE/"outputs"; MODELS=BASE/"models"
VOCAB=10000; MAXLEN=100; DIM=100; N=50000; SEED=42

def clean(s):
    s=str(s)
    s=re.sub(r"http\S+|www\.\S+"," ",s)
    s=re.sub(r"@\w+"," ",s).replace("#","")
    s=re.sub(r"&\w+;"," ",s)
    s=re.sub(r"[^\w\s.,!?;:'\"()\-\u00A9\u00AE\u203C\u2049\u2122\u2139\u2190-\u21FF\u2300-\u23FF\u25A0-\u27BF\uFE00-\uFE0F\U0001F300-\U0001FAFF]"," ",s)
    return re.sub(r"\s+"," ",s).strip().lower()

def load_data():
    if not DATA.exists(): raise FileNotFoundError(f"Missing {DATA}")
    c=["label","id","date","flag","user","text"]
    df=pd.read_csv(DATA,encoding="latin-1",header=None,names=c,nrows=N)
    df=df[df.label.isin([0,4])].copy()
    df["sentiment"]=(df.label==4).astype(int)
    df["clean_text"]=df.text.map(clean)
    df["date_parsed"]=pd.to_datetime(df.date,errors="coerce",utc=True)
    return df[df.clean_text.str.len()>0].reset_index(drop=True)

def glove_matrix(tok):
    if not GLOVE.exists(): raise FileNotFoundError(f"Missing {GLOVE}")
    emb={}
    with GLOVE.open(encoding="utf-8") as f:
        for line in f:
            v=line.rstrip().split(" "); emb[v[0]]=np.asarray(v[1:],dtype="float32")
    m=np.zeros((VOCAB,DIM),dtype="float32")
    for w,i in tok.word_index.items():
        if i<VOCAB and w in emb and len(emb[w])==DIM: m[i]=emb[w]
    return m

def main():
    OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    print("TensorFlow:",tf.__version__)
    df=load_data()
    print(f"Loaded {len(df):,} tweets")
    print(df.sentiment.value_counts().sort_index())

    xt,xv,yt,yv=train_test_split(df.clean_text,df.sentiment,test_size=.2,random_state=SEED,stratify=df.sentiment)
    tok=Tokenizer(num_words=VOCAB,oov_token="<OOV>"); tok.fit_on_texts(xt)
    xtr=pad_sequences(tok.texts_to_sequences(xt),maxlen=MAXLEN,padding="post",truncating="post")
    xte=pad_sequences(tok.texts_to_sequences(xv),maxlen=MAXLEN,padding="post",truncating="post")
    matrix=glove_matrix(tok)

    model=Sequential([Input(shape=(MAXLEN,)),Embedding(VOCAB,DIM,weights=[matrix],trainable=False),
                      SpatialDropout1D(.2),LSTM(128),Dense(64,activation="relu"),Dense(1,activation="sigmoid")])
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    model.summary()
    h=model.fit(xtr,yt,validation_split=.1,epochs=10,batch_size=128,
                callbacks=[ModelCheckpoint(MODELS/"best_lstm.keras",save_best_only=True)])
    model.save(MODELS/"final_lstm.keras")

    p=model.predict(xte,verbose=0).ravel()
    print("\nLSTM report:\n",classification_report(yv,(p>=.5).astype(int),target_names=["negative","positive"]))
    print("LSTM AUC:",roc_auc_score(yv,p))

    vec=TfidfVectorizer(max_features=10000,ngram_range=(1,2))
    a=vec.fit_transform(xt); b=vec.transform(xv)
    lr=LogisticRegression(max_iter=1000,random_state=SEED).fit(a,yt)
    bp=lr.predict_proba(b)[:,1]
    print("Baseline accuracy:",accuracy_score(yv,(bp>=.5).astype(int)))
    print("Baseline AUC:",roc_auc_score(yv,bp))

    f1,t1,_=roc_curve(yv,p); f2,t2,_=roc_curve(yv,bp)
    plt.figure(figsize=(8,6)); plt.plot(f1,t1,label=f"LSTM AUC={roc_auc_score(yv,p):.3f}")
    plt.plot(f2,t2,label=f"TF-IDF LR AUC={roc_auc_score(yv,bp):.3f}")
    plt.plot([0,1],[0,1],"--",label="Random"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"roc_curve.png",dpi=150); plt.close()

    pred=(p>=.5).astype(int); conf=np.where(pred==1,p,1-p); wrong=np.where(pred!=yv.to_numpy())[0]
    if len(wrong):
        rank=wrong[np.argsort(-conf[wrong])[:10]]
        pd.DataFrame({"tweet":xv.reset_index(drop=True).iloc[rank].values,
                      "true_label":yv.reset_index(drop=True).iloc[rank].map({0:"negative",1:"positive"}).values,
                      "predicted_label":pd.Series(pred[rank]).map({0:"negative",1:"positive"}).values,
                      "confidence":conf[rank]}).to_csv(OUT/"confidently_wrong_predictions.csv",index=False)

    days=df.dropna(subset=["date_parsed"]).copy(); days["day"]=days.date_parsed.dt.date
    daily=days.groupby("day").sentiment.mean().sort_index().iloc[:30]
    if len(daily):
        plt.figure(figsize=(10,5)); plt.plot(pd.to_datetime(daily.index),daily.values,marker="o")
        plt.ylim(0,1); plt.xlabel("Date"); plt.ylabel("Average positive sentiment")
        plt.title("30-Day Simulated Sentiment Trend"); plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(OUT/"30_day_sentiment_trend.png",dpi=150); plt.close()

    print("\nDone. See outputs/ and models/.")

if __name__=="__main__": main()
