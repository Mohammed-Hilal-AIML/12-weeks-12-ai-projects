import string
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevents script from hanging on plot window display
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

print("Starting execution...", flush=True)

# STEP 1: DOWNLOAD & LOAD DATASET
print("Step 1: Downloading NLTK resources and dataset...", flush=True)
nltk.download('stopwords', quiet=True)

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
print(f"Data Loaded Successfully! Total rows: {len(df)}", flush=True)

# STEP 2: EXPLORE DATA
print("\n--- Step 2: Data Exploration ---", flush=True)
print("Class Distribution:\n", df['label'].value_counts(), flush=True)

# STEP 3: CLEAN TEXT
print("\n--- Step 3: Cleaning Text ---", flush=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

df['cleaned_message'] = df['message'].apply(clean_text)
print("Text cleaning complete.", flush=True)

# STEP 4: FEATURE EXTRACTION
print("\n--- Step 4: Extracting Features ---", flush=True)
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_message']).toarray()
y = df['label'].map({'ham': 0, 'spam': 1})

# STEP 5: TRAIN MODELS
print("\n--- Step 5: Training Models ---", flush=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("Model training complete.", flush=True)

# STEP 6: EVALUATION & SAVE PLOT
print("\n--- Step 6: Evaluation ---", flush=True)
nb_preds = nb_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, nb_preds, target_names=['Ham', 'Spam']), flush=True)

# Save confusion matrix as image file instead of displaying GUI plot
cm = confusion_matrix(y_test, nb_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Spam Detector Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved matrix visualization to 'confusion_matrix.png'", flush=True)

# STEP 7: DEMO
print("\n--- Step 7: Running Demo Predictions ---", flush=True)

def predict_spam(raw_text):
    cleaned = clean_text(raw_text)
    vectorized = tfidf.transform([cleaned]).toarray()
    prediction = nb_model.predict(vectorized)[0]
    probabilities = nb_model.predict_proba(vectorized)[0]
    label = "SPAM" if prediction == 1 else "HAM"
    confidence = probabilities[prediction] * 100
    return label, confidence

custom_emails = [
    "URGENT! You have won a $1,000 Walmart gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch at 1 PM today?",
    "Congratulations! Your account has been selected for a cash prize.",
    "Please find attached the updated project report for Week 1.",
    "Free entry in a $100 weekly draw! Text WIN to 80082 now."
]

for email in custom_emails:
    result, score = predict_spam(email)
    print(f"Message: \"{email}\"")
    print(f"Prediction: {result} (Confidence: {score:.2f}%)\n", flush=True)

print("Project 1 Execution Finished Successfully!", flush=True)
