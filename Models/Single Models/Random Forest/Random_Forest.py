# Random Forest הרצת מודל
import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText

RESULTS = {}

# הרצת מודל
def run_rf_model(X, y, config_name, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = report['accuracy']
    precision = np.mean([report['0']['precision'], report['1']['precision']])
    recall = np.mean([report['0']['recall'], report['1']['recall']])
    f1 = np.mean([report['0']['f1-score'], report['1']['f1-score']])

    RESULTS[f"{dataset_name} - {config_name}"] = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

fakenews = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv")
politifact = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv")

datasets = {
    "FakeNewsNet": fakenews,
    "PolitiFact": politifact
}

for name, df in datasets.items():
    print(f"\n🔍 Processing dataset: {name}")
    texts = df['clean_text'].astype(str).tolist()
    labels = df['label'].tolist()
    tokenized = [preprocess(t) for t in texts]
    clean_texts = tokens_to_text(tokenized)

    # 1. TF-IDF
    tfidf_vec = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf_vec.fit_transform(clean_texts).toarray()
    run_rf_model(X_tfidf, labels, "TF-IDF", name)

    # 2. TF-IDF + PCA
    X_tfidf_pca = apply_pca(X_tfidf)
    run_rf_model(X_tfidf_pca, labels, "TF-IDF + PCA", name)

    # 3. Word2Vec
    w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4, seed=42)
    X_w2v = vectorize_w2v(w2v_model, tokenized, 100)
    run_rf_model(X_w2v, labels, "Word2Vec", name)

    # 4. Word2Vec + PCA
    X_w2v_pca = apply_pca(X_w2v)
    run_rf_model(X_w2v_pca, labels, "Word2Vec + PCA", name)

    # 5. FastText
    ft_model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4, seed=42)
    X_ft = vectorize_w2v(ft_model, tokenized, 100)
    run_rf_model(X_ft, labels, "FastText", name)

    # 6. FastText + PCA
    X_ft_pca = apply_pca(X_ft)
    run_rf_model(X_ft_pca, labels, "FastText + PCA", name)

# שמירת תוצאות
with open("rf_comparison_results.json", "w") as f:
    json.dump(RESULTS, f, indent=4)

print("\n Results saved to rf_comparison_results.json")
