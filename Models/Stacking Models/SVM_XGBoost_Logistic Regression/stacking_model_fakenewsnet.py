# Stacking SVM_XGBoost_Logistic Regression
import os
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec, FastText

fakenews_path = (r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv")
politifact_path = (r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv")
output_path = "stacking_model_results.json"

def load_dataset(path):
    return pd.read_csv(path)

def get_word2vec_embeddings(texts):
    tokenized = [text.split() for text in texts]
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    return np.array([
        np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(100)], axis=0)
        for words in tokenized
    ])

def get_fasttext_embeddings(texts):
    tokenized = [text.split() for text in texts]
    model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    return np.array([
        np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(100)], axis=0)
        for words in tokenized
    ])


def get_stacking_model():
    estimators = [
        ('svm', SVC(kernel='linear', probability=True)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ('logreg', LogisticRegression(max_iter=1000))
    ]
    return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

def evaluate_model(X, y, use_pca=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if use_pca:
        pca = PCA(n_components=100)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    model = get_stacking_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

def run_all_evaluations(data, name_prefix):
    results = {}
    texts = data["clean_text"].astype(str).fillna("")
    labels = data["label"]

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_vecs = tfidf.fit_transform(texts).toarray()
    results[f"{name_prefix} - TF-IDF"] = evaluate_model(tfidf_vecs, labels)
    results[f"{name_prefix} - TF-IDF + PCA"] = evaluate_model(tfidf_vecs, labels, use_pca=True)

    # Word2Vec
    w2v_vecs = get_word2vec_embeddings(texts)
    results[f"{name_prefix} - Word2Vec"] = evaluate_model(w2v_vecs, labels)
    results[f"{name_prefix} - Word2Vec + PCA"] = evaluate_model(w2v_vecs, labels, use_pca=True)

    # FastText
    ft_vecs = get_fasttext_embeddings(texts)
    results[f"{name_prefix} - FastText"] = evaluate_model(ft_vecs, labels)
    results[f"{name_prefix} - FastText + PCA"] = evaluate_model(ft_vecs, labels, use_pca=True)

    return results

if __name__ == "__main__":
    results = {}
    fakenews = load_dataset(fakenews_path)
    politifact = load_dataset(politifact_path)

    if fakenews is not None:
        results.update(run_all_evaluations(fakenews, "FakeNewsNet"))
    if politifact is not None:
        results.update(run_all_evaluations(politifact, "PolitiFact"))


    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {output_path}")
