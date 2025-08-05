import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from gensim.models import Word2Vec, FastText
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, preds, average="weighted", zero_division=0)
    }

def apply_pca(X, n_components=100):
    return PCA(n_components=n_components).fit_transform(X)

def get_word_embeddings(texts, method="word2vec"):
    tokens = [text.split() for text in texts]
    if method == "word2vec":
        model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4)
    else:
        model = FastText(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4)
    vectors = []
    for sentence in tokens:
        vecs = [model.wv[word] for word in sentence if word in model.wv]
        vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(100))
    return np.array(vectors)

def run_all_configs(df, dataset_name):
    results = {}
    df = df.dropna(subset=["clean_text", "label"])
    texts = df["clean_text"].astype(str).values
    labels = df["label"].values
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    feature_extractors = {
        "TF-IDF": lambda x: TfidfVectorizer(max_features=1000).fit_transform(x).toarray(),
        "Word2Vec": lambda x: get_word_embeddings(x, method="word2vec"),
        "FastText": lambda x: get_word_embeddings(x, method="fasttext")
    }

    for name, extractor in feature_extractors.items():
        X_train = extractor(X_train_texts)
        X_test = extractor(X_test_texts)

        for use_pca in [False, True]:
            suffix = f"{name} + PCA" if use_pca else name
            if use_pca:
                X_train_proc = apply_pca(X_train)
                X_test_proc = apply_pca(X_test)
            else:
                X_train_proc = X_train
                X_test_proc = X_test

            base_models = [
                ("svm", SVC(probability=True)),
                ("rf", RandomForestClassifier()),
                ("lr", LogisticRegression(max_iter=1000))
            ]
            meta_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            stack = StackingClassifier(estimators=base_models, final_estimator=meta_model)

            scores = evaluate_model(stack, X_train_proc, X_test_proc, y_train, y_test)
            results[f"{dataset_name} - {suffix}"] = scores

    return results

# נתיבים לקבצים
fakenews_path = (r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv")
politifact_path = (r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv")

# קריאה והרצה
fakenews_df = pd.read_csv(fakenews_path)
politifact_df = pd.read_csv(politifact_path)

final_results = {}
final_results.update(run_all_configs(fakenews_df, "FakeNewsNet"))
final_results.update(run_all_configs(politifact_df, "PolitiFact"))

# שמירת תוצאות
with open("stacking_svm_rf_lr_xgb_results.json", "w") as f:
    json.dump(final_results, f, indent=4)
