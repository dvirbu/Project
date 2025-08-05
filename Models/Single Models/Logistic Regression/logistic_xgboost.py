# הרצת מודלים Logistic Regression + SGBoost
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize

fakenews_df = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv")
politifact_df = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv")

def apply_pca(X, n_components=100):
    return PCA(n_components=n_components).fit_transform(X)

def vectorize_text(df, method='tfidf'):
    texts = df['clean_text'].astype(str).tolist()
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=1000)
        return vectorizer.fit_transform(texts).toarray()
    tokenized = [word_tokenize(text) for text in texts]
    if method == 'word2vec':
        model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    elif method == 'fasttext':
        model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    return np.array([
        np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(100)], axis=0)
        for words in tokenized
    ])

def evaluate_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'xgboost':
        model = XGBClassifier(=eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

results = {}
models = ['logistic', 'xgboost']
methods = ['tfidf', 'word2vec', 'fasttext']
datasets = {
    'FakeNewsNet': fakenews_df,
    'PolitiFact': politifact_df
}

for model_name in models:
    for dataset_name, df in datasets.items():
        y = df['label'].values
        for method in methods:
            X = vectorize_text(df, method)
            for with_pca in [False, True]:
                X_used = apply_pca(X) if with_pca else X
                config = f"{dataset_name} - {model_name.upper()} - {method.upper()}{' + PCA' if with_pca else ''}"
                results[config] = evaluate_model(X_used, y, model_name)

# שמירת התוצאות
output_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\Model\logistic_xgboost_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n Results saved to {output_path}")
