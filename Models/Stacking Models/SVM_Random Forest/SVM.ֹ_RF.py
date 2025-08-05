
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import nltk
from nltk.tokenize import word_tokenize


def get_word2vec_embeddings(texts, method='word2vec'):
    tokenized = [word_tokenize(text) for text in texts]
    if method == 'word2vec':
        model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    elif method == 'fasttext':
        model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    else:
        raise ValueError("Method must be 'word2vec' or 'fasttext'.")

    vectors = []
    for tokens in tokenized:
        word_vecs = [model.wv[token] for token in tokens if token in model.wv]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)

def run_stacking_model(X, y, dataset_name, method_name, results_dict):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results_dict[f"{dataset_name} - {method_name}"] = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }

results = {}

for dataset_name, path in {
    'FakeNewsNet': r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv",
    'PolitiFact': r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv"
}.items():
    df = pd.read_csv(path).dropna()
    texts = df['clean_text']
    labels = df['label']

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(texts).toarray()
    run_stacking_model(X_tfidf, labels, dataset_name, "TF-IDF", results)

    # TF-IDF + PCA
    X_tfidf_pca = PCA(n_components=100).fit_transform(X_tfidf)
    run_stacking_model(X_tfidf_pca, labels, dataset_name, "TF-IDF + PCA", results)

    # Word2Vec
    X_w2v = get_word2vec_embeddings(texts, method='word2vec')
    run_stacking_model(X_w2v, labels, dataset_name, "Word2Vec", results)

    # Word2Vec + PCA
    X_w2v_pca = PCA(n_components=50).fit_transform(X_w2v)
    run_stacking_model(X_w2v_pca, labels, dataset_name, "Word2Vec + PCA", results)

    # FastText
    X_fasttext = get_word2vec_embeddings(texts, method='fasttext')
    run_stacking_model(X_fasttext, labels, dataset_name, "FastText", results)

    # FastText + PCA
    X_fasttext_pca = PCA(n_components=50).fit_transform(X_fasttext)
    run_stacking_model(X_fasttext_pca, labels, dataset_name, "FastText + PCA", results)

with open("stacking_comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)
