import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec, FastText
from sklearn.preprocessing import StandardScaler


def vectorize_word2vec(texts, model, dim):
    vectors = []
    for tokens in texts.str.split():
        vec = np.mean([model.wv[w] for w in tokens if w in model.wv] or [np.zeros(dim)], axis=0)
        vectors.append(vec)
    return np.array(vectors)

def apply_pca(vectors, n=100):
    return PCA(n_components=n, random_state=42).fit_transform(vectors)

def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

def run_config(X_train, X_test, y_train, y_test, label):
    base_models = [
        ('svm', SVC(kernel='linear', probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
    stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)
    stack_model.fit(X_train, y_train)
    y_pred = stack_model.predict(X_test)
    return label, evaluate(y_test, y_pred)

fakenews = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv")
politifact = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv")

datasets = {
    "FakeNewsNet": fakenews,
    "PolitiFact": politifact
}

results = {}
for name, df in datasets.items():
    texts = df["clean_text"].astype(str).fillna("")
    labels = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

    # 1. TF-IDF
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    label, res = run_config(X_train_tfidf, X_test_tfidf, y_train, y_test, f"{name} - TF-IDF")
    results[label] = res

    # 2. TF-IDF + PCA
    X_train_pca = apply_pca(X_train_tfidf.toarray())
    X_test_pca = apply_pca(X_test_tfidf.toarray())
    label, res = run_config(X_train_pca, X_test_pca, y_train, y_test, f"{name} - TF-IDF + PCA")
    results[label] = res

    # 3. Word2Vec
    w2v_model = Word2Vec(sentences=[t.split() for t in X_train], vector_size=100, window=5, min_count=1, workers=4)
    X_train_w2v = vectorize_word2vec(X_train, w2v_model, 100)
    X_test_w2v = vectorize_word2vec(X_test, w2v_model, 100)
    label, res = run_config(X_train_w2v, X_test_w2v, y_train, y_test, f"{name} - Word2Vec")
    results[label] = res

    # 4. Word2Vec + PCA
    X_train_w2v_pca = apply_pca(X_train_w2v)
    X_test_w2v_pca = apply_pca(X_test_w2v)
    label, res = run_config(X_train_w2v_pca, X_test_w2v_pca, y_train, y_test, f"{name} - Word2Vec + PCA")
    results[label] = res

    # 5. FastText
    ft_model = FastText(sentences=[t.split() for t in X_train], vector_size=100, window=5, min_count=1, workers=4)
    X_train_ft = vectorize_word2vec(X_train, ft_model, 100)
    X_test_ft = vectorize_word2vec(X_test, ft_model, 100)
    label, res = run_config(X_train_ft, X_test_ft, y_train, y_test, f"{name} - FastText")
    results[label] = res

    # 6. FastText + PCA
    X_train_ft_pca = apply_pca(X_train_ft)
    X_test_ft_pca = apply_pca(X_test_ft)
    label, res = run_config(X_train_ft_pca, X_test_ft_pca, y_train, y_test, f"{name} - FastText + PCA")
    results[label] = res

# שמירת תוצאות
with open("stacking_svm_logistic_rf_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("Results saved to 'stacking_svm_logistic_rf_results.json'")
