import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
    return PCA(n_components=n, random_state=42)

def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

def run_config(X_train_vec, X_test_vec, y_train, y_test):
    base_models = [
        ('svm', SVC(kernel='linear', probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]

    stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5,n_jobs=-1)
    stack_model.fit(X_train_vec, y_train)
    y_pred = stack_model.predict(X_test_vec)
    return evaluate(y_test, y_pred)


fakenews = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv")
politifact = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv")

datasets = {
    "FakeNewsNet": fakenews,
    "PolitiFact": politifact
}

results = {}
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for name, df in datasets.items():
    texts = df["clean_text"].astype(str).fillna("")
    labels = df["label"]

    print(f"\n Running Cross-Validation for {name} with {n_splits} folds ")

    tfidf_scores = []
    tfidf_pca_scores = []
    w2v_scores = []
    w2v_pca_scores = []
    ft_scores = []
    ft_pca_scores = []

    # לולאת Cross-Validation
    for fold, (train_index, test_index) in enumerate(skf.split(texts, labels)):
        print(f"\n {name} - Fold {fold + 1}/{n_splits} ")
        X_train_fold, X_test_fold = texts.iloc[train_index], texts.iloc[test_index]
        y_train_fold, y_test_fold = labels.iloc[train_index], labels.iloc[test_index]

        # 1. TF-IDF
        print(f"{name} - Fold {fold + 1}: TF-IDF Vectorization ")
        tfidf = TfidfVectorizer(max_features=3000)
        X_train_tfidf = tfidf.fit_transform(X_train_fold)
        X_test_tfidf = tfidf.transform(X_test_fold)
        tfidf_scores.append(run_config(X_train_tfidf, X_test_tfidf, y_train_fold, y_test_fold))

        # 2. TF-IDF + PCA
        print(f"{name} - Fold {fold + 1}: TF-IDF + PCA ")
        pca_tfidf = apply_pca(X_train_tfidf.toarray(), n=100)
        X_train_pca_tfidf = pca_tfidf.fit_transform(X_train_tfidf.toarray())
        X_test_pca_tfidf = pca_tfidf.transform(X_test_tfidf.toarray())
        tfidf_pca_scores.append(run_config(X_train_pca_tfidf, X_test_pca_tfidf, y_train_fold, y_test_fold))

        # 3. Word2Vec
        print(f"{name} - Fold {fold + 1}: Word2Vec Vectorization ")
        w2v_model = Word2Vec(sentences=[t.split() for t in X_train_fold], vector_size=100, window=5, min_count=1,workers=4)
        X_train_w2v = vectorize_word2vec(X_train_fold, w2v_model, 100)
        X_test_w2v = vectorize_word2vec(X_test_fold, w2v_model, 100)
        scaler_w2v = StandardScaler()
        X_train_w2v_scaled = scaler_w2v.fit_transform(X_train_w2v)
        X_test_w2v_scaled = scaler_w2v.transform(X_test_w2v)
        w2v_scores.append(run_config(X_train_w2v_scaled, X_test_w2v_scaled, y_train_fold, y_test_fold))

        # 4. Word2Vec + PCA
        print(f"{name} - Fold {fold + 1}: Word2Vec + PCA ")
        pca_w2v = apply_pca(X_train_w2v_scaled, n=100)
        X_train_pca_w2v = pca_w2v.fit_transform(X_train_w2v_scaled)
        X_test_pca_w2v = pca_w2v.transform(X_test_w2v_scaled)
        w2v_pca_scores.append(run_config(X_train_pca_w2v, X_test_pca_w2v, y_train_fold, y_test_fold))

        # 5. FastText
        print(f"{name} - Fold {fold + 1}: FastText Vectorization ")
        ft_model = FastText(sentences=[t.split() for t in X_train_fold], vector_size=100, window=5, min_count=1,workers=4)
        X_train_ft = vectorize_word2vec(X_train_fold, ft_model, 100)
        X_test_ft = vectorize_word2vec(X_test_fold, ft_model, 100)
        scaler_ft = StandardScaler()
        X_train_ft_scaled = scaler_ft.fit_transform(X_train_ft)
        X_test_ft_scaled = scaler_ft.transform(X_test_ft)
        ft_scores.append(run_config(X_train_ft_scaled, X_test_ft_scaled, y_train_fold, y_test_fold))

        # 6. FastText + PCA
        print(f"{name} - Fold {fold + 1}: FastText + PCA...")
        pca_ft = apply_pca(X_train_ft_scaled, n=100)
        X_train_pca_ft = pca_ft.fit_transform(X_train_ft_scaled)
        X_test_pca_ft = pca_ft.transform(X_test_ft_scaled)
        ft_pca_scores.append(run_config(X_train_pca_ft, X_test_pca_ft, y_train_fold, y_test_fold))

    # חישוב ממוצע וסטיית תקן עבור כל קונפיגורציה
    def calculate_mean_std_scores(scores_list):
        if not scores_list:
            return {}

        metrics = {k: [dic[k] for dic in scores_list] for k in scores_list[0]}

        mean_std_scores = {}
        for metric, values in metrics.items():
            mean_std_scores[f"{metric}_mean"] = np.mean(values)
            mean_std_scores[f"{metric}_std"] = np.std(values)
        return mean_std_scores


    results[f"{name} - TF-IDF"] = calculate_mean_std_scores(tfidf_scores)
    results[f"{name} - TF-IDF + PCA"] = calculate_mean_std_scores(tfidf_pca_scores)
    results[f"{name} - Word2Vec"] = calculate_mean_std_scores(w2v_scores)
    results[f"{name} - Word2Vec + PCA"] = calculate_mean_std_scores(w2v_pca_scores)
    results[f"{name} - FastText"] = calculate_mean_std_scores(ft_scores)
    results[f"{name} - FastText + PCA"] = calculate_mean_std_scores(ft_pca_scores)

# הדפסת תוצאות סופיות לסיכום
print("\n Final Cross-Validation Results ")
for config_name, scores in results.items():
    print(f"\n{config_name}:")
    for metric, value in scores.items():
        if '_mean' in metric:
            std_metric = metric.replace('_mean', '_std')
            if std_metric in scores:
                print(f"  {metric.replace('_mean', '').capitalize()}: {value:.4f} (Std: {scores[std_metric]:.4f})")
            else:
                print(f"  {metric.replace('_mean', '').capitalize()}: {value:.4f}")
        elif '_std' not in metric:
            print(f"  {metric.capitalize()}: {value:.4f}")

# שמירת תוצאות לקובץ JSON
with open("stacking_svm_logistic_rf_cv_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("\nResults saved to 'stacking_svm_logistic_rf_cv_results.json'")