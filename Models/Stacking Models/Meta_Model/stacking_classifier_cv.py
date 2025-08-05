import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from gensim.models import Word2Vec, FastText
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, preds, average="weighted", zero_division=0)
    }


def get_word_embeddings(texts, model, dim):
    vectors = []
    for tokens in texts:
        vec = np.mean([model.wv[w] for w in tokens if w in model.wv] or [np.zeros(dim)], axis=0)
        vectors.append(vec)
    return np.array(vectors)


# נתיבים לקבצים
fakenews_path = (r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv")
politifact_path = (r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv")

# קריאה והרצה
fakenews_df = pd.read_csv(fakenews_path)
politifact_df = pd.read_csv(politifact_path)

datasets = {
    "FakeNewsNet": fakenews_df,
    "PolitiFact": politifact_df
}

final_results = {}
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for dataset_name, df in datasets.items():
    df = df.dropna(subset=["clean_text", "label"])
    texts = df["clean_text"].astype(str).values
    labels = df["label"].values

    print(f"\n--- Running Cross-Validation for {dataset_name} with {n_splits} folds ---")

    config_scores = {
        "TF-IDF": [],
        "TF-IDF + PCA": [],
        "Word2Vec": [],
        "Word2Vec + PCA": [],
        "FastText": [],
        "FastText + PCA": []
    }

    for fold, (train_index, test_index) in enumerate(skf.split(texts, labels)):
        print(f"\n {dataset_name} - Fold {fold + 1}/{n_splits} ")
        X_train_fold_texts, X_test_fold_texts = texts[train_index], texts[test_index]
        y_train_fold, y_test_fold = labels[train_index], labels[test_index]

        # 1. TF-IDF
        print(f" TF-IDF Vectorization ")
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_fold_texts).toarray()
        X_test_tfidf = tfidf_vectorizer.transform(X_test_fold_texts).toarray()

        # Run TF-IDF without PCA
        base_models_tfidf = [("svm", SVC(probability=True, random_state=42)),
                             ("rf", RandomForestClassifier(random_state=42)),
                             ("lr", LogisticRegression(max_iter=1000, random_state=42))]
        meta_model_tfidf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        stack_tfidf = StackingClassifier(estimators=base_models_tfidf, final_estimator=meta_model_tfidf, cv=5,
                                         n_jobs=-1)
        config_scores["TF-IDF"].append(
            evaluate_model(stack_tfidf, X_train_tfidf, X_test_tfidf, y_train_fold, y_test_fold))

        # 2. TF-IDF + PCA
        print(f" TF-IDF + PCA ")
        pca_tfidf = PCA(n_components=100, random_state=42)
        X_train_tfidf_pca = pca_tfidf.fit_transform(X_train_tfidf)
        X_test_tfidf_pca = pca_tfidf.transform(X_test_tfidf)

        base_models_tfidf_pca = [("svm", SVC(probability=True, random_state=42)),
                                 ("rf", RandomForestClassifier(random_state=42)),
                                 ("lr", LogisticRegression(max_iter=1000, random_state=42))]
        meta_model_tfidf_pca = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        stack_tfidf_pca = StackingClassifier(estimators=base_models_tfidf_pca, final_estimator=meta_model_tfidf_pca,
                                             cv=5, n_jobs=-1)
        config_scores["TF-IDF + PCA"].append(
            evaluate_model(stack_tfidf_pca, X_train_tfidf_pca, X_test_tfidf_pca, y_train_fold, y_test_fold))

        # 3. Word2Vec
        print(f" Word2Vec Vectorization")
        w2v_model = Word2Vec(sentences=[t.split() for t in X_train_fold_texts], vector_size=100, window=5, min_count=1,
                             workers=4)
        X_train_w2v = get_word_embeddings([t.split() for t in X_train_fold_texts], w2v_model,
                                          100)  # Pass tokenized texts
        X_test_w2v = get_word_embeddings([t.split() for t in X_test_fold_texts], w2v_model, 100)  # Pass tokenized texts

        scaler_w2v = StandardScaler()
        X_train_w2v_scaled = scaler_w2v.fit_transform(X_train_w2v)
        X_test_w2v_scaled = scaler_w2v.transform(X_test_w2v)

        base_models_w2v = [("svm", SVC(probability=True, random_state=42)),
                           ("rf", RandomForestClassifier(random_state=42)),
                           ("lr", LogisticRegression(max_iter=1000, random_state=42))]
        meta_model_w2v = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        stack_w2v = StackingClassifier(estimators=base_models_w2v, final_estimator=meta_model_w2v, cv=5, n_jobs=-1)
        config_scores["Word2Vec"].append(
            evaluate_model(stack_w2v, X_train_w2v_scaled, X_test_w2v_scaled, y_train_fold, y_test_fold))

        # 4. Word2Vec + PCA
        print(f" Word2Vec + PCA ")
        pca_w2v = PCA(n_components=100, random_state=42)
        X_train_w2v_pca = pca_w2v.fit_transform(X_train_w2v_scaled)
        X_test_w2v_pca = pca_w2v.transform(X_test_w2v_scaled)

        base_models_w2v_pca = [("svm", SVC(probability=True, random_state=42)),
                               ("rf", RandomForestClassifier(random_state=42)),
                               ("lr", LogisticRegression(max_iter=1000, random_state=42))]
        meta_model_w2v_pca = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        # Fix: Removed cv='prefit'
        stack_w2v_pca = StackingClassifier(estimators=base_models_w2v_pca, final_estimator=meta_model_w2v_pca, cv=5,
                                           n_jobs=-1)
        config_scores["Word2Vec + PCA"].append(
            evaluate_model(stack_w2v_pca, X_train_w2v_pca, X_test_w2v_pca, y_train_fold, y_test_fold))

        # 5. FastText
        print(f" FastText Vectorization ")
        ft_model = FastText(sentences=[t.split() for t in X_train_fold_texts], vector_size=100, window=5, min_count=1,
                            workers=4)
        X_train_ft = get_word_embeddings([t.split() for t in X_train_fold_texts], ft_model, 100)  # Pass tokenized texts
        X_test_ft = get_word_embeddings([t.split() for t in X_test_fold_texts], ft_model, 100)  # Pass tokenized texts

        scaler_ft = StandardScaler()
        X_train_ft_scaled = scaler_ft.fit_transform(X_train_ft)
        X_test_ft_scaled = scaler_ft.transform(X_test_ft)

        base_models_ft = [("svm", SVC(probability=True, random_state=42)),
                          ("rf", RandomForestClassifier(random_state=42)),
                          ("lr", LogisticRegression(max_iter=1000, random_state=42))]
        meta_model_ft = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        stack_ft = StackingClassifier(estimators=base_models_ft, final_estimator=meta_model_ft, cv=5, n_jobs=-1)
        config_scores["FastText"].append(
            evaluate_model(stack_ft, X_train_ft_scaled, X_test_ft_scaled, y_train_fold, y_test_fold))

        # 6. FastText + PCA
        print(f" FastText + PCA ")
        pca_ft = PCA(n_components=100, random_state=42)
        X_train_ft_pca = pca_ft.fit_transform(X_train_ft_scaled)
        X_test_ft_pca = pca_ft.transform(X_test_ft_scaled)

        base_models_ft_pca = [("svm", SVC(probability=True, random_state=42)),
                              ("rf", RandomForestClassifier(random_state=42)),
                              ("lr", LogisticRegression(max_iter=1000, random_state=42))]
        meta_model_ft_pca = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        stack_ft_pca = StackingClassifier(estimators=base_models_ft_pca, final_estimator=meta_model_ft_pca, cv=5,
                                          n_jobs=-1)
        config_scores["FastText + PCA"].append(
            evaluate_model(stack_ft_pca, X_train_ft_pca, X_test_ft_pca, y_train_fold, y_test_fold))

    # חישוב ממוצע וסטיית תקן עבור כל קונפיגורציה
    for config_name, scores_list in config_scores.items():
        if not scores_list:
            final_results[f"{dataset_name} - {config_name}"] = {}
            continue

        metrics = {k: [dic[k] for dic in scores_list] for k in scores_list[0]}

        mean_std_scores = {}
        for metric, values in metrics.items():
            mean_std_scores[f"{metric}_mean"] = np.mean(values)
            mean_std_scores[f"{metric}_std"] = np.std(values)
        final_results[f"{dataset_name} - {config_name}"] = mean_std_scores

# הדפסת תוצאות סופיות לסיכום
print("\n Final Cross-Validation Results ")
for config_name, scores in final_results.items():
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
with open("stacking_svm_rf_lr_xgb_cv_results.json", "w") as f:
    json.dump(final_results, f, indent=4)
print("\nResults saved to 'stacking_svm_rf_lr_xgb_cv_results.json'")