import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

fakenews_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv"
politifact_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv"
output_path = "emotion_onehot_analysis_stacking_all.csv"

emotion_categories = ['joy', 'surprise', 'anger', 'disgust', 'fear', 'sadness']
emotion_groups = {
    'positive': ['joy', 'surprise'],
    'negative': ['anger', 'disgust', 'fear', 'sadness'],
    'no_neutral': ['joy', 'surprise', 'anger', 'disgust', 'fear', 'sadness']
}

def vectorize_texts(texts, model, dim=100):
    vectors = []
    for text in texts:
        words = word_tokenize(text)
        word_vecs = [model.wv[word] for word in words if word in model.wv]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(dim))
    return np.array(vectors)

try:
    fakenews_df = pd.read_csv(fakenews_path)
    politifact_df = pd.read_csv(politifact_path)
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    exit()

results = []

for dataset_name, df in {"FakeNewsNet": fakenews_df, "PolitiFact": politifact_df}.items():
    print(f" Processing dataset: {dataset_name}")

    df = df.dropna(subset=['clean_text', 'label', 'emotion'])
    df['clean_text'] = df['clean_text'].astype(str)

    texts = df['clean_text']
    labels = df['label']
    emotions = df['emotion']

    # אימון Word2Vec
    w2v_model = Word2Vec(sentences=[text.split() for text in texts], vector_size=100, window=5, min_count=1, workers=4)
    X_vectors = vectorize_texts(texts, w2v_model)
    y = labels.values

    config_options = emotion_categories + list(emotion_groups.keys()) + ['all_emotions']

    for category in config_options:
        print(f" Running configuration: {category}")

        if category in emotion_categories:
            df[f"emotion_{category}"] = (emotions == category).astype(int)
            emotion_feature = df[f"emotion_{category}"].values.reshape(-1, 1)

        elif category in emotion_groups:
            df[f"emotion_{category}"] = emotions.isin(emotion_groups[category]).astype(int)
            emotion_feature = df[f"emotion_{category}"].values.reshape(-1, 1)

        elif category == "all_emotions":
            emotion_feature = pd.get_dummies(emotions).values  # one-hot מלא

        else:
            continue

        # שילוב עם וקטורי טקסט
        X_combined = np.hstack((X_vectors, emotion_feature))

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        all_preds, all_trues = [], []

        for train_idx, test_idx in skf.split(X_combined, y):
            X_train, X_test = X_combined[train_idx], X_combined[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            base_learners = [
                ('svm', SVC(kernel='linear', probability=True, random_state=42)),
                ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
            ]

            model = StackingClassifier(
                estimators=base_learners,
                final_estimator=LogisticRegression(random_state=42),
                cv=3
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            all_preds.extend(preds)
            all_trues.extend(y_test)

        report = classification_report(all_trues, all_preds, output_dict=True, zero_division=0)
        results.append({
            "Dataset": dataset_name,
            "Emotion_Feature": category,
            "Accuracy": report['accuracy'],
            "Precision": report['weighted avg']['precision'],
            "Recall": report['weighted avg']['recall'],
            "F1-Score": report['weighted avg']['f1-score']
        })

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"✅ Results saved to: {output_path}")
