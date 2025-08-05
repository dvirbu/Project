import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize

try:
    english_vocab = set(words.words())
except Exception as e:
    print(f"Error loading English vocabulary: {e}")
    english_vocab = set()


def avg_spelling_errors(text_series):

    def count_errors(text):
        if not isinstance(text, str) or not text:
            return 0
        tokens = word_tokenize(text.lower())
        if not tokens:
            return 0
        errors = sum(1 for word in tokens if word.isalpha() and word not in english_vocab)
        return errors / len(tokens)

    return text_series.apply(count_errors)


def evaluate_model(X, y, model, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    print(f"  > Starting cross-validation with {n_splits} splits...")
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        try:
            print(f"    - Training on split {i + 1}/{n_splits}...")
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])

            metrics['accuracy'].append(accuracy_score(y[test_idx], y_pred))
            metrics['precision'].append(precision_score(y[test_idx], y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y[test_idx], y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y[test_idx], y_pred, zero_division=0))

        except Exception as e:
            print(f"  - Error during split {i + 1}: {e}")
            continue

    if not metrics['accuracy']:
        print("  > No successful splits were completed.")
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    return {k: float(np.mean(v)) for k, v in metrics.items()}

def get_stacking_model():
    base_models = [
        ('svm', SVC(kernel='linear', probability=True, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
    return StackingClassifier(estimators=base_models,
                              final_estimator=LogisticRegression(),
                              cv=3, n_jobs=-1)


datasets = {
    'FakeNewsNet': r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv",
    'PolitiFact': r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv"
}

emotion_map = {
    'joy': 0, 'anger': 1, 'sadness': 2, 'fear': 3,
    'surprise': 4, 'neutral': 5, 'disgust': 6
}

results = {}

for name, path in datasets.items():
    print(f"\n Processing dataset: {name} ")
    try:
        df = pd.read_csv(path)
        df = df.dropna(subset=['clean_text', 'label', 'emotion'])
        df['emotion'] = df['emotion'].map(emotion_map)

        if df.shape[0] < 3:
            print(f"Skipping {name} due to insufficient data (less than 3 rows).")
            continue

    except FileNotFoundError:
        print(f" Error: File not found at path: {path}. Skipping this dataset.")
        continue
    except Exception as e:
        print(f" Error processing {name}: {e}. Skipping this dataset.")
        continue

    print(f" Data loaded successfully. Shape: {df.shape}")

    print("\n  Creating TF-IDF features ")
    tfidf = TfidfVectorizer(max_features=3000)
    X_tfidf = tfidf.fit_transform(df['clean_text'].astype(str)).toarray()
    y = df['label'].values

    print("  Calculating average spelling errors ")
    spell_errors = avg_spelling_errors(df['clean_text'].astype(str)).values.reshape(-1, 1)

    print("  Extracting emotion feature ")
    emotion_feature = df['emotion'].astype(float).values.reshape(-1, 1)

    # TF-IDF בלבד
    print("\n Running configuration: TF-IDF only")
    results[f"{name} - TF-IDF"] = evaluate_model(X_tfidf, y, get_stacking_model())

    # TF-IDF + שגיאות כתיב
    print("\n Running configuration: TF-IDF + Spelling Errors")
    X_with_spell = np.hstack([X_tfidf, spell_errors])
    results[f"{name} - TF-IDF + Spelling Errors"] = evaluate_model(X_with_spell, y, get_stacking_model())

    #  TF-IDF + שגיאות כתיב + רגשות
    print("\n Running configuration: TF-IDF + Spelling + Emotion")
    X_with_spell_emotion = np.hstack([X_tfidf, spell_errors, emotion_feature])
    results[f"{name} - TF-IDF + Spelling + Emotion"] = evaluate_model(X_with_spell_emotion, y, get_stacking_model())

output_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\Model\results\compare_emotions_spellchecker\stacking_tfidf_feature_comparison.json"
try:
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n Results saved successfully to: {output_path}")
except Exception as e:
    print(f"❌ Error saving results to {output_path}: {e}")