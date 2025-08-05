# תהליך הכנת הנתונים
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from langdetect import detect
import os

# הגדרת נתיבים לקבצים המעובדים משלב ה-EDA
fakenewsnet_processed_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\FakeNewsNet\FakeNewsNet_processed.csv"
politifact_processed_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\PolitiFact\PolitiFact_processed.csv"

# טעינת קבצים מעובדים מה-EDA
fakenewsnet_df = pd.read_csv(fakenewsnet_processed_path)
politifact_df = pd.read_csv(politifact_processed_path)

print("Original row count :")
print(f"FakeNewsNet: {len(fakenewsnet_df)}")
print(f"PolitiFact: {len(politifact_df)}")

# סינון טקסטים לאנגלית בלבד
def is_english(text):
    try:
        return detect(str(text)) == 'en'
    except:
        return False

rows_before_english_filter_fakenewsnet = len(fakenewsnet_df)
rows_before_english_filter_politifact = len(politifact_df)

fakenewsnet_df = fakenewsnet_df[fakenewsnet_df['text_content'].apply(is_english)].copy()
politifact_df = politifact_df[politifact_df['text_content'].apply(is_english)].copy()

print("\nAfter filtering non-English texts:")
print(f"FakeNewsNet: {len(fakenewsnet_df)} (Removed {rows_before_english_filter_fakenewsnet - len(fakenewsnet_df)} non-English rows)")
print(f"PolitiFact: {len(politifact_df)} (Removed {rows_before_english_filter_politifact - len(politifact_df)} non-English rows)")


# עיבוד טקסט (ניקוי ולימטיזציה)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

fakenewsnet_df['clean_text'] = fakenewsnet_df['text_content'].apply(preprocess_text)
politifact_df['clean_text'] = politifact_df['text_content'].apply(preprocess_text)

print("\nAfter text preprocessing:")
print(f"FakeNewsNet: {len(fakenewsnet_df)}")
print(f"PolitiFact: {len(politifact_df)}")

# הדפסת חלוקת תוויות לפני איזון
print("\nLabel distribution before balancing:")
print("FakeNewsNet:")
print(fakenewsnet_df['label'].value_counts())
print("\nPolitiFact:")
print(politifact_df['label'].value_counts())

# איזון תוויות
def balance_labels(df):
    if df['label'].nunique() < 2:
        print(f"Warning: Cannot balance dataframe with only one unique label. Returning original dataframe.")
        return df

    min_count = df['label'].value_counts().min()
    df_balanced = pd.concat([
        df[df['label'] == 0].sample(min_count, random_state=42),
        df[df['label'] == 1].sample(min_count, random_state=42)
    ])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

fakenewsnet_df_balanced = balance_labels(fakenewsnet_df[['clean_text', 'label', 'emotion']])
politifact_df_balanced = balance_labels(politifact_df[['clean_text', 'label', 'emotion']])

# הדפסת חלוקת תוויות לאחר איזון
print("\nLabel distribution after balancing:")
print("FakeNewsNet:")
print(fakenewsnet_df_balanced['label'].value_counts())
print("\nPolitiFact:")
print(politifact_df_balanced['label'].value_counts())

# שמירת תוצאות לקבצים חדשים
fakenewsnet_balanced_save_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv"
politifact_balanced_save_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv"

fakenewsnet_df_balanced.to_csv(fakenewsnet_balanced_save_path, index=False)
politifact_df_balanced.to_csv(politifact_balanced_save_path, index=False)

print("\nFiles saved successfully.")