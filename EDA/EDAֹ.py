# EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from wordcloud import WordCloud
from langdetect import detect
from tabulate import tabulate
from transformers import pipeline
import re
import os
from spellchecker import SpellChecker
import sys

# הגדרת נתיב לקובץ הפלט הטקסטואלי
output_text_file_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\EDA_output_log.txt"

# פונקציה מותאמת אישית לניתוב פלט גם למסך וגם לקובץ
class DualOutput:
    def __init__(self, filepath, console_out):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.console = console_out

    def write(self, message):
        self.file.write(message)
        self.console.write(message)

    def flush(self):
        self.file.flush()
        self.console.flush()

    def close(self):
        self.file.close()

# הפעלת הניתוב הכפול בתחילת הסקריפט
original_stdout = sys.stdout
sys.stdout = DualOutput(output_text_file_path, original_stdout)

print(f"--- EDA Script Output Log - {pd.Timestamp.now()} ---")
print(f"All console outputs are being redirected to: {output_text_file_path}\n")

# הורדת משאבי NLTK
try:
    nltk.data.find('corpora/words')
except nltk.downloader.DownloadError:
    nltk.download('words')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# הגדרת נתיבים
base_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\FakeNewsNet"
politifact_json_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\PolitiFact\politifact_factcheck_data.json"
politifact_filtered_save_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\PolitiFact\PolitiFact_processed.csv"
fakenewsnet_processed_save_path = os.path.join(base_path, "FakeNewsNet_processed.csv")

# נתיב חדש לשמירת גרפים
figure_save_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\EDA\Figure"
os.makedirs(figure_save_path, exist_ok=True)


# שמות קבצי המקור FakeNewsNet
fakenewsnet_file_names = [
    "gossipcop_fake.csv",
    "gossipcop_real.csv",
    "politifact_fake.csv",
    "politifact_real.csv"
]

print("\n Previewing the FakeNewsNet Dataset (CSV files) ")

# הצגת 5 שורות ראשונות מכל קובץ CSV של FakeNewsNet
for file in fakenewsnet_file_names:
    file_path = os.path.join(base_path, file)
    df = pd.read_csv(file_path)
    print(f"\nFirst 5 rows of {file}:")
    print(tabulate(df.head(), headers='keys', tablefmt='grid', showindex=True))

# קריאה ואיחוד הקבצים של FakeNewsNet
dataframes = []
for file in fakenewsnet_file_names:
    df = pd.read_csv(os.path.join(base_path, file))
    source = "gossipcop" if "gossipcop" in file else "politifact_fakenewsnet"
    label = "fake" if "fake" in file else "real"
    df['source'] = source
    df['label'] = label
    dataframes.append(df)

# יצירת הדאטהסט המאוחד FakeNewsNet
fakenewsnet_df = pd.concat(dataframes, ignore_index=True)
initial_rows_fakenewsnet = fakenewsnet_df.shape[0]
print(f"\n📊 Initial FakeNewsNet Dataset size: {initial_rows_fakenewsnet} rows")

# הצגת סוגי עמודות של FakeNewsNet לפני השמטת עמודות (קובץ מקורי)
print("\nData types for each column in FakeNewsNet (Original - before dropping columns):")
print(tabulate(fakenewsnet_df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}), headers="keys", tablefmt="grid"))


# עיבוד והאחדת עמודות עבור FakeNewsNet
# השמטת עמודות לא רלוונטיות
columns_to_drop_fakenewsnet = ['id', 'news_url', 'tweet_ids']
fakenewsnet_df_processed = fakenewsnet_df.drop(columns=columns_to_drop_fakenewsnet, errors='ignore')
print(f"Dropped columns: {columns_to_drop_fakenewsnet} from FakeNewsNet dataset.")

# שינוי שם עמודת הכותרת ל-'text_content'
fakenewsnet_df_processed.rename(columns={'title': 'text_content'}, inplace=True)

# המרת תווית לבינארית (0 ו-1)
fakenewsnet_df_processed['label'] = fakenewsnet_df_processed['label'].map({'fake': 0, 'real': 1})
print("Converted 'fake' to 0 and 'real' to 1 in FakeNewsNet 'label' column.")

# יצירת pipeline לזיהוי רגשות
print("\nRunning emotion analysis for FakeNewsNet...")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
fakenewsnet_df_processed['emotion'] = fakenewsnet_df_processed['text_content'].apply(lambda x: emotion_classifier(x)[0][0]['label'] if isinstance(x, str) else 'neutral')
print("Emotion column added to FakeNewsNet dataset.")


# שמירת הקובץ המאוחד והמעובד של FakeNewsNet
try:
    fakenewsnet_df_processed.to_csv(fakenewsnet_processed_save_path, index=False)
    print(f"\n✅ FakeNewsNet combined and processed data saved to: {fakenewsnet_processed_save_path}")
except Exception as e:
    print(f"\n❌ Error saving FakeNewsNet processed data: {e}")

# DataFrame המעובד עבור כל הניתוחים הבאים
fakenewsnet_df = fakenewsnet_df_processed.copy()

# הצגת סוגי עמודות של FakeNewsNet לאחר העיבוד
print("\nData types for each column in FakeNewsNet (after processing):")
print(tabulate(fakenewsnet_df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}), headers="keys", tablefmt="grid"))

print("\n Current columns in FakeNewsNet after processing: ")
print(fakenewsnet_df.columns.tolist())


# Plot label balance for FakeNewsNet
print("\n Label Balance Plot (FakeNewsNet) ")
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=fakenewsnet_df, x='label', hue='label', palette={0: 'red', 1: 'green'}, legend=False)
plt.title("Overall Label Balance in FakeNewsNet Dataset", fontsize=14)
plt.xlabel("Label (0=Fake, 1=Real)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks([0, 1], ['Fake', 'Real'])
# Add exact numbers
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'FakeNewsNet_Label_Balance.png'))
plt.close()


# פונקציה לבדיקת שפה
def is_english(text):
    if pd.isna(text):
        return False
    try:
        return detect(str(text)) == 'en'
    except:
        return False

# הצגת כמות טקסטים שאינם בשפה האנגלית
print("\n Language Check for FakeNewsNet - Reporting Non-English Rows) ")

# Count non-English text_content
non_english_text_content_mask_fakenewsnet = fakenewsnet_df['text_content'].apply(lambda x: not is_english(x))
count_non_english_text_content_fakenewsnet = non_english_text_content_mask_fakenewsnet.sum()

total_non_english_rows_fakenewsnet = count_non_english_text_content_fakenewsnet

print(f"Found {total_non_english_rows_fakenewsnet} rows with non-English content in FakeNewsNet dataset.")
print(f"FakeNewsNet: {initial_rows_fakenewsnet} rows initially, {fakenewsnet_df.shape[0]} rows after column drops and language check (no language rows removed).")


# בדיקת אנומליות FakeNewsNet
print("\n Anomaly Detection in Text Content Length (FakeNewsNet) ")
fakenewsnet_df['text_length'] = fakenewsnet_df['text_content'].apply(lambda x: len(str(x).split()))
text_lengths = fakenewsnet_df['text_length']
mean_text_length = text_lengths.mean()
std_text_length = text_lengths.std()
threshold = mean_text_length + (3 * std_text_length)
anomalous_texts = fakenewsnet_df[fakenewsnet_df['text_length'] > threshold]

print(f"Mean text content length: {mean_text_length:.2f}")
print(f"Standard deviation of text content length: {std_text_length:.2f}")
print(f"Anomaly threshold (Mean + 3*StdDev): {threshold:.2f}")
print(f"Number of anomalous texts: {anomalous_texts.shape[0]}")
print("\nAnomalous Text Contents (FakeNewsNet - first 5):")
print(tabulate(anomalous_texts[['text_content', 'text_length', 'source', 'label']].head(), headers='keys', tablefmt='grid', showindex=True))

# גרף רגשות עבור FakeNewsNet
print("\n Emotion Distribution by Label (FakeNewsNet - X-axis as Label) ")
plt.figure(figsize=(12, 6))
# הסרה של neutral מהגרף
fakenewsnet_df_filtered_emotion = fakenewsnet_df[fakenewsnet_df['emotion'] != 'neutral']
ax = sns.countplot(data=fakenewsnet_df_filtered_emotion, x='label', hue='emotion', palette='Set2')
plt.title("Emotion Distribution by Label (FakeNewsNet - Excluding Neutral)", fontsize=14)
plt.xlabel("Label (0=Fake, 1=Real)")
plt.ylabel("Count")
plt.xticks([0, 1], ['Fake', 'Real'])
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'FakeNewsNet_Emotion_Distribution_by_Label.png'))
plt.close()


# הצגת ביטויים דו-מילתיים נפוצים
print("\n 2-gram Analysis for FakeNewsNet ")
def get_top_ngrams(corpus, n=2, top_k=10):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]

all_fakenewsnet_text_content = fakenewsnet_df['text_content'].dropna().astype(str)
top_2grams_fakenewsnet = get_top_ngrams(all_fakenewsnet_text_content)
df_2grams_fakenewsnet = pd.DataFrame(top_2grams_fakenewsnet, columns=['2-gram', 'Frequency'])

plt.figure(figsize=(12, 7))
ax = sns.barplot(x='Frequency', y='2-gram', data=df_2grams_fakenewsnet, palette='viridis', hue='2-gram', legend=False)
plt.title("Top 10 2-grams in FakeNewsNet Text Content (Combined)", fontsize=14)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("2-gram", fontsize=12)
# Add exact numbers
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'FakeNewsNet_Top_2grams.png'))
plt.close()


# ניתוח שגיאות כתיב
print("\n Spell Check Analysis for FakeNewsNet ")
spell = SpellChecker()

def count_misspellings(text):
    if pd.isna(text):
        return 0
    words_list = word_tokenize(str(text).lower())
    misspelled = spell.unknown(words_list)
    return len(misspelled)

fakenewsnet_df['misspellings'] = fakenewsnet_df['text_content'].apply(count_misspellings)

avg_misspellings_fakenewsnet_combined = fakenewsnet_df['misspellings'].mean()

print(f"Average misspellings per text content in FakeNewsNet (Combined): {avg_misspellings_fakenewsnet_combined:.2f}")


# הצגת מילים פופולריות
print("\n Word Cloud for FakeNewsNet ")
all_fakenewsnet_text_content_for_cloud = " ".join(fakenewsnet_df['text_content'].dropna().astype(str))
wordcloud_fakenewsnet_combined = WordCloud(width=800, height=400, background_color='white').generate(all_fakenewsnet_text_content_for_cloud)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_fakenewsnet_combined, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for FakeNewsNet Text Content (Combined)")
plt.savefig(os.path.join(figure_save_path, 'FakeNewsNet_Word_Cloud.png'))
plt.close()

# PolitiFact
politifact_json_df = pd.read_json(politifact_json_path, lines=True)
initial_rows_politifact_json = politifact_json_df.shape[0]
print(f"\n📊 Initial PolitiFact Dataset (JSON file) size: {initial_rows_politifact_json} rows")

# הצגת 10 שורות PolitiFact
print("\n Previewing the PolitiFact Dataset (JSON file) --\n")
print(tabulate(politifact_json_df.head(10), headers='keys', tablefmt='grid', showindex=True))

# עיבוד והאחדת עמודות עבור PolitiFact
# שמירת רק את העמודות 'statement' ו-'verdict'
filtered_politifact_json_df = politifact_json_df[['statement', 'verdict']].copy()

# שינוי שם עמודת ההצהרה ל-'text_content'
filtered_politifact_json_df.rename(columns={'statement': 'text_content'}, inplace=True)

# הוספת עמודת source
filtered_politifact_json_df['source'] = 'politifact_json'
print("Added 'source' column with 'politifact_json' to PolitiFact dataset.")

# יצירת pipeline לזיהוי רגשות
print("\nRunning emotion analysis for PolitiFact...")
filtered_politifact_json_df['emotion'] = filtered_politifact_json_df['text_content'].apply(lambda x: emotion_classifier(x)[0][0]['label'] if isinstance(x, str) else 'neutral')
print("Emotion column added to PolitiFact dataset.")

# הצגת סוגי עמודות PolitiFact במקור
print("\nData types for each column in PolitiFact (JSON file - original columns):")
print(tabulate(politifact_json_df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}), headers="keys", tablefmt="grid"))

# גרף התפלגות verdict
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=politifact_json_df, x='verdict', hue='verdict', order=politifact_json_df['verdict'].value_counts().index, palette='viridis', legend=False)
plt.title("Distribution of Verdicts in PolitiFact Dataset", fontsize=14)
plt.xlabel("Verdict", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
# Add exact numbers
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'PolitiFact_Verdict_Distribution.png'))
plt.close()


# בדיקה של כמות השורות שאינם בשפה האנגלית עבור PolitiFact
print("\n Language Check for PolitiFact ")
non_english_statements_mask_politifact_json = filtered_politifact_json_df['text_content'].apply(lambda x: not is_english(x))
total_non_english_rows_politifact_json = non_english_statements_mask_politifact_json.sum()

print(f"Found {total_non_english_rows_politifact_json} rows with non-English content (statement) in PolitiFact JSON dataset.")
print(f"PolitiFact (JSON file): {initial_rows_politifact_json} rows initially, {filtered_politifact_json_df.shape[0]} rows after language check (no rows removed).")


# גרף רגשות לפי verdict (עם הסתרת 'neutral')
plt.figure(figsize=(12, 6))
filtered_politifact_json_df_filtered_emotion = filtered_politifact_json_df[filtered_politifact_json_df['emotion'] != 'neutral']
ax = sns.countplot(data=filtered_politifact_json_df_filtered_emotion, x='verdict', hue='emotion', palette='Set2', order=politifact_json_df['verdict'].value_counts().index)
plt.title("Distribution of Emotions by Verdict (PolitiFact JSON file - Excluding Neutral)", fontsize=14)
plt.xlabel("Verdict")
plt.ylabel("Count")
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
# Add exact numbers
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'PolitiFact_Emotion_Distribution.png'))
plt.close()

# המרת verdict ל-label בינארי PolitiFact
filtered_politifact_json_df['label'] = filtered_politifact_json_df['verdict'].apply(lambda x: 1 if x.lower() in ['true', 'mostly-true'] else 0)
filtered_politifact_json_df.drop(columns='verdict', inplace=True)

# הוספת עמודת אורך הטקסט ל-PolitiFact
filtered_politifact_json_df['text_length'] = filtered_politifact_json_df['text_content'].apply(lambda x: len(str(x).split()))

# הוספת עמודת שגיאות כתיב ל-PolitiFact
filtered_politifact_json_df['misspellings'] = filtered_politifact_json_df['text_content'].apply(count_misspellings)


# שמירת הקובץ החדש של PolitiFact
columns_for_final_save = ['text_content', 'text_length', 'emotion', 'misspellings', 'label', 'source']
filtered_politifact_json_df_final = filtered_politifact_json_df[columns_for_final_save].copy()

try:
    filtered_politifact_json_df_final.to_csv(politifact_filtered_save_path, index=False)
    print(f"\n✅ PolitiFact processed data saved to: {politifact_filtered_save_path}")
except Exception as e:
    print(f"\n❌ Error saving PolitiFact processed data: {e}")

# הצגת סוגי עמודות של PolitiFact לאחר העיבוד
print("\nData types for each column in PolitiFact (after processing and final column selection):")
print(tabulate(filtered_politifact_json_df_final.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}), headers="keys", tablefmt="grid"))

print("\nCurrent columns in PolitiFact after processing:")
print(filtered_politifact_json_df_final.columns.tolist())


# גרף השוואת תוויות בPolitiFact
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=filtered_politifact_json_df_final, x='label', hue='label', palette={0: 'red', 1: 'green'}, legend=False)
plt.title("Label Balance in PolitiFact Dataset (JSON file)", fontsize=14)
plt.xlabel("Label (0=False, 1=True)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks([0, 1], ['False', 'True'])
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'PolitiFact_Label_Balance.png'))
plt.close()


# רגשות לפי תווית בינארית עבור PolitiFact
print("\n Emotion Distribution by Label - PolitiFact")
plt.figure(figsize=(12, 6))
politifact_df_filtered_emotion_binary_label = filtered_politifact_json_df_final[filtered_politifact_json_df_final['emotion'] != 'neutral']
ax = sns.countplot(data=politifact_df_filtered_emotion_binary_label, x='label', hue='emotion', palette='Set2')
plt.title("Emotion Distribution by Label (PolitiFact - Excluding Neutral)", fontsize=14)
plt.xlabel("Label (0=False, 1=True)")
plt.ylabel("Count")
plt.xticks([0, 1], ['False', 'True'])
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
# Add exact numbers
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'PolitiFact_Emotion_Distribution_by_Binary_Label.png'))
plt.close()


# 2-gram הצגת PolitiFact
print("\n 2-gram Analysis for PolitiFact ")
politifact_json_statements = filtered_politifact_json_df_final['text_content'].dropna().astype(str)
top_2grams_politifact_json = get_top_ngrams(politifact_json_statements)
df_2grams_politifact_json = pd.DataFrame(top_2grams_politifact_json, columns=['2-gram', 'Frequency'])

plt.figure(figsize=(12, 7))
ax = sns.barplot(x='Frequency', y='2-gram', data=df_2grams_politifact_json, palette='viridis', hue='2-gram', legend=False)
plt.title("Top 10 2-grams in PolitiFact Statements (JSON file)", fontsize=14)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("2-gram", fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'PolitiFact_Top_2grams.png'))
plt.close()


# בדיקת שגיאות PolitiFact
print("\n Spell Check Analysis for PolitiFact ")
avg_misspellings_politifact_json_statements = filtered_politifact_json_df_final['misspellings'].mean()
print(f"Average misspellings per statement in PolitiFact : {avg_misspellings_politifact_json_statements:.2f}")

# בדיקת אנומליות PolitiFact
print("\n Anomaly Detection in Text Content Length - PolitiFact ")
text_lengths_json = filtered_politifact_json_df_final['text_length']
mean_text_length_json = text_lengths_json.mean()
std_text_length_json = text_lengths_json.std()
threshold_text_json = mean_text_length_json + (3 * std_text_length_json)
anomalous_statements_json = filtered_politifact_json_df_final[filtered_politifact_json_df_final['text_length'] > threshold_text_json]

print(f"Mean text content length: {mean_text_length_json:.2f}")
print(f"Standard deviation of text content length: {std_text_length_json:.2f}")
print(f"Anomaly threshold (Mean + 3*StdDev): {threshold_text_json:.2f}")
print(f"Number of anomalous statements: {anomalous_statements_json.shape[0]}")
print("\nAnomalous Statements (PolitiFact JSON file - first 5):")
print(tabulate(anomalous_statements_json[['text_content', 'text_length', 'label', 'source']].head(), headers='keys', tablefmt='grid', showindex=True))


# Word Cloud for PolitiFact
print("\n Word Clouds for PolitiFact ")
all_politifact_json_statements = " ".join(filtered_politifact_json_df_final['text_content'].dropna().astype(str))
wordcloud_politifact_json_statements = WordCloud(width=800, height=400, background_color='white').generate(all_politifact_json_statements)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_politifact_json_statements, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for PolitiFact Statements ")
plt.savefig(os.path.join(figure_save_path, 'PolitiFact_Word_Cloud.png'))
plt.close()

filtered_df_loaded = pd.read_csv(politifact_filtered_save_path)

# סטטיסטיקה
print("\nText Content length statistics (PolitiFact Filtered):")
print(filtered_df_loaded['text_length'].describe())

# התפלגות אורך טענות (היסטוגרמה)
plt.figure(figsize=(10, 5))
sns.histplot(data=filtered_df_loaded, x='text_length', bins=30, kde=True)
plt.title("Distribution of Text Content Lengths (PolitiFact Filtered)")
plt.xlabel("Number of Words in Text Content")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'PolitiFact_Text_Length_Histogram.png'))
plt.close()

# Boxplot לפי תווית
plt.figure(figsize=(8, 5))
sns.boxplot(data=filtered_df_loaded, x='label', y='text_length', hue='label', palette={0: 'red', 1: 'green'}, legend=False)
plt.xticks([0, 1], ['False', 'True'])
plt.title("Text Content Length by Label (Boxplot - PolitiFact Filtered)", fontsize=14)
plt.xlabel("Label")
plt.ylabel("Number of Words")
plt.tight_layout()
plt.savefig(os.path.join(figure_save_path, 'PolitiFact_Text_Length_Boxplot.png'))
plt.close()

print("\n Final Dataset Sizes After Processing ")
print(f"FakeNewsNet: {initial_rows_fakenewsnet} rows initially, {fakenewsnet_df.shape[0]} rows after processing and column drops.")
print(f"PolitiFact (JSON file): {initial_rows_politifact_json} rows initially, {filtered_politifact_json_df_final.shape[0]} rows after processing and column selection.")

print("\n Final columns for both datasets: ")
print("FakeNewsNet columns:", fakenewsnet_df.columns.tolist())
print("PolitiFact columns:", filtered_politifact_json_df_final.columns.tolist())

if sorted(fakenewsnet_df.columns.tolist()) == sorted(filtered_politifact_json_df_final.columns.tolist()):
    print("\n✅ Columns are identical in both FakeNewsNet and PolitiFact processed datasets.")
else:
    print("\n❌ Columns are NOT identical in FakeNewsNet and PolitiFact processed datasets. Please review.")
    print("FakeNewsNet unique columns:", set(fakenewsnet_df.columns.tolist()) - set(filtered_politifact_json_df_final.columns.tolist()))
    print("PolitiFact unique columns:", set(filtered_politifact_json_df_final.columns.tolist()) - set(fakenewsnet_df.columns.tolist()))

# סגירת קובץ הפלט בתום הריצה
sys.stdout.close()
sys.stdout = original_stdout
print(f"\nScript finished. All console output saved to: {output_text_file_path}")