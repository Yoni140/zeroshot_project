"""
preprocessing.py - pipeline ניקוי ועיבוד מקדים לציוצים
שימוש: python scripts/preprocessing.py
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split


# ===== פונקציות ניקוי =====

def clean_tweet(text: str, max_chars: int = 350) -> str:
    """ניקוי ציוץ: הסרת URLs, mentions, ניקוי תווים, הגבלת אורך."""
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)        # הסרת URLs
    text = re.sub(r'@\w+', '', text)                     # הסרת @mentions
    text = re.sub(r'#(\w+)', r'\1', text)                # הסרת # שמירת טקסט
    text = re.sub(r'^RT\s*[:]?\s*', '', text, flags=re.IGNORECASE)  # הסרת RT
    text = text.encode('ascii', 'ignore').decode('ascii') # הסרת non-ASCII
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"\\-]', ' ', text)  # תווים מיוחדים
    text = re.sub(r'\s+', ' ', text).strip()             # ניקוי רווחים
    return text[:max_chars]


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """נרמול עמודת Rumour לתיוגים אחידים."""
    df = df.copy()
    df['label'] = df['Rumour'].str.strip().str.capitalize()
    df['label'] = df['label'].replace({
        'True': 'reliable',
        'Fake': 'misinformation',
        'Not related': 'not_related'
    })
    df['rumour_type'] = df['Type of rumour'].fillna('')
    return df


def run_preprocessing(input_path: str, output_dir: str, gold_dir: str,
                      min_words: int = 5, max_chars: int = 350,
                      reliable_sample: int = 2000, random_state: int = 42):
    """Pipeline עיבוד מקדים מלא."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    print(f'טוען: {input_path}')
    df = pd.read_excel(input_path) if input_path.endswith('.xlsx') else pd.read_csv(input_path)
    print(f'נטען: {len(df):,} שורות')

    df = normalize_labels(df)
    df['cleaned_tweet'] = df['OrigTweet'].apply(lambda x: clean_tweet(x, max_chars))
    df['word_count'] = df['cleaned_tweet'].str.split().str.len()

    df = df[df['word_count'] >= min_words]
    print(f'אחרי סינון (>={min_words} מילים): {len(df):,}')

    df = df.drop_duplicates(subset='cleaned_tweet')
    print(f'אחרי dedup: {len(df):,}')

    # שמירת קובץ נקי מלא
    cols = ['Id', 'CreatedAt', 'author_id', 'OrigTweet', 'cleaned_tweet',
            'label', 'rumour_type', 'mVader', 'mRetweets', 'mLikes',
            'mReplies', 'mHasURL', 'mHasMedia', 'mUFollowers', 'mUFollowing']
    cols = [c for c in cols if c in df.columns]
    clean_path = os.path.join(output_dir, 'manchester_clean.csv')
    df[cols].to_csv(clean_path, index=False, encoding='utf-8')
    print(f'נשמר: {clean_path}')

    # Gold Standard
    misinfo = df[df['label'] == 'misinformation']
    reliable_pool = df[df['label'] == 'reliable']
    reliable_n = min(reliable_sample, len(reliable_pool))
    reliable = reliable_pool.sample(reliable_n, random_state=random_state)

    gold = pd.concat([misinfo, reliable], ignore_index=True)
    gold = gold.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f'\nGold Standard: {len(gold):,}')
    print(gold['label'].value_counts().to_string())

    train, temp = train_test_split(gold, test_size=0.30, stratify=gold['label'], random_state=random_state)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp['label'], random_state=random_state)

    gold.to_csv(os.path.join(gold_dir, 'manchester_gold_standard.csv'), index=False, encoding='utf-8')
    train.to_csv(os.path.join(gold_dir, 'manchester_train.csv'), index=False, encoding='utf-8')
    val.to_csv(os.path.join(gold_dir, 'manchester_val.csv'), index=False, encoding='utf-8')
    test.to_csv(os.path.join(gold_dir, 'manchester_test.csv'), index=False, encoding='utf-8')

    print(f'\nTrain: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}')
    print('הושלם.')


if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_preprocessing(
        input_path=os.path.join(BASE, 'data', 'raw', 'manchester_raw.xlsx'),
        output_dir=os.path.join(BASE, 'data', 'processed'),
        gold_dir=os.path.join(BASE, 'data', 'gold_standard'),
    )
