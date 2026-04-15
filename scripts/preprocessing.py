"""
preprocessing.py - pipeline ניקוי ועיבוד מקדים לציוצים
שימוש: python scripts/preprocessing.py
מריץ את כל שלושת ה-datasets: Manchester, Monkeypox, PHEME
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


def _split_and_save(gold: pd.DataFrame, gold_dir: str, prefix: str,
                    random_state: int = 42) -> tuple:
    """
    מחלק gold standard ל-train/val/test (70/15/15) ושומר את כל הקבצים.
    מחזיר (train, val, test).
    """
    train, temp = train_test_split(gold, test_size=0.30,
                                   stratify=gold['label'],
                                   random_state=random_state)
    val, test = train_test_split(temp, test_size=0.50,
                                 stratify=temp['label'],
                                 random_state=random_state)

    gold.to_csv(os.path.join(gold_dir, f'{prefix}_gold_standard.csv'),
                index=False, encoding='utf-8')
    train.to_csv(os.path.join(gold_dir, f'{prefix}_train.csv'),
                 index=False, encoding='utf-8')
    val.to_csv(os.path.join(gold_dir, f'{prefix}_val.csv'),
               index=False, encoding='utf-8')
    test.to_csv(os.path.join(gold_dir, f'{prefix}_test.csv'),
                index=False, encoding='utf-8')

    print(f'Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}')
    return train, val, test


# ──────────────────────────────────────────────
# Manchester
# ──────────────────────────────────────────────

def normalize_labels_manchester(df: pd.DataFrame) -> pd.DataFrame:
    """נרמול עמודת Rumour לתיוגים אחידים (Manchester)."""
    df = df.copy()
    df['label'] = df['Rumour'].str.strip().str.capitalize()
    df['label'] = df['label'].replace({
        'True': 'reliable',
        'Fake': 'misinformation',
        'Not related': 'not_related'
    })
    df['rumour_type'] = df['Type of rumour'].fillna('')
    return df


def run_preprocessing_manchester(input_path: str, output_dir: str, gold_dir: str,
                                  min_words: int = 5, max_chars: int = 350,
                                  reliable_sample: int = 2000, random_state: int = 42):
    """Pipeline עיבוד מקדים מלא - Manchester."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print('MANCHESTER PREPROCESSING')
    print(f'{"="*60}')
    print(f'טוען: {input_path}')
    df = pd.read_excel(input_path) if input_path.endswith('.xlsx') else pd.read_csv(input_path)
    print(f'נטען: {len(df):,} שורות')

    df = normalize_labels_manchester(df)
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

    _split_and_save(gold, gold_dir, 'manchester', random_state)
    print('Manchester: הושלם.')


# ──────────────────────────────────────────────
# Monkeypox
# ──────────────────────────────────────────────

def normalize_labels_monkeypox(df: pd.DataFrame) -> pd.DataFrame:
    """
    נרמול עמודת binary_class לתיוגים אחידים (Monkeypox).
    binary_class: 0 = reliable, 1 = misinformation
    ternary_class: 9 = reliable, 0 = borderline, 1 = misinformation
    """
    df = df.copy()
    df['label'] = df['binary_class'].map({0: 'reliable', 1: 'misinformation'})
    if 'ternary_class' in df.columns:
        df['label_3'] = df['ternary_class'].map(
            {9: 'reliable', 0: 'borderline', 1: 'misinformation'}
        )
    return df


def run_preprocessing_monkeypox(main_path: str, followup_path: str,
                                  output_dir: str, gold_dir: str,
                                  min_words: int = 5, max_chars: int = 350,
                                  reliable_sample: int = 2000, random_state: int = 42):
    """
    Pipeline עיבוד מקדים מלא - Monkeypox.
    טוען שני קבצי CSV (main + followup) ומאחד אותם.
    עמודת טקסט: 'text'
    עמודת תיוג: 'binary_class' (0=reliable, 1=misinformation)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print('MONKEYPOX PREPROCESSING')
    print(f'{"="*60}')

    df_main = pd.read_csv(main_path)
    df_followup = pd.read_csv(followup_path)
    df_main['source_file'] = 'main'
    df_followup['source_file'] = 'followup'
    df = pd.concat([df_main, df_followup], ignore_index=True)
    print(f'נטען (main + followup): {len(df):,} שורות')

    df = normalize_labels_monkeypox(df)
    # שמור רק שורות עם תיוג תקין
    df = df.dropna(subset=['label'])

    df['cleaned_tweet'] = df['text'].apply(lambda x: clean_tweet(x, max_chars))
    df['word_count'] = df['cleaned_tweet'].str.split().str.len()

    df = df[df['word_count'] >= min_words]
    print(f'אחרי סינון (>={min_words} מילים): {len(df):,}')

    df = df.drop_duplicates(subset='cleaned_tweet')
    print(f'אחרי dedup: {len(df):,}')
    print(df['label'].value_counts().to_string())

    # שמירת קובץ נקי מלא
    cols = ['number', 'created_at', 'text', 'cleaned_tweet', 'label', 'label_3',
            'source_file', 'retweet_count', 'reply_count', 'like_count',
            'quote_count', 'followers count', 'following count',
            'user is verified', 'user has url']
    cols = [c for c in cols if c in df.columns]
    clean_path = os.path.join(output_dir, 'monkeypox_clean.csv')
    df[cols].to_csv(clean_path, index=False, encoding='utf-8')
    print(f'נשמר: {clean_path}')

    # Gold Standard — all misinformation + sample of reliable
    misinfo = df[df['label'] == 'misinformation']
    reliable_pool = df[df['label'] == 'reliable']
    reliable_n = min(reliable_sample, len(reliable_pool))
    reliable = reliable_pool.sample(reliable_n, random_state=random_state)

    gold = pd.concat([misinfo, reliable], ignore_index=True)
    gold = gold.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f'\nGold Standard: {len(gold):,}')
    print(gold['label'].value_counts().to_string())

    _split_and_save(gold, gold_dir, 'monkeypox', random_state)
    print('Monkeypox: הושלם.')


# ──────────────────────────────────────────────
# PHEME
# ──────────────────────────────────────────────

def normalize_labels_pheme(df: pd.DataFrame) -> pd.DataFrame:
    """
    נרמול עמודת is_rumor לתיוגים אחידים (PHEME).
    is_rumor: 0.0 = not_rumour, 1.0 = rumour
    """
    df = df.copy()
    df['label'] = df['is_rumor'].map({0.0: 'not_rumour', 1.0: 'rumour'})
    if 'topic' in df.columns:
        df['topic'] = df['topic'].fillna('unknown')
    return df


def run_preprocessing_pheme(input_path: str, output_dir: str, gold_dir: str,
                              min_words: int = 5, max_chars: int = 350,
                              reliable_sample: int = 2000, random_state: int = 42):
    """
    Pipeline עיבוד מקדים מלא - PHEME.
    עמודת טקסט: 'text'
    עמודת תיוג: 'is_rumor' (0.0=not_rumour, 1.0=rumour)
    הערה: PHEME לא מאוזן (rumours << not_rumours), לכן דוגמים not_rumour.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print('PHEME PREPROCESSING')
    print(f'{"="*60}')
    print(f'טוען: {input_path}')
    df = pd.read_csv(input_path)
    print(f'נטען: {len(df):,} שורות')

    df = normalize_labels_pheme(df)
    # שמור רק שורות עם תיוג תקין
    df = df.dropna(subset=['label'])
    print(f'לאחר הסרת שורות ללא תיוג: {len(df):,}')

    df['cleaned_tweet'] = df['text'].apply(lambda x: clean_tweet(x, max_chars))
    df['word_count'] = df['cleaned_tweet'].str.split().str.len()

    df = df[df['word_count'] >= min_words]
    print(f'אחרי סינון (>={min_words} מילים): {len(df):,}')

    df = df.drop_duplicates(subset='cleaned_tweet')
    print(f'אחרי dedup: {len(df):,}')
    print(df['label'].value_counts().to_string())

    # שמירת קובץ נקי מלא
    cols = ['text', 'cleaned_tweet', 'label', 'topic', 'user.handle', 'word_count']
    cols = [c for c in cols if c in df.columns]
    clean_path = os.path.join(output_dir, 'pheme_clean.csv')
    df[cols].to_csv(clean_path, index=False, encoding='utf-8')
    print(f'נשמר: {clean_path}')

    # Gold Standard — all rumours + sample of not_rumour
    rumour = df[df['label'] == 'rumour']
    not_rumour_pool = df[df['label'] == 'not_rumour']
    not_rumour_n = min(reliable_sample, len(not_rumour_pool))
    not_rumour = not_rumour_pool.sample(not_rumour_n, random_state=random_state)

    gold = pd.concat([rumour, not_rumour], ignore_index=True)
    gold = gold.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f'\nGold Standard: {len(gold):,}')
    print(gold['label'].value_counts().to_string())

    _split_and_save(gold, gold_dir, 'pheme', random_state)
    print('PHEME: הושלם.')


# ──────────────────────────────────────────────
# Legacy wrapper — kept for backward compatibility
# ──────────────────────────────────────────────

def run_preprocessing(input_path: str, output_dir: str, gold_dir: str,
                      min_words: int = 5, max_chars: int = 350,
                      reliable_sample: int = 2000, random_state: int = 42):
    """
    Backward-compatible wrapper that calls run_preprocessing_manchester.
    Use the dataset-specific functions for new code.
    """
    run_preprocessing_manchester(
        input_path=input_path,
        output_dir=output_dir,
        gold_dir=gold_dir,
        min_words=min_words,
        max_chars=max_chars,
        reliable_sample=reliable_sample,
        random_state=random_state,
    )


# ──────────────────────────────────────────────
# CLI entry point — runs all three datasets
# ──────────────────────────────────────────────

if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW  = os.path.join(BASE, 'data', 'raw')
    PROC = os.path.join(BASE, 'data', 'processed')
    GOLD = os.path.join(BASE, 'data', 'gold_standard')

    run_preprocessing_manchester(
        input_path=os.path.join(RAW, 'manchester_raw.xlsx'),
        output_dir=PROC,
        gold_dir=GOLD,
    )

    run_preprocessing_monkeypox(
        main_path=os.path.join(RAW, 'monkeypox.csv'),
        followup_path=os.path.join(RAW, 'monkeypox-followup.csv'),
        output_dir=PROC,
        gold_dir=GOLD,
    )

    run_preprocessing_pheme(
        input_path=os.path.join(RAW, 'PHEME-rumourdetection.csv'),
        output_dir=PROC,
        gold_dir=GOLD,
    )

    print('\n✅ כל ה-datasets עובדו בהצלחה.')
