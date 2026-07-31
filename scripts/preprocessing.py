"""
preprocessing.py - pipeline ניקוי ועיבוד מקדים לציוצים
שימוש: python scripts/preprocessing.py
מריץ את כל שלושת ה-datasets: Manchester, Monkeypox, PHEME

==========================================================================
סכמת 3 קלאסים (3-CLASS SCHEME)
==========================================================================
כל דאטהסט מקבל קלאס שלישי "unrelated" (ציוץ שאינו רלוונטי לאירוע/נושא):

  Manchester : reliable / misinformation / unrelated
  Monkeypox  : reliable / misinformation / unrelated
  PHEME      : not_rumour / rumour / unrelated   (שמות נטיביים + unrelated)

מקור הקלאס "unrelated" בכל דאטהסט:
  Manchester : relevance filter לפי מילות-מפתח של האירוע. ציוץ מתויג
               'reliable' שאינו מזכיר את פיגוע מנצ'סטר ארנה -> unrelated
               (וכן 3 ה-'Not related' המקוריים).
  Monkeypox  : עמודת ternary_class המקורית — 9 -> unrelated, 0 -> reliable,
               1 -> misinformation.
  PHEME      : topic == 'unknown' (ציוץ שלא שויך לאף אירוע מתועד) -> unrelated.
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split


# ===== קונפיגורציה =====

# כמה דוגמאות מקסימום לכל קלאס ב-gold standard (לאיזון יחסי).
PER_CLASS_CAP = 1500

# Manchester — מילות מפתח של פיגוע מנצ'סטר ארנה (2017).
# ציוץ 'reliable' שאינו מכיל אף אחת מהן נחשב לא-רלוונטי (unrelated).
MANCHESTER_KEYWORDS = re.compile(
    r'manchester|arena|\bbomb|blast|explos|\battack|ariana|grande|concert|'
    r'terror|abedi|suicide|victim|evacuat|injur|prayformanchester|attacker|'
    r'salman|libyan|\bmci\b|ambulanc|emergency|wewillnot|standtogether',
    re.IGNORECASE,
)


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


def _build_gold_3class(df: pd.DataFrame, classes: list, per_class_cap: int,
                       random_state: int = 42) -> pd.DataFrame:
    """
    בונה gold standard מאוזן יחסית: עד `per_class_cap` דוגמאות מכל קלאס.
    קלאס קטן מה-cap נלקח במלואו.
    """
    parts = []
    for cls in classes:
        pool = df[df['label'] == cls]
        n = min(per_class_cap, len(pool))
        parts.append(pool.sample(n, random_state=random_state))
    gold = pd.concat(parts, ignore_index=True)
    gold = gold.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return gold


def _split_and_save(gold: pd.DataFrame, gold_dir: str, prefix: str,
                    random_state: int = 42) -> tuple:
    """
    מחלק gold standard ל-train/val/test (70/15/15, stratified) ושומר את כל הקבצים.
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
    print('  test label dist:', test['label'].value_counts().to_dict())
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


def apply_relevance_filter_manchester(df: pd.DataFrame) -> pd.DataFrame:
    """
    קלאס שלישי 'unrelated' עבור Manchester באמצעות relevance filter:
      - ציוץ 'reliable' שאינו מזכיר את האירוע (אין מילת מפתח) -> 'unrelated'
      - ציוץ 'not_related' מקורי                              -> 'unrelated'
      - 'misinformation' נשמר כפי שהוא (תיוג אנושי קיים לא נדרס)
    """
    df = df.copy()
    has_kw = df['cleaned_tweet'].apply(lambda t: bool(MANCHESTER_KEYWORDS.search(str(t))))
    reliable_off_topic = (df['label'] == 'reliable') & (~has_kw)
    native_not_related = (df['label'] == 'not_related')
    df.loc[reliable_off_topic | native_not_related, 'label'] = 'unrelated'
    return df


def run_preprocessing_manchester(input_path: str, output_dir: str, gold_dir: str,
                                  min_words: int = 5, max_chars: int = 350,
                                  per_class_cap: int = PER_CLASS_CAP, random_state: int = 42):
    """Pipeline עיבוד מקדים מלא - Manchester (3 קלאסים)."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print('MANCHESTER PREPROCESSING (3-class)')
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

    df = apply_relevance_filter_manchester(df)
    print('התפלגות תיוג (3-class):')
    print(df['label'].value_counts().to_string())

    # שמירת קובץ נקי מלא
    cols = ['Id', 'CreatedAt', 'author_id', 'OrigTweet', 'cleaned_tweet',
            'label', 'rumour_type', 'mVader', 'mRetweets', 'mLikes',
            'mReplies', 'mHasURL', 'mHasMedia', 'mUFollowers', 'mUFollowing']
    cols = [c for c in cols if c in df.columns]
    clean_path = os.path.join(output_dir, 'manchester_clean.csv')
    df[cols].to_csv(clean_path, index=False, encoding='utf-8')
    print(f'נשמר: {clean_path}')

    gold = _build_gold_3class(df, ['reliable', 'misinformation', 'unrelated'],
                              per_class_cap, random_state)
    print(f'\nGold Standard: {len(gold):,}')
    print(gold['label'].value_counts().to_string())

    _split_and_save(gold, gold_dir, 'manchester', random_state)
    print('Manchester: הושלם.')


# ──────────────────────────────────────────────
# Monkeypox
# ──────────────────────────────────────────────

def normalize_labels_monkeypox(df: pd.DataFrame) -> pd.DataFrame:
    """
    נרמול לתיוג 3-class (Monkeypox) מתוך עמודת ternary_class המקורית:
        ternary_class: 9 = unrelated, 0 = reliable, 1 = misinformation
    שורות ללא ternary_class תקין מושמטות.
    (binary_class נשמר ב-label_binary לצורך השוואה/אסמכתא.)
    """
    df = df.copy()
    if 'binary_class' in df.columns:
        df['label_binary'] = df['binary_class'].map({0: 'reliable', 1: 'misinformation'})
    df['label'] = df['ternary_class'].map(
        {9: 'unrelated', 0: 'reliable', 1: 'misinformation'}
    )
    return df


def run_preprocessing_monkeypox(main_path: str, followup_path: str,
                                  output_dir: str, gold_dir: str,
                                  min_words: int = 5, max_chars: int = 350,
                                  per_class_cap: int = PER_CLASS_CAP, random_state: int = 42):
    """
    Pipeline עיבוד מקדים מלא - Monkeypox (3 קלאסים).
    טוען שני קבצי CSV (main + followup) ומאחד אותם.
    עמודת טקסט: 'text' | עמודת תיוג: 'ternary_class'
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print('MONKEYPOX PREPROCESSING (3-class)')
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
    print('התפלגות תיוג (3-class):')
    print(df['label'].value_counts().to_string())

    # שמירת קובץ נקי מלא
    cols = ['number', 'created_at', 'text', 'cleaned_tweet', 'label', 'label_binary',
            'source_file', 'retweet_count', 'reply_count', 'like_count',
            'quote_count', 'followers count', 'following count',
            'user is verified', 'user has url']
    cols = [c for c in cols if c in df.columns]
    clean_path = os.path.join(output_dir, 'monkeypox_clean.csv')
    df[cols].to_csv(clean_path, index=False, encoding='utf-8')
    print(f'נשמר: {clean_path}')

    gold = _build_gold_3class(df, ['reliable', 'misinformation', 'unrelated'],
                              per_class_cap, random_state)
    print(f'\nGold Standard: {len(gold):,}')
    print(gold['label'].value_counts().to_string())

    _split_and_save(gold, gold_dir, 'monkeypox', random_state)
    print('Monkeypox: הושלם.')


# ──────────────────────────────────────────────
# PHEME
# ──────────────────────────────────────────────

def normalize_labels_pheme(df: pd.DataFrame) -> pd.DataFrame:
    """
    נרמול לתיוג 3-class (PHEME):
        is_rumor: 0.0 = not_rumour, 1.0 = rumour
        topic == 'unknown' (ציוץ שלא שויך לאף אירוע מתועד) -> unrelated
    הקלאס 'unrelated' גובר על rumour/not_rumour עבור ציוצים ללא אירוע.
    """
    df = df.copy()
    df['label'] = df['is_rumor'].map({0.0: 'not_rumour', 1.0: 'rumour'})
    if 'topic' in df.columns:
        df['topic'] = df['topic'].fillna('unknown')
        df.loc[df['topic'] == 'unknown', 'label'] = 'unrelated'
    return df


def run_preprocessing_pheme(input_path: str, output_dir: str, gold_dir: str,
                              min_words: int = 5, max_chars: int = 350,
                              per_class_cap: int = PER_CLASS_CAP, random_state: int = 42):
    """
    Pipeline עיבוד מקדים מלא - PHEME (3 קלאסים).
    עמודת טקסט: 'text' | עמודת תיוג: 'is_rumor' + 'topic'
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print('PHEME PREPROCESSING (3-class)')
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
    print('התפלגות תיוג (3-class):')
    print(df['label'].value_counts().to_string())

    # שמירת קובץ נקי מלא
    cols = ['text', 'cleaned_tweet', 'label', 'topic', 'user.handle', 'word_count']
    cols = [c for c in cols if c in df.columns]
    clean_path = os.path.join(output_dir, 'pheme_clean.csv')
    df[cols].to_csv(clean_path, index=False, encoding='utf-8')
    print(f'נשמר: {clean_path}')

    gold = _build_gold_3class(df, ['not_rumour', 'rumour', 'unrelated'],
                              per_class_cap, random_state)
    print(f'\nGold Standard: {len(gold):,}')
    print(gold['label'].value_counts().to_string())

    _split_and_save(gold, gold_dir, 'pheme', random_state)
    print('PHEME: הושלם.')


# ──────────────────────────────────────────────
# Legacy wrapper — kept for backward compatibility
# ──────────────────────────────────────────────

def run_preprocessing(input_path: str, output_dir: str, gold_dir: str,
                      min_words: int = 5, max_chars: int = 350,
                      per_class_cap: int = PER_CLASS_CAP, random_state: int = 42):
    """Backward-compatible wrapper that calls run_preprocessing_manchester."""
    run_preprocessing_manchester(
        input_path=input_path,
        output_dir=output_dir,
        gold_dir=gold_dir,
        min_words=min_words,
        max_chars=max_chars,
        per_class_cap=per_class_cap,
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

    print('\n✅ כל ה-datasets עובדו בהצלחה (3-class).')
