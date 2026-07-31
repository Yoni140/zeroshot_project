"""
Near-duplicate leakage audit for Manchester (train+val vs test).
Checks whether RoBERTa's high test F1 is inflated by near-duplicate tweets
that cross the train/test boundary.

Method:
  1. Lexical:  TF-IDF (char 3-5 grams) cosine, test vs train+val.
  2. Semantic: sentence-embeddings (all-MiniLM-L6-v2) cosine, test vs train+val.
  3. For each test tweet, take its MAX similarity to any train tweet.
  4. Flag test tweets above thresholds (0.80 / 0.90 / 0.95 / exact).
  5. Re-score RoBERTa test F1 EXCLUDING flagged test tweets.
     If F1 barely moves -> result is NOT driven by leakage.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / 'data' / 'gold_standard'
P = ROOT / 'results' / 'predictions'
TEXT = 'cleaned_tweet'

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.concat([pd.read_csv(G / 'manchester_train.csv'),
                   pd.read_csv(G / 'manchester_val.csv')], ignore_index=True)
test  = pd.read_csv(G / 'manchester_test.csv')
preds = pd.read_csv(P / 'manchester_roberta_test_predictions.csv')

train_txt = train[TEXT].astype(str).tolist()
test_txt  = test[TEXT].astype(str).tolist()
print(f"train+val: {len(train_txt):,}   test: {len(test_txt):,}")

# Align predictions to test rows (same order, produced from manchester_test.csv)
n = min(len(test), len(preds))
y_true = preds['true_label_int'].values[:n]
y_pred = preds['pred_label_int'].values[:n]
base_f1 = f1_score(y_true, y_pred, average='macro')
base_acc = accuracy_score(y_true, y_pred)
print(f"Baseline (all {n} test tweets):  F1 Macro={base_f1:.4f}  Acc={base_acc:.4f}")

# ── Exact duplicate check ─────────────────────────────────────────────────────
train_set = set(t.strip().lower() for t in train_txt)
exact = np.array([t.strip().lower() in train_set for t in test_txt[:n]])
print(f"\nExact duplicates (test tweet appears verbatim in train+val): {exact.sum()}")

# ── Lexical similarity: TF-IDF char n-grams ──────────────────────────────────
print("\n[1/2] Lexical TF-IDF (char 3-5 gram) cosine...")
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X = vec.fit_transform(train_txt + test_txt[:n])
Xtr, Xte = X[:len(train_txt)], X[len(train_txt):]
# max cosine of each test row vs all train rows (sparse dot = cosine on L2-normed tfidf)
lex_sim = (Xte @ Xtr.T).max(axis=1).toarray().ravel()

# ── Semantic similarity: sentence embeddings ─────────────────────────────────
print("[2/2] Semantic embeddings (all-MiniLM-L6-v2) cosine...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
emb_tr = model.encode(train_txt, normalize_embeddings=True, show_progress_bar=False, batch_size=128)
emb_te = model.encode(test_txt[:n], normalize_embeddings=True, show_progress_bar=False, batch_size=128)
sem_sim = (emb_te @ emb_tr.T).max(axis=1)

# ── Report at thresholds + re-score ──────────────────────────────────────────
def rescore(mask_keep, label):
    if mask_keep.sum() < 2 or len(np.unique(y_true[mask_keep])) < 2:
        return f"  {label:<42} kept={mask_keep.sum():3d}  (too few/!2 classes to score)"
    f1 = f1_score(y_true[mask_keep], y_pred[mask_keep], average='macro')
    return f"  {label:<42} kept={mask_keep.sum():3d}  F1 Macro={f1:.4f}  (delta {f1-base_f1:+.4f})"

print("\n" + "="*70)
print(" LEAKAGE AUDIT RESULTS — Manchester (test vs train+val)")
print("="*70)
print(f"\nBaseline F1 Macro (all test): {base_f1:.4f}\n")

for name, sim in [('Lexical (TF-IDF char)', lex_sim), ('Semantic (embeddings)', sem_sim)]:
    print(f"--- {name} ---")
    print(f"  max-sim distribution: min={sim.min():.3f}  median={np.median(sim):.3f}  "
          f"p90={np.percentile(sim,90):.3f}  max={sim.max():.3f}")
    for thr in [0.95, 0.90, 0.80]:
        flagged = sim >= thr
        keep = ~flagged
        print(f"  >= {thr:.2f}: flagged {flagged.sum():3d}/{n} test tweets "
              f"({flagged.sum()/n*100:.1f}%)")
        print(rescore(keep, f"    re-score excluding >= {thr:.2f}:"))
    print()

# Combined: flag if EITHER lexical or semantic >= 0.90
combo = (lex_sim >= 0.90) | (sem_sim >= 0.90)
print("--- Combined (lexical OR semantic >= 0.90) ---")
print(f"  flagged {combo.sum()}/{n} ({combo.sum()/n*100:.1f}%)")
print(rescore(~combo, "  re-score excluding combined flags:"))

# Save flagged examples for inspection
out = test.iloc[:n].copy()
out['lex_max_sim'] = lex_sim
out['sem_max_sim'] = sem_sim
out['correct'] = (y_true == y_pred)
out_path = P / 'manchester_leakage_audit.csv'
out[[TEXT, 'label', 'lex_max_sim', 'sem_max_sim', 'correct']].sort_values('sem_max_sim', ascending=False).to_csv(out_path, index=False)
print(f"\nFull audit table saved: {out_path}")
print("="*70)
