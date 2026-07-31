"""
Experiment: "Traditional" full-data training vs the Gold Standard approach.

Trains RoBERTa on the FULL cleaned Manchester data (natural ~0.7% misinformation
prevalence) instead of the balanced Gold Standard, and evaluates on:
  (A) the SAME gold-standard test set (365)  -> apples-to-apples vs F1=0.986
  (B) a natural-distribution held-out test   -> shows the misleading high accuracy

Fairness: uses the SAME class-weighted CrossEntropyLoss as the Gold Standard model,
so the ONLY thing that changes is the training-data composition.

Outputs -> results/_fulldata_experiment/  (does NOT touch existing results)
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset
from transformers import (RobertaTokenizerFast, RobertaForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback, set_seed)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

SEED=42; MODEL='roberta-base'; MAX_LEN=128; BATCH=16; EPOCHS=3
LABEL_MAP={'reliable':0,'misinformation':1}; ID2={0:'reliable',1:'misinformation'}
NAMES=['reliable','misinformation']; TEXT='cleaned_tweet'; LAB='label'
set_seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',DEV, torch.cuda.get_device_name(0) if DEV.type=='cuda' else '')

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'_fulldata_experiment'; OUT.mkdir(parents=True,exist_ok=True)

# ── Load ──
full=pd.read_csv(ROOT/'data'/'processed'/'manchester_clean.csv')
full=full[full[LAB].isin(LABEL_MAP)].dropna(subset=[TEXT,LAB]).copy()
full[TEXT]=full[TEXT].astype(str)
gold_test=pd.read_csv(ROOT/'data'/'gold_standard'/'manchester_test.csv').dropna(subset=[TEXT,LAB])
gold_test[TEXT]=gold_test[TEXT].astype(str)

# Remove gold-test tweets from the full pool (prevent leakage)
test_set=set(gold_test[TEXT].str.strip())
pool=full[~full[TEXT].str.strip().isin(test_set)].copy()
print(f"Full cleaned: {len(full):,}  | pool after removing gold-test: {len(pool):,}")
print(f"Pool labels: {dict(pool[LAB].value_counts())}")

# Carve a NATURAL-distribution held-out test (5%) to show inflated accuracy
pool_tr, nat_test = train_test_split(pool, test_size=0.05, stratify=pool[LAB], random_state=SEED)
# train/val for early stopping
tr, val = train_test_split(pool_tr, test_size=0.1, stratify=pool_tr[LAB], random_state=SEED)
print(f"train={len(tr):,}  val={len(val):,}  natural-test={len(nat_test):,} "
      f"(misinfo in nat-test: {(nat_test[LAB]=='misinformation').sum()})")
print(f"train misinformation fraction: {(tr[LAB]=='misinformation').mean()*100:.2f}%")

tok=RobertaTokenizerFast.from_pretrained(MODEL)
def enc(df): return np.array([LABEL_MAP[l] for l in df[LAB]])
class DS(Dataset):
    def __init__(s,texts,labels):
        e=tok(list(texts),truncation=True,padding='max_length',max_length=MAX_LEN,return_tensors='pt')
        s.ii=e['input_ids']; s.am=e['attention_mask']; s.y=torch.tensor(labels,dtype=torch.long)
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return {'input_ids':s.ii[i],'attention_mask':s.am[i],'labels':s.y[i]}

class WTrainer(Trainer):
    def __init__(s,cw,*a,**k): super().__init__(*a,**k); s.cw=cw.to(DEV)
    def compute_loss(s,model,inputs,return_outputs=False,**k):
        y=inputs.pop('labels'); out=model(**inputs)
        loss=nn.CrossEntropyLoss(weight=s.cw)(out.logits,y)
        return (loss,out) if return_outputs else loss

def metrics(ep):
    lg,lb=ep; pr=np.argmax(lg,axis=-1)
    return {'accuracy':accuracy_score(lb,pr),'f1_macro':f1_score(lb,pr,average='macro')}

print('\nTokenizing + building datasets...')
ds_tr=DS(tr[TEXT].values,enc(tr)); ds_val=DS(val[TEXT].values,enc(val))
ds_gold=DS(gold_test[TEXT].values,enc(gold_test)); ds_nat=DS(nat_test[TEXT].values,enc(nat_test))

cw_arr=compute_class_weight('balanced',classes=np.array([0,1]),y=enc(tr))
print(f"Class weights: reliable={cw_arr[0]:.3f}  misinformation={cw_arr[1]:.3f}")
cw=torch.tensor(cw_arr,dtype=torch.float)

args=TrainingArguments(output_dir=str(OUT/'ckpt'),num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,per_device_eval_batch_size=BATCH*2,
    learning_rate=2e-5,weight_decay=0.01,warmup_ratio=0.1,
    eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
    metric_for_best_model='f1_macro',greater_is_better=True,logging_steps=200,
    seed=SEED,fp16=torch.cuda.is_available(),report_to='none',dataloader_num_workers=0)

model=RobertaForSequenceClassification.from_pretrained(MODEL,num_labels=2,id2label=ID2,label2id=LABEL_MAP)
trainer=WTrainer(cw,model=model,args=args,train_dataset=ds_tr,eval_dataset=ds_val,
                 compute_metrics=metrics,callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
print('\n=== Training on FULL data ===')
trainer.train()

def evaluate(ds, y_true, tag):
    out=trainer.predict(ds); pr=np.argmax(out.predictions,axis=-1)
    acc=accuracy_score(y_true,pr); f1m=f1_score(y_true,pr,average='macro')
    rec_m=recall_score(y_true,pr,pos_label=1,zero_division=0)
    pre_m=precision_score(y_true,pr,pos_label=1,zero_division=0)
    f1_m=f1_score(y_true,pr,pos_label=1,zero_division=0)
    print(f"\n--- {tag} ---")
    print(classification_report(y_true,pr,target_names=NAMES,digits=4,zero_division=0))
    print(f"  CM:\n{confusion_matrix(y_true,pr)}")
    return {'eval':tag,'n':len(y_true),'accuracy':acc,'f1_macro':f1m,
            'precision_misinfo':pre_m,'recall_misinfo':rec_m,'f1_misinfo':f1_m}

r_gold=evaluate(ds_gold,enc(gold_test),'FULL-DATA model on GOLD test (n=365)')
r_nat =evaluate(ds_nat, enc(nat_test), f'FULL-DATA model on NATURAL test (n={len(nat_test)})')

res=pd.DataFrame([r_gold,r_nat])
res.to_csv(OUT/'fulldata_results.csv',index=False)

print('\n'+'='*70)
print(' COMPARISON SUMMARY')
print('='*70)
print(f"{'Setup':<42}{'F1_macro':>10}{'Acc':>9}{'Recall_misinfo':>16}")
print(f"{'Gold Standard model (our approach)':<42}{0.9857:>10.4f}{0.9918:>9.4f}{0.9830:>16.4f}")
print(f"{'Full-data model -> GOLD test':<42}{r_gold['f1_macro']:>10.4f}{r_gold['accuracy']:>9.4f}{r_gold['recall_misinfo']:>16.4f}")
print(f"{'Full-data model -> NATURAL test':<42}{r_nat['f1_macro']:>10.4f}{r_nat['accuracy']:>9.4f}{r_nat['recall_misinfo']:>16.4f}")
print('='*70)
print(f"Results saved: {OUT/'fulldata_results.csv'}")
