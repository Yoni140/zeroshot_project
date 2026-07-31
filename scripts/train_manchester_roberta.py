"""
Train RoBERTa on Manchester dataset (5-fold CV + final model).
Run from project root: python scripts/train_manchester_roberta.py

NOTE: This file is now a thin wrapper around the canonical trainer
`train_manchester_roberta_smoothed.py`, which is the improved version
(class weighting + label smoothing + OvR ROC/PR), updated for the 3-class
scheme (reliable / misinformation / unrelated). Kept so the historical
entry-point name still works and produces identical, up-to-date results.
"""
import runpy
from pathlib import Path

if __name__ == '__main__':
    target = Path(__file__).with_name('train_manchester_roberta_smoothed.py')
    runpy.run_path(str(target), run_name='__main__')
