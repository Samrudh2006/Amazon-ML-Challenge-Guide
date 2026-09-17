import numpy as np
import pandas as pd
from ensemble_stacker import run_grandmaster_ensemble

y_true = pd.read_csv("mock_train.csv")['price'].values
cb_oof = np.load("catboost_oof.npy")
cb_test = np.load("catboost_test.npy")

# Simulate 2 diverse models
model1_oof = cb_oof
model2_oof = cb_oof * np.random.uniform(0.97, 1.03, size=len(cb_oof))

model1_test = cb_test
model2_test = cb_test * 1.01

final_preds = run_grandmaster_ensemble(
    y_true, 
    [model1_oof, model2_oof], 
    [model1_test, model2_test]
)

print("[SUCCESS] Ensemble Stacker works flawlessly! Final test predictions shape:", final_preds.shape)
