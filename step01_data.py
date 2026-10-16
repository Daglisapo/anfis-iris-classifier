"""
Ευφυή Συστήματα — 1η Εργασία (IRIS, ταξινομητής TSK/ANFIS)
ΒΗΜΑ 1: Φόρτωση δεδομένων + stratified διαχωρισμός σε Training / Checking / Testing.

Τι ζητάει η εκφώνηση:
  - τυχαίος διαχωρισμός σε 3 σύνολα, με διατήρηση της αναλογίας κλάσεων (stratified)
  - να αναφερθούν τα ποσοστά που χρησιμοποιήθηκαν
Εδώ: 70% Training, 15% Checking, 15% Testing.

Τρέξιμο:  python step01_data.py
Βγάζει:   data/train.csv, data/check.csv, data/test.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

SEED = 42                 # ίδιο seed -> ίδιος διαχωρισμός κάθε φορά (αναπαραγωγιμότητα)
P_TRAIN, P_CHECK, P_TEST = 0.70, 0.15, 0.15

# 1) Φόρτωση: 150 δείγματα, 4 χαρακτηριστικά, 3 κλάσεις (50 η καθεμία)
iris = load_iris()
X = iris.data                      # sepal length, sepal width, petal length, petal width (cm)
y = iris.target + 1                # κλάσεις 1, 2, 3 (όπως στην εκφώνηση)
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 2) Πρώτα κόβουμε το Testing (15%), μετά το υπόλοιπο σε Training/Checking
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=P_TEST, stratify=y, random_state=SEED)
X_train, X_check, y_train, y_check = train_test_split(
    X_rest, y_rest, test_size=P_CHECK / (P_TRAIN + P_CHECK), stratify=y_rest, random_state=SEED)

# 3) Έλεγχος: πόσα δείγματα από κάθε κλάση πήγαν σε κάθε σύνολο
def counts(v):
    return {c: int((v == c).sum()) for c in (1, 2, 3)}

print(f"{'Σύνολο':<10}{'N':>5}   ανά κλάση {{1,2,3}}")
for name, v in [("Training", y_train), ("Checking", y_check), ("Testing", y_test)]:
    print(f"{name:<10}{len(v):>5}   {counts(v)}")

# 4) Αποθήκευση για τα επόμενα βήματα
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
for name, Xs, ys in [("train", X_train, y_train), ("check", X_check, y_check), ("test", X_test, y_test)]:
    df = pd.DataFrame(Xs, columns=features)
    df["class"] = ys
    df.to_csv(out / f"{name}.csv", index=False)
print(f"\nΑποθηκεύτηκαν στο {out}")
