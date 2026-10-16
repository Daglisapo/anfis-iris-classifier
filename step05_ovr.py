"""
Ευφυή Συστήματα — 1η Εργασία (IRIS, ταξινομητής TSK/ANFIS)
ΒΗΜΑ 5 (προαιρετικό): Σύγκριση με 3 ANFIS one-vs-rest.

Ιδέα: ένα ANFIS ανά κλάση k, με στόχο  t = 1 αν το δείγμα είναι κλάσης k, αλλιώς 0.
Πρόβλεψη: η κλάση με τη μεγαλύτερη έξοδο ->  κλάση = argmax(y_1, y_2, y_3) + 1

Ξαναχρησιμοποιούμε αυτούσια τα ANFIS (βήμα 2) και train() (βήμα 3).
Τρέξιμο:  python step05_ovr.py
Βγάζει:   results/confusion_ovr.png, results/results_ovr.txt
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from step02_anfis import ANFIS, load
from step03_train import train
from step04_evaluate import CLASSES, RESULTS, plot_confusion


def train_ovr(X_tr, y_tr, X_ch, y_ch):
    """Εκπαίδευση 3 ANFIS (16 κανόνες το καθένα), ένα για κάθε κλάση."""
    models = []
    for k in (1, 2, 3):
        t_tr = (y_tr == k).astype(float)                  # 1 = «είναι κλάση k», 0 = «δεν είναι»
        t_ch = (y_ch == k).astype(float)
        model = ANFIS(X_tr, n_mf=2)
        _, best = train(model, X_tr, t_tr, X_ch, t_ch)    # ίδια εκπαίδευση με το βήμα 3
        print(f"  ANFIS «{CLASSES[k - 1]} vs rest»: καλύτερη epoch {best:>3}, "
              f"RMSE Checking {model.rmse(X_ch, t_ch):.4f}")
        models.append(model)
    return models


def scores(models, X):
    """Έξοδοι των 3 μοντέλων, μία στήλη ανά κλάση.  shape (N, 3)"""
    return np.column_stack([m.forward(X) for m in models])


def predict(models, X):
    """Κερδίζει το μοντέλο με τη μεγαλύτερη έξοδο (argmax). +1 γιατί οι κλάσεις είναι 1..3."""
    return scores(models, X).argmax(axis=1) + 1


def report_errors(X, y_true, S, y_pred):
    """Για κάθε λάθος: χαρακτηριστικά και οι 3 έξοδοι (ποιος «ψήφισε» τι)."""
    lines = []
    for i in np.where(y_pred != y_true)[0]:
        lines.append(f"  δείγμα #{i:>2}: x = {X[i]}  πραγματική = {CLASSES[int(y_true[i]) - 1]:<10}"
                     f" πρόβλεψη = {CLASSES[y_pred[i] - 1]:<10} έξοδοι [set, ver, vir] = {S[i].round(3)}")
    return lines or ["  (κανένα λάθος)"]


if __name__ == "__main__":
    sets = {name: load(f) for name, f in
            [("Training", "train"), ("Checking", "check"), ("Testing", "test")]}
    print("Εκπαίδευση:")
    models = train_ovr(*sets["Training"], *sets["Checking"])

    lines = ["===== 3 × ANFIS one-vs-rest (16 κανόνες το καθένα) ====="]
    for name, (X, y) in sets.items():
        acc = accuracy_score(y, predict(models, X))
        lines.append(f"Accuracy {name:<9}: {acc * 100:6.2f}%  ({len(y)} δείγματα)")
    X_te, y_te = sets["Testing"]
    S, y_pred = scores(models, X_te), predict(models, X_te)
    cm = confusion_matrix(y_te, y_pred, labels=[1, 2, 3])
    lines += ["Confusion matrix (Testing), γραμμή = πραγματική, στήλη = πρόβλεψη:", str(cm),
              "Λάθος ταξινομήσεις (Testing):"] + report_errors(X_te, y_te, S, y_pred)

    plot_confusion(cm, "Testing — 3 × ANFIS one-vs-rest", RESULTS / "confusion_ovr.png")
    text = "\n".join(lines)
    print("\n" + text)
    (RESULTS / "results_ovr.txt").write_text(text + "\n", encoding="utf-8")
