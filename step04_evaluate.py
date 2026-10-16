"""
Ευφυή Συστήματα — 1η Εργασία (IRIS, ταξινομητής TSK/ANFIS)
ΒΗΜΑ 4: Τελική αξιολόγηση στο Testing set.

Έξοδος -> κλάση:  κλάση = clip(round(y), 1, 3)
  δηλαδή  y < 1.5 -> 1 (setosa),  1.5 <= y < 2.5 -> 2 (versicolor),  y >= 2.5 -> 3 (virginica)

Μετρικές:  accuracy, confusion matrix, λίστα λάθος ταξινομημένων δειγμάτων.

Τρέξιμο:  python step04_evaluate.py      (χρειάζεται πρώτα το step03_train.py)
Βγάζει:   results/confusion_*.png, results/outputs_*.png, results/results.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix

from step02_anfis import ANFIS, load
from step03_train import FEATURES, GRAY, INK, style

RESULTS = Path(__file__).parent / "results"
CLASSES = ["setosa", "versicolor", "virginica"]
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # μπλε, πορτοκαλί, πράσινο
CLASS_MARKERS = ["o", "s", "^"]                           # και σχήμα, όχι μόνο χρώμα


def load_model(n_rules, X_tr):
    """Φόρτωση αποθηκευμένου μοντέλου από το βήμα 3."""
    data = np.load(RESULTS / f"model_{n_rules}rules.npz")
    model = ANFIS(X_tr, n_mf=int(data["n_mf"]))
    model.set_params((data["centers"], data["sigmas"], data["consequents"]))
    return model


def to_class(y_out):
    """Αριθμητική έξοδος ANFIS -> κλάση 1, 2 ή 3."""
    return np.clip(np.round(y_out), 1, 3).astype(int)


def report_errors(X, y_true, y_out, y_pred):
    """Γραμμές κειμένου για κάθε λάθος δείγμα: χαρακτηριστικά, έξοδος, απόσταση από το όριο."""
    lines = []
    for i in np.where(y_pred != y_true)[0]:
        boundary = 1.5 if min(y_true[i], y_pred[i]) == 1 else 2.5   # όριο ανάμεσα στις δύο κλάσεις
        lines.append(f"  δείγμα #{i:>2}: x = {X[i]}  πραγματική = {CLASSES[int(y_true[i]) - 1]:<10}"
                     f" πρόβλεψη = {CLASSES[y_pred[i] - 1]:<10} έξοδος = {y_out[i]:.3f}"
                     f" (απόσταση από όριο {boundary}: {abs(y_out[i] - boundary):.3f})")
    return lines or ["  (κανένα λάθος)"]


def plot_confusion(cm, title, path):
    """Confusion matrix: γραμμή = πραγματική κλάση, στήλη = πρόβλεψη."""
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#184f95"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    for r in range(3):
        for c in range(3):
            dark = cm[r, c] > cm.max() / 2                # σκούρο κελί -> λευκά γράμματα
            ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=13,
                    color="white" if dark else INK)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set(xlabel="Πρόβλεψη", ylabel="Πραγματική κλάση", title=title)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_outputs(y_true, y_out, title, path):
    """Η αριθμητική έξοδος κάθε δείγματος Testing, με τα όρια 1.5 και 2.5."""
    order = np.argsort(y_true, kind="stable")             # ταξινόμηση ανά πραγματική κλάση
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for b in (1.5, 2.5):
        ax.axhline(b, color=GRAY, lw=1, ls="--")
    for k in (1, 2, 3):
        idx = [j for j, i in enumerate(order) if y_true[i] == k]
        ax.scatter(idx, y_out[order][idx], s=40, color=CLASS_COLORS[k - 1],
                   marker=CLASS_MARKERS[k - 1], edgecolor="white", lw=1, label=CLASSES[k - 1])
    wrong = [j for j, i in enumerate(order) if to_class(y_out[i]) != y_true[i]]
    ax.scatter(wrong, y_out[order][wrong], s=180, facecolor="none", edgecolor=INK, lw=1.5,
               label="λάθος")
    for j in wrong:                                       # ετικέτα με τον αριθμό του δείγματος
        ax.annotate(f"#{order[j]}", (j, y_out[order][j]), xytext=(10, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set(xlabel="Δείγμα Testing (ταξινομημένα ανά κλάση)", ylabel="Έξοδος ANFIS",
           title=title, yticks=[1, 1.5, 2, 2.5, 3])
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate(n_rules, sets):
    """Αξιολόγηση ενός μοντέλου: accuracy σε όλα τα σύνολα, λεπτομέρειες στο Testing."""
    model = load_model(n_rules, sets["Training"][0])
    lines = [f"===== ANFIS {n_rules} κανόνες ====="]
    for name, (X, y) in sets.items():
        acc = accuracy_score(y, to_class(model.forward(X)))
        lines.append(f"Accuracy {name:<9}: {acc * 100:6.2f}%  ({len(y)} δείγματα)")
    X_te, y_te = sets["Testing"]
    y_out = model.forward(X_te)
    y_pred = to_class(y_out)
    cm = confusion_matrix(y_te, y_pred, labels=[1, 2, 3])
    lines += ["Confusion matrix (Testing), γραμμή = πραγματική, στήλη = πρόβλεψη:", str(cm),
              "Λάθος ταξινομήσεις (Testing):"] + report_errors(X_te, y_te, y_out, y_pred)
    plot_confusion(cm, f"Testing — {n_rules} κανόνες", RESULTS / f"confusion_{n_rules}rules.png")
    plot_outputs(y_te, y_out, f"Έξοδος ANFIS στο Testing — {n_rules} κανόνες",
                 RESULTS / f"outputs_{n_rules}rules.png")
    return lines


if __name__ == "__main__":
    sets = {name: load(f) for name, f in
            [("Training", "train"), ("Checking", "check"), ("Testing", "test")]}
    lines = [f"Χαρακτηριστικά x = {FEATURES}", ""]
    for n_rules in (16, 81):
        lines += evaluate(n_rules, sets) + [""]
    text = "\n".join(lines)
    print(text)
    (RESULTS / "results.txt").write_text(text + "\n", encoding="utf-8")
