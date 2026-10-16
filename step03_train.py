"""
Ευφυή Συστήματα — 1η Εργασία (IRIS, ταξινομητής TSK/ANFIS)
ΒΗΜΑ 3: Εκπαίδευση ANFIS με υβριδική μάθηση + παρακολούθηση με το Checking set.

Κάθε epoch:
  1. least squares (ridge) για τα consequents p, r
  2. RMSE σε Training και Checking
  3. αν το Checking RMSE είναι το μικρότερο ως τώρα -> κρατάμε αντίγραφο των παραμέτρων
  4. ένα gradient step στα c, σ
  5. αν το Training RMSE ανέβηκε σε σχέση με την προηγούμενη epoch -> lr = lr / 2

Το Testing ΔΕΝ χρησιμοποιείται εδώ (μόνο στο βήμα 4).

Τρέξιμο:  python step03_train.py
Βγάζει:   results/errors_*.png, results/mfs_*.png, results/model_*.npz, results/train_summary.txt
"""
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # σχεδίαση σε αρχείο, χωρίς παράθυρο
import matplotlib.pyplot as plt
import numpy as np

from step02_anfis import ANFIS, load

EPOCHS = 100
LR = 5.0                                  # αρχικό βήμα gradient descent
LAM = 0.01                                # ridge λ στο least squares
RESULTS = Path(__file__).parent / "results"
FEATURES = ["Μήκος σέπαλου (cm)", "Πλάτος σέπαλου (cm)",
            "Μήκος πετάλου (cm)", "Πλάτος πετάλου (cm)"]
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9a9a94", "#3b3b38"


def train(model, X_tr, y_tr, X_ch, y_ch):
    """Υβριδική εκπαίδευση. Επιστρέφει ιστορικό σφαλμάτων και την καλύτερη epoch."""
    hist = {"train": [], "check": [], "lr": []}
    lr, best_err, best_epoch, best_params = LR, np.inf, 0, None
    for epoch in range(1, EPOCHS + 1):
        model.fit_consequents(X_tr, y_tr, lam=LAM)
        e_tr, e_ch = model.rmse(X_tr, y_tr), model.rmse(X_ch, y_ch)
        if hist["train"] and e_tr > hist["train"][-1]:    # το σφάλμα ανέβηκε -> βήμα πολύ μεγάλο
            lr /= 2
        hist["train"].append(e_tr)
        hist["check"].append(e_ch)
        hist["lr"].append(lr)
        if e_ch < best_err:                               # νέο ελάχιστο Checking -> κράτα το
            best_err, best_epoch, best_params = e_ch, epoch, model.get_params()
        model.gradient_step(X_tr, y_tr, lr=lr)
    model.set_params(best_params)                         # επιστροφή στην καλύτερη epoch
    return hist, best_epoch


def plot_errors(hist, best_epoch, title, path):
    """Καμπύλες RMSE Training/Checking ανά epoch, με σημάδι στην καλύτερη epoch."""
    epochs = np.arange(1, len(hist["train"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(epochs, hist["train"], color=BLUE, lw=2, label="Training")
    ax.plot(epochs, hist["check"], color=ORANGE, lw=2, label="Checking")
    ax.axvline(best_epoch, color=GRAY, lw=1, ls=":")
    near_end = best_epoch > 0.7 * len(epochs)             # κοντά στο δεξί άκρο -> ετικέτα αριστερά
    ax.annotate(f"καλύτερη epoch = {best_epoch}", (best_epoch, max(hist["check"])),
                xytext=(-4 if near_end else 4, -2), textcoords="offset points",
                ha="right" if near_end else "left", color=INK, fontsize=9, va="top")
    ax.set(xlabel="Epoch", ylabel="RMSE", title=title)
    style(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mfs(before, after, X_tr, path):
    """Οι Gaussian MFs κάθε εισόδου πριν (διακεκομμένες) και μετά (συνεχείς) την εκπαίδευση."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    for i, ax in enumerate(axes.flat):
        x = np.linspace(X_tr[:, i].min() - 0.5, X_tr[:, i].max() + 0.5, 300)
        for m in range(after.n_mf):
            for model, color, ls in [(before, GRAY, "--"), (after, BLUE, "-")]:
                c, s = model.centers[i, m], model.sigmas[i, m]
                ax.plot(x, np.exp(-(x - c) ** 2 / (2 * s ** 2)), color=color, lw=2, ls=ls)
        ax.plot(X_tr[:, i], np.zeros(len(X_tr)), "|", color=INK, ms=6, alpha=0.4)  # τα δεδομένα
        ax.set(xlabel=FEATURES[i], ylabel="μ(x)", ylim=(-0.05, 1.05))
        style(ax)
    axes[0, 0].plot([], [], color=GRAY, lw=2, ls="--", label="πριν")
    axes[0, 0].plot([], [], color=BLUE, lw=2, label="μετά")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def style(ax):
    """Διακριτικοί άξονες και πλέγμα, ώστε να ξεχωρίζουν τα δεδομένα."""
    ax.grid(color="#e4e4e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def run(n_mf, X_tr, y_tr, X_ch, y_ch):
    """Εκπαίδευση ενός μοντέλου, γραφήματα, αποθήκευση. Επιστρέφει γραμμή σύνοψης."""
    model = ANFIS(X_tr, n_mf=n_mf)
    before = copy.deepcopy(model)                         # αντίγραφο για το γράφημα «πριν»
    hist, best = train(model, X_tr, y_tr, X_ch, y_ch)
    tag = f"{model.n_rules}rules"
    plot_errors(hist, best, f"ANFIS {model.n_rules} κανόνες ({n_mf} MF/είσοδο)",
                RESULTS / f"errors_{tag}.png")
    plot_mfs(before, model, X_tr, RESULTS / f"mfs_{tag}.png")
    np.savez(RESULTS / f"model_{tag}.npz", n_mf=n_mf, centers=model.centers,
             sigmas=model.sigmas, consequents=model.consequents)
    return (f"{model.n_rules:>3} κανόνες | καλύτερη epoch {best:>3} | "
            f"RMSE Training {model.rmse(X_tr, y_tr):.4f} | "
            f"RMSE Checking {model.rmse(X_ch, y_ch):.4f} | τελικό lr {hist['lr'][-1]:.3g}")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    X_tr, y_tr = load("train")
    X_ch, y_ch = load("check")
    lines = [f"Ρυθμίσεις: epochs={EPOCHS}, αρχικό lr={LR}, ridge λ={LAM}"]
    for n_mf in (2, 3):                                   # 16 και 81 κανόνες
        lines.append(run(n_mf, X_tr, y_tr, X_ch, y_ch))
        print(lines[-1])
    (RESULTS / "train_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nΑποθηκεύτηκαν στο {RESULTS}")
