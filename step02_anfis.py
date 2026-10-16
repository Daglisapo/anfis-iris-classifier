"""
Ευφυή Συστήματα — 1η Εργασία (IRIS, ταξινομητής TSK/ANFIS)
ΒΗΜΑ 2: Η κλάση ANFIS (ασαφές μοντέλο TSK 1ης τάξης), γραμμένη από εμάς σε numpy.

Δομή (5 στρώματα):
  1. Ασαφοποίηση:     μ(x) = exp(-(x-c)^2 / (2σ^2))        (Gaussian MF)
  2. Ισχύς κανόνα:    w = μ1 · μ2 · μ3 · μ4                (AND = γινόμενο)
  3. Κανονικοποίηση:  w̄ = w / Σw
  4. Consequent:      f = p1·x1 + p2·x2 + p3·x3 + p4·x4 + r   (TSK 1ης τάξης)
  5. Έξοδος:          y = Σ w̄·f

Υβριδική μάθηση:
  - consequents (p, r)  -> least squares (ακριβής λύση, γιατί η y είναι γραμμική ως προς αυτά)
  - premise (c, σ)      -> gradient descent με αριθμητικό gradient (κεντρικές διαφορές)

Τρέξιμο (μικρό τεστ):  python step02_anfis.py
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


class ANFIS:
    def __init__(self, X_train, n_mf=2):
        """Αρχικοποίηση από το εύρος κάθε εισόδου στο Training."""
        self.n_inputs = X_train.shape[1]          # 4 είσοδοι
        self.n_mf = n_mf                          # MF ανά είσοδο (2 ή 3)
        lo, hi = X_train.min(axis=0), X_train.max(axis=0)

        # Κέντρα: ισαπέχοντα από min ως max κάθε εισόδου.  shape (4, n_mf)  ≈ double[4][n_mf]
        self.centers = np.linspace(lo, hi, n_mf).T
        # Πλάτη: μισή απόσταση ανάμεσα σε διπλανά κέντρα -> οι καμπύλες επικαλύπτονται
        width = (hi - lo) / (2 * (n_mf - 1))
        self.sigmas = np.tile(width, (n_mf, 1)).T

        # Κανόνες (grid partition): όλοι οι συνδυασμοί MF, π.χ. (0,1,0,1) = «μικρό, μεγάλο, μικρό, μεγάλο»
        self.rules = np.array(list(product(range(n_mf), repeat=self.n_inputs)))
        self.n_rules = len(self.rules)            # 2^4 = 16  ή  3^4 = 81

        # Consequents: για κάθε κανόνα [p1, p2, p3, p4, r].  shape (n_rules, 5)
        self.consequents = np.zeros((self.n_rules, self.n_inputs + 1))

    # ---------- Στρώματα 1–3 ----------
    def memberships(self, X):
        """Στρώμα 1: βαθμός συμμετοχής κάθε δείγματος σε κάθε MF.  shape (N, 4, n_mf)"""
        diff = X[:, :, None] - self.centers[None, :, :]       # broadcasting: (N,4,1) - (1,4,n_mf)
        return np.exp(-diff ** 2 / (2 * self.sigmas[None, :, :] ** 2))

    def normalized_strengths(self, X):
        """Στρώματα 2–3: ισχύς κάθε κανόνα, κανονικοποιημένη.  shape (N, n_rules)"""
        mu = self.memberships(X)
        w = np.ones((X.shape[0], self.n_rules))
        for r, combo in enumerate(self.rules):                # για κάθε κανόνα
            for i, m in enumerate(combo):                     # για κάθε είσοδο i, MF m
                w[:, r] *= mu[:, i, m]                        # AND = γινόμενο
        return w / (w.sum(axis=1, keepdims=True) + 1e-12)     # +1e-12: αποφυγή διαίρεσης με 0

    # ---------- Στρώματα 4–5 ----------
    def forward(self, X):
        """Αριθμητική έξοδος του μοντέλου για κάθε δείγμα.  shape (N,)"""
        wn = self.normalized_strengths(X)
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])         # [x1..x4, 1] -> το 1 πολλαπλασιάζει το r
        f = Xa @ self.consequents.T                           # f κάθε κανόνα, shape (N, n_rules)
        return (wn * f).sum(axis=1)                           # σταθμισμένος μέσος

    def mse(self, X, y):
        return np.mean((self.forward(X) - y) ** 2)

    def rmse(self, X, y):
        return np.sqrt(self.mse(X, y))

    # ---------- Μάθηση ----------
    def fit_consequents(self, X, y, lam=0.01):
        """Least squares με ridge για τα p, r (με σταθερά c, σ).

        Λύνουμε  min ||A·θ - y||^2 + λ·||θ||^2   ->   (AᵀA + λI)·θ = Aᵀy
        Το λ «τιμωρεί» τους τεράστιους συντελεστές: χωρίς αυτό, με 80 συντελεστές για
        104 δείγματα, το μοντέλο αποστηθίζει το Training (overfitting).
        """
        wn = self.normalized_strengths(X)
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])
        # Κάθε κανόνας r συνεισφέρει στήλες w̄_r·[x1..x4, 1] -> A έχει n_rules*5 στήλες
        A = np.hstack([wn[:, [r]] * Xa for r in range(self.n_rules)])
        I = np.eye(A.shape[1])
        theta = np.linalg.solve(A.T @ A + lam * I, A.T @ y)
        self.consequents = theta.reshape(self.n_rules, self.n_inputs + 1)

    def gradient_step(self, X, y, lr=5.0, h=1e-4):
        """Ένα βήμα gradient descent στα c, σ με αριθμητικό gradient (κεντρικές διαφορές)."""
        grads = []
        for arr in (self.centers, self.sigmas):
            g = np.zeros_like(arr)
            for idx in np.ndindex(arr.shape):                 # για κάθε παράμετρο του πίνακα
                old = arr[idx]
                arr[idx] = old + h
                e_plus = self.mse(X, y)                       # σφάλμα λίγο «μπροστά»
                arr[idx] = old - h
                e_minus = self.mse(X, y)                      # σφάλμα λίγο «πίσω»
                arr[idx] = old                                # επαναφορά
                g[idx] = (e_plus - e_minus) / (2 * h)         # κλίση
            grads.append(g)
        self.centers -= lr * grads[0]                         # βήμα προς τα «κάτω»
        self.sigmas -= lr * grads[1]
        self.sigmas = np.maximum(self.sigmas, 1e-3)           # το σ πρέπει να μείνει θετικό

    # ---------- Αποθήκευση / επαναφορά (για την καλύτερη epoch) ----------
    def get_params(self):
        """Αντίγραφο (όχι αναφορά!) των παραμέτρων — όπως το .clone() στη Java."""
        return self.centers.copy(), self.sigmas.copy(), self.consequents.copy()

    def set_params(self, params):
        c, s, q = params
        self.centers, self.sigmas, self.consequents = c.copy(), s.copy(), q.copy()


def load(name):
    """Φόρτωση ενός CSV από το βήμα 1 -> (X, y)."""
    df = pd.read_csv(Path(__file__).parent / "data" / f"{name}.csv")
    return df.drop(columns="class").to_numpy(), df["class"].to_numpy(dtype=float)


# ---------- Μικρό τεστ: τρέχει μόνο αν εκτελέσουμε απευθείας αυτό το αρχείο ----------
if __name__ == "__main__":
    X_tr, y_tr = load("train")
    X_ch, y_ch = load("check")

    model = ANFIS(X_tr, n_mf=2)
    print(f"Κανόνες: {model.n_rules},  premise παράμ.: {model.centers.size + model.sigmas.size},"
          f"  consequent παράμ.: {model.consequents.size}")
    print("Αρχικά κέντρα (γραμμή = είσοδος):\n", model.centers.round(2))

    # lr μεγάλο (5) γιατί η κλίση του MSE ως προς c, σ βγαίνει πολύ μικρή σε μέγεθος
    for epoch in range(1, 51):                                # 50 epochs υβριδικής μάθησης
        model.fit_consequents(X_tr, y_tr, lam=0.01)
        if epoch % 10 == 1 or epoch == 50:                    # τύπωσε κάθε 10 epochs
            print(f"epoch {epoch:>2}:  RMSE Training = {model.rmse(X_tr, y_tr):.4f}   "
                  f"RMSE Checking = {model.rmse(X_ch, y_ch):.4f}")
        model.gradient_step(X_tr, y_tr, lr=5.0)
    print("Τελικά κέντρα:\n", model.centers.round(2))
