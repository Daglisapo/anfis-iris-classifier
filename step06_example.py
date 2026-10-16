"""
Ευφυή Συστήματα — 1η Εργασία (IRIS, ταξινομητής TSK/ANFIS)
ΒΗΜΑ 6: Παράδειγμα υπολογισμού — τα νούμερα κάθε στρώματος για ΕΝΑ δείγμα.

Σαν debugging με breakpoints: τυπώνουμε τι γίνεται σε κάθε στρώμα του ANFIS,
από τα 4 χαρακτηριστικά (cm) μέχρι την τελική κλάση. Για το paper και την παρουσίαση.

Τρέξιμο:  python step06_example.py [αριθμός δείγματος Testing, προεπιλογή 3]
"""
import sys

import numpy as np

from step02_anfis import load
from step04_evaluate import CLASSES, load_model, to_class

SHORT = ["ΜΣ", "ΠΣ", "ΜΠ", "ΠΠ"]            # μήκος/πλάτος σέπαλου, μήκος/πλάτος πετάλου
TERMS = ["μικρό", "μεγάλο"]
TOP = 4                                    # πόσους κανόνες να δείξουμε αναλυτικά


def rule_name(combo):
    """Ο κανόνας σε λέξεις, π.χ. (0,1,1,0) -> 'ΜΣ μικρό, ΠΣ μεγάλο, ΜΠ μεγάλο, ΠΠ μικρό'."""
    return ", ".join(f"{SHORT[i]} {TERMS[m]}" for i, m in enumerate(combo))


def fmt(v):
    """Πολύ μικροί αριθμοί σε επιστημονική μορφή (π.χ. 3.2e-05), αλλιώς 3 δεκαδικά."""
    return f"{v:.3f}" if v >= 1e-3 else f"{v:.1e}"


def layer1(model, x):
    """Στρώμα 1: οι βαθμοί συμμετοχής κάθε εισόδου."""
    mu = model.memberships(x[None, :])[0]                 # shape (4, 2)
    print("Στρώμα 1 — βαθμοί συμμετοχής μ:")
    for i in range(4):
        print(f"  {SHORT[i]} = {x[i]:.1f} cm:  μ_μικρό = {fmt(mu[i, 0])}   μ_μεγάλο = {fmt(mu[i, 1])}")


def layers2to5(model, x):
    """Στρώματα 2–5: οι πιο ισχυροί κανόνες, η γνώμη τους και η τελική έξοδος."""
    wn = model.normalized_strengths(x[None, :])[0]        # w̄ κάθε κανόνα, shape (16,)
    f = model.consequents @ np.append(x, 1.0)             # f κάθε κανόνα = p·x + r
    order = np.argsort(wn)[::-1]                          # από τον ισχυρότερο στον ασθενέστερο
    print(f"\nΣτρώματα 2–5 — οι {TOP} ισχυρότεροι από τους {model.n_rules} κανόνες:")
    for k in order[:TOP]:
        print(f"  w̄ = {wn[k]:.3f}   f = {f[k]:+7.3f}   w̄·f = {wn[k] * f[k]:+7.3f}"
              f"   [{rule_name(model.rules[k])}]")
    rest = order[TOP:]
    print(f"  υπόλοιποι {len(rest)} κανόνες: Σw̄ = {wn[rest].sum():.3f}, "
          f"Σw̄·f = {(wn[rest] * f[rest]).sum():+.3f}")
    print(f"  έλεγχος: Σw̄ όλων = {wn.sum():.3f}")
    return (wn * f).sum()


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    X_tr, _ = load("train")
    X_te, y_te = load("test")
    model = load_model(16, X_tr)
    x = X_te[idx]
    print(f"Δείγμα Testing #{idx}: x = {x}, πραγματική κλάση = {CLASSES[int(y_te[idx]) - 1]}\n")
    layer1(model, x)
    y = layers2to5(model, x)
    c = to_class(np.array([y]))[0]
    print(f"\nΈξοδος y = Σ w̄·f = {y:.3f}  ->  round/clip -> κλάση {c} ({CLASSES[c - 1]})")
