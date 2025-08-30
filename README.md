

## IR ALÉM: Curva Precision–Recall

Como a base é altamente desbalanceada (apenas ~0,17% de fraudes), além da AUC-ROC também avaliei a **Curva Precision–Recall (PR)**.  
Essa curva mostra melhor o trade-off entre detectar fraudes (**Recall**) e evitar falsos positivos (**Precision**).  
A métrica associada é o **Average Precision (AP)**.

### Código
```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def pr_data(model):
    proba = model.predict_proba(X_test)[:, 1]
    p, r, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    return p, r, ap

models = {
    "Baseline": pipe_baseline,
    "GridSearch (best)": grid.best_estimator_,
    "RandomizedSearch (best)": rnd.best_estimator_,
}

plt.figure(figsize=(6,5))
for name, mdl in models.items():
    p, r, ap = pr_data(mdl)
    plt.plot(r, p, label=f"{name} (AP={ap:.4f})")

# linha de referência: prevalência da classe positiva
plt.hlines(y_test.mean(), 0, 1, colors="gray", linestyles="--", label="Taxa base")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid(True)
plt.show()
