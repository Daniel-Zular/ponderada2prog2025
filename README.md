# Relatório 

## Objetivo
O objetivo da atividade foi otimizar um modelo de rede neural pré-treinado para detecção de fraudes em cartões de crédito.  
Foram aplicadas técnicas de ajuste de hiperparâmetros, como GridSearchCV e RandomizedSearchCV, visando melhorar métricas de desempenho: **Accuracy, Precision, Recall, F1-score e AUC-ROC**.  
Também foi realizada uma comparação entre o modelo original (baseline) e os modelos otimizados, para avaliar o impacto das modificações.

---

## Etapas Realizadas

### 1. Carregamento e análise dos dados
- O dataset foi carregado a partir de um link no Google Drive.  
- O conjunto de dados contém **284.807 transações** e **31 colunas**.  
- A variável alvo é **Class** (0 = transação legítima, 1 = fraude).  
- A distribuição das classes mostrou **desbalanceamento extremo**: apenas 0,17% das transações são fraudes.

### 2. Divisão em treino e teste
- As colunas de entrada foram todas, exceto `Class`.  
- Os dados foram divididos em **80% treino** e **20% teste**, de forma **estratificada** para manter a proporção de fraudes em ambos os conjuntos.

### 3. Pré-processamento
- As variáveis **Amount** e **Time** foram padronizadas com `StandardScaler`.  
- As demais variáveis (V1–V28) já estavam transformadas e foram mantidas.

### 4. Modelo Baseline
- Foi construído um **MLP (rede neural multicamadas)** com os seguintes hiperparâmetros padrão:  
  - 2 camadas ocultas  
  - 32 neurônios por camada  
  - Dropout de 0.2  
  - Ativação ReLU  
  - Learning rate de 0.001  
- Para lidar com o desbalanceamento, foram aplicados **pesos de classe balanceados**.  
- Foi usado **EarlyStopping** monitorando a métrica AUC.  
- O modelo baseline atingiu:  
  - **Accuracy:** 0.9738  
  - **Precision:** 0.0566  
  - **Recall:** 0.9082  
  - **F1-score:** 0.1066  
  - **AUC-ROC:** 0.9812  

### 5. GridSearchCV
- Foi definida uma grade de valores para taxa de aprendizado, número de neurônios, camadas ocultas, dropout, função de ativação, batch size e épocas.  
- O **GridSearch** testou todas as combinações usando validação cruzada estratificada com 3 folds.  
- Melhor configuração encontrada:  
  - Learning rate = 0.0003  
  - 16 neurônios, 1 camada, sem dropout, ativação ReLU, batch_size = 2048  
- Desempenho do melhor modelo no teste:  
  - **Accuracy:** 0.9767  
  - **Precision:** 0.0640  
  - **Recall:** 0.9184  
  - **F1-score:** 0.1196  
  - **AUC-ROC:** 0.9805  

### 6. RandomizedSearchCV
- Foram definidas distribuições para os mesmos hiperparâmetros.  
- O **RandomizedSearch** testou 12 combinações aleatórias.  
- Melhor configuração encontrada:  
  - Learning rate ≈ 0.0016  
  - 90 neurônios, 3 camadas, dropout ≈ 0.36, ativação ReLU, batch_size = 4096  
- Desempenho do melhor modelo no teste:  
  - **Accuracy:** 0.9816  
  - **Precision:** 0.0788  
  - **Recall:** 0.9082  
  - **F1-score:** 0.1451  
  - **AUC-ROC:** 0.9809  

---

## Comparação dos Resultados

| Modelo               | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|----------------------|----------|-----------|--------|----------|---------|
| Baseline             | 0.9738   | 0.0566    | 0.9082 | 0.1066   | 0.9812  |
| GridSearch (best)    | 0.9767   | 0.0640    | 0.9184 | 0.1196   | 0.9805  |
| RandomizedSearch (best) | 0.9816 | 0.0788    | 0.9082 | 0.1451   | 0.9809  |

---

## Conclusão
O modelo baseline já apresentou bom desempenho em AUC, mas tinha baixo valor de F1-score devido à baixa precisão.  
O **GridSearch** trouxe pequenas melhorias, principalmente em Recall.  
O **RandomizedSearch** apresentou o melhor equilíbrio entre métricas, com Recall alto, maior F1 e melhor precisão entre os três modelos.  

Assim, o ajuste de hiperparâmetros não alterou muito a AUC-ROC, mas contribuiu para melhorar o **equilíbrio entre Recall e Precision**, aumentando a capacidade do modelo em detectar fraudes de forma prática.


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
