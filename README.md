# Ciência de Dados — Previsão de Churn (Telco)

Projeto completo de **ciência de dados / machine learning** para prever a evasão
(*churn*) de clientes de uma empresa de telecomunicações.

- **Dataset:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) — 7.043 clientes, 21 atributos
- **Problema:** classificação binária (`Churn`: Yes/No), classe desbalanceada (~26,5%)
- **Stack:** Python 3.12, pandas, scikit-learn, seaborn/matplotlib, Jupyter

## Resultados

| Métrica (teste, 1.407 clientes) | Valor |
|---|---|
| ROC-AUC | **0,835** |
| PR-AUC | 0,641 |
| Recall (limiar 0,50) | 0,794 |
| Precisão (limiar 0,50) | 0,509 |
| F1 (limiar 0,50) | 0,620 |

- **Modelo final:** Random Forest (`n_estimators=800, max_depth=8, min_samples_leaf=3`)
  — ROC-AUC 0,849 na validação cruzada 5-fold, à frente de Gradient Boosting (0,848)
  e Regressão Logística (0,847).
- **Limiar ótimo para F1:** 0,60 (precisão 0,56 / recall 0,70) — o limiar é
  ajustável conforme o custo de abordar um cliente que não ia sair.
- **Principais fatores de risco:** contrato mês a mês, `tenure` baixo (primeiros
  12 meses), fibra óptica com gasto alto e ausência de TechSupport/OnlineSecurity.

## Estrutura do projeto

```
churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── .venv/                      # ambiente virtual
├── data/
│   ├── raw/Telco-Customer-Churn.csv
│   └── processed/              # train/test splits (gerado pelo nb 02)
├── notebooks/
│   ├── 01_eda.ipynb                       # análise exploratória
│   ├── 02_preprocessamento_feature_engineering.ipynb
│   ├── 03_modelagem.ipynb                 # CV + GridSearch
│   └── 04_avaliacao_interpretacao.ipynb   # teste, limiar, importância
├── models/                     # preprocessor.joblib, modelo_churn.joblib
└── reports/figures/            # 11 gráficos gerados
```

## Como executar

```bash
cd ~/PROJETOS/churn-prediction
source .venv/bin/activate
jupyter notebook notebooks/
```

Para recriar o ambiente do zero:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Execute os notebooks **na ordem** (01 → 04).

## Pipeline

1. **EDA** — qualidade dos dados, distribuição do alvo, churn por categoria,
   distribuições e correlações.
2. **Pré-processamento** — limpeza, 7 features novas (`faixa_tenure`,
   `qtd_servicos`, `gasto_por_mes`, flags…), split estratificado 80/20 e
   `ColumnTransformer` (imputação + scaler + one-hot), salvo em disco.
3. **Modelagem** — baseline Dummy, Regressão Logística, Random Forest e
   Gradient Boosting; `GridSearchCV` com ROC-AUC; modelo salvo em `models/`.
4. **Avaliação** — métricas no teste, curvas ROC/PR, análise de limiar,
   permutation importance e recomendações de negócio.

## Licença

Distribuído sob a licença [MIT](LICENSE).

Dataset: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (domínio público).
