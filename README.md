# Projetos de Machine Learning

Coleção de estudos práticos de machine learning organizados por assunto. Cada
pasta contém um exemplo independente e um README com os detalhes do projeto.

## Projetos

| Tema | Descrição | Técnicas |
| --- | --- | --- |
| [Classificação](./classificacao/) | Prevê se um projeto será finalizado com base no preço e nas horas esperadas. | SVM linear, divisão treino/teste e acurácia |
| [Regressão linear](./regressao-linear/) | Estima a geração de energia de uma usina e analisa a qualidade do modelo. | OLS, R², VIF e análise de resíduos |

## Como executar

Requer Python 3.10 ou mais recente.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python classificacao/main.py
python regressao-linear/main.py
```

> No Windows, ative o ambiente com `.venv\Scripts\activate`.

## Estrutura

```text
projetos_ml/
├── classificacao/
│   ├── README.md
│   └── main.py
├── regressao-linear/
│   ├── README.md
│   ├── main.py
│   └── usina.csv
├── .gitignore
├── README.md
└── requirements.txt
```
