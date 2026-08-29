# Projetos de Machine Learning

estudos de machine learning organizados por assunto. 
cada pasta contém um exemplo independente e um README com os detalhes do projeto.

## Projetos

| Tema | Descrição | Técnicas |
| --- | --- | --- |
| [classificação](./classificacao/) | prevê se um projeto será finalizado com base no preço e nas horas esperadas. | SVM linear, divisão treino/teste e acurácia |
| [clusterização](./clusterizacao/) | segmenta consumidores em grupos com interesses e características semelhantes. | K-Means, normalização, método do cotovelo e silhueta |
| [regressão linear](./regressao-linear/) | estima a geração de energia de uma usina e analisa a qualidade do modelo. | OLS, R², VIF e análise de resíduos |

## Como executar

Requer Python 3.10 ou mais recente.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python classificacao/main.py
python clusterizacao/main.py
python regressao-linear/main.py
```

> No Windows, ative o ambiente com `.venv\Scripts\activate`.

## Estrutura

```text
projetos_ml/
├── classificacao/
│   ├── README.md
│   └── main.py
├── clusterizacao/
│   ├── dados_mkt.csv
│   ├── main.py
│   └── novas_entradas.csv
├── regressao-linear/
│   ├── README.md
│   ├── main.py
│   └── usina.csv
├── .gitignore
├── README.md
└── requirements.txt
```
