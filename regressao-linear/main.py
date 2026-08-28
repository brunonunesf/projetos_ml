"""Analisa a geração de energia de uma usina com regressão linear."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


CAMINHO_DADOS = Path(__file__).with_name("usina.csv")
SEED = 670


def main() -> None:
    dados = pd.read_csv(CAMINHO_DADOS)
    y = dados["PE"]
    x = dados.drop(columns="PE")

    x_treino, _, y_treino, _ = train_test_split(
        x, y, test_size=0.25, random_state=SEED
    )
    x_treino = sm.add_constant(x_treino)

    modelo = sm.OLS(y_treino, x_treino).fit()
    previsoes = modelo.predict(x_treino)

    vif = pd.DataFrame(
        {
            "variavel": x_treino.columns,
            "vif": [
                variance_inflation_factor(x_treino.values, indice)
                for indice in range(len(x_treino.columns))
            ],
        }
    )

    print(f"R² do modelo: {modelo.rsquared:.2%}")
    print("\nFator de inflação da variância (VIF):")
    print(vif.to_string(index=False))

    px.scatter(
        x=previsoes,
        y=y_treino,
        title="Geração prevista × real",
        labels={"x": "Geração prevista", "y": "Geração real"},
    ).show()

    px.scatter(
        x=previsoes,
        y=modelo.resid,
        title="Resíduos do modelo",
        labels={"x": "Geração prevista", "y": "Resíduo"},
    ).show()


if __name__ == "__main__":
    main()
