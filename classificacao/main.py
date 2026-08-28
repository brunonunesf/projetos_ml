"""Classifica projetos como finalizados ou não usando um SVM linear."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


URL_DADOS = (
    "https://gist.githubusercontent.com/guilhermesilveira/"
    "12291c548acaf544596795709020e3db/raw/"
    "325bdef098bd9cbc2189215b7e32e22f437f29f3/projetos.csv"
)
SEED = 20


def main() -> None:
    dados = pd.read_csv(URL_DADOS)
    dados["finalizado"] = dados["nao_finalizado"].map({1: 0, 0: 1})

    x = dados[["horas_esperadas", "preco"]]
    y = dados["finalizado"]

    treino_x, teste_x, treino_y, teste_y = train_test_split(
        x, y, random_state=SEED, test_size=0.25, stratify=y
    )

    modelo = LinearSVC(random_state=SEED)
    modelo.fit(treino_x, treino_y)
    previsoes = modelo.predict(teste_x)
    acuracia = accuracy_score(teste_y, previsoes) * 100

    print(f"Treino: {len(treino_x)} registros | Teste: {len(teste_x)} registros")
    print(f"Acurácia: {acuracia:.2f}%")


if __name__ == "__main__":
    main()
