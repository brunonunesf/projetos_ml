import pandas as pd

url = "https://gist.githubusercontent.com/guilhermesilveira/12291c548acaf544596795709020e3db/raw/325bdef098bd9cbc2189215b7e32e22f437f29f3/projetos.csv"
dados = pd.read_csv(url)
#print(dados.head())
dados["finalizado"] = dados["nao_finalizado"].map({1 : 0, 0 : 1}) # o que eh 1 vira 0 e o que eh 0 vira 1
# para melhor compreensao, declarei uma nova coluna "finalizado" que vai ser o contrario da coluna nao_finalizado
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.scatterplot(x="horas_esperadas", y="preco", data=dados, hue="finalizado")
# # "hue" colore os pontos
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

x = dados[["horas_esperadas", "preco"]]
y = dados["finalizado"]

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state=SEED, test_size=0.25)
print(f"treinaremos com {len(treino_x)} e testaremos com {len(teste_x)}")

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"acurácia de {acuracia:.2f}%")