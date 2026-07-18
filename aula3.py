import pandas as pd

url = "https://gist.githubusercontent.com/guilhermesilveira/12291c548acaf544596795709020e3db/raw/325bdef098bd9cbc2189215b7e32e22f437f29f3/projetos.csv"
dados = pd.read_csv(url)
#print(dados.head())
dados["finalizado"] = dados["nao_finalizado"].map({1 : 0, 0 : 1}) # o que eh 1 vira 0 e o que eh 0 vira 1
# para melhor compreensao, declarei uma nova coluna "finalizado" que vai ser o contrario da coluna nao_finalizado
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x="horas_esperadas", y="preco", data=dados, hue="finalizado")
plt.show()
