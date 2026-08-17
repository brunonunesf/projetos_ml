import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


dados = pd.read_csv("aulas/regressao_linear/exercicio/ex1/usina.csv")
y = dados["PE"]
X = dados.drop(columns="PE") # tudo menos a coluna PE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=670) # criando as variaveis usadas para treinar e testar
# criando a variavel contanste para que evitar alguns erros caso alguma variavel X seja 0
X_train = sm.add_constant(X_train)
explicativas = X_train.columns
# criando o modelo
modelo = sm.OLS(y_train, X_train).fit() # treino do modelo
# print(modelo.summary()) # -> R-squared deu 93%


# testando a multicolinearidade
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_1 = pd.DataFrame()
vif_1["variavel"] = explicativas
vif_1["vif"] = [variance_inflation_factor(X_train[explicativas].values, i) for i in range(len(explicativas))]
print(vif_1) # -> a grande maioria deu um vif maior que 5, que é moderado e indica uma colinearidade boa, mas tem que se interpretar para evitar possiveis erros

#fazendo a analise de residuos:
import plotly.express as px
y_previsto_train = modelo.predict(X_train)
fig = px.scatter(x = y_previsto_train, y = y_train, title="Previsão x Real", labels={'x':'Preço previsto', 'y':'Preço real'})
fig.show() # se saiu incrivelmente bem

residuos = modelo.resid 
fig = px.scatter(x = y_previsto_train, y = residuos, title="Previsão x Real", labels={'x':'Preço previsto', 'y':'Redíduos'})
fig.show() # por nao existir um padrao na disposicao dos pontos no grafico, eh um caso de homocedasticidade
