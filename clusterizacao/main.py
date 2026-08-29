# dessa vez, o trabalho eh com dados nao supervisionados (nao rotulados)

# usaremos a clusterizacao (agrupamento)

# objetivo: identificar padroes nos dados e segmentar os consumidores em suas bolhas de interesse semelhante e encontrar as caracteristicas dessas bolhas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("aulas/clusterizacao/dados_mkt.csv")

# pela analise das colunas, a unica que nao eh do tipo string, eh a de sexo (M, F ou NE) -> dados categoricos (sem informacoes numericas) -> usaremos o OneHotEncoder para substiuir eles por valores numericos
encoder = OneHotEncoder(categories=[['F', 'M', 'NE']], sparse_output=False)
encoded_sexo = encoder.fit_transform(df[['sexo']])
encoded_df = pd.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(['sexo'])) # novo data frame eh criado aqui
dados = pd.concat([df, encoded_df], axis=1).drop('sexo', axis=1) # axis=1 eh so pra confirmar que a operacao eh feita nas linhas
# print(dados.head()) # -> criou novas colunas 0 ou 1 com sexo_M, sexo_F e sexo_NE

from sklearn.cluster import KMeans # modelo que faz a clusterizacao
mod_kmeans = KMeans(n_clusters=2, random_state=45)
modelo = mod_kmeans.fit(dados)

# verificar o valor de inercia -> métrica utilizada para avaliar a qualidade dos clusters em um modelo K-Means -> retorna a soma das distâncias ao quadrado entre os pontos de um cluster e o seu centroide (ou seja, o quão próximos estão os pontos do centro do seu prórpio cluster)

# o valor de inércia é diretamente proporcional à distância dos pontos ao centro -> quanto menor, melhor, indicando um agrupamento de alta qualidade

# print(mod_kmeans.inertia_) # deu um valor BEM ALTO -> significa que os pontos estao bem distantes do centroide

from sklearn.metrics import silhouette_score
# print(silhouette_score(dados, mod_kmeans.predict(dados))) #  .predict() eh pra fazer a clusterizacao -> deu 0.37
# o siluette_scoure retorna a média das silhuetas que:
    # -> varia de -1 até 1, sendo: 
        # 1- objeto bem dentro do seu cluster e longe dos vizinhos
        # 0- objeto próximo da fronteira entre dois clusters
        # -1- objeto possivelmente em um clsuter errado

# criando uma funcao para avaliar os clusters:
def avaliacao(dados):
    inercia = []
    silhueta = []
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=45, n_init='auto')
        kmeans.fit(dados)
        inercia.append(kmeans.inertia_)
        silhueta.append(f"k={k} - {str(silhouette_score(dados, kmeans.predict(dados)))}")
    return silhueta, inercia
silhueta, inercia = avaliacao(dados)



import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples
#.             nº de clusters 
def grafico_silhueta(n_clusters, dados):
    # aplicar o KMeans ao conjunto de dados:
    kmeans = KMeans(n_clusters=n_clusters, random_state=45, n_init='auto')
    cluster_previsoes = kmeans.fit_predict(dados) # treina e prevê os dados 

    # calcular o silhouette score médio
    silhueta_media = silhouette_score(dados, cluster_previsoes)
    print(f"valor médio para {n_clusters} clusters: {silhueta_media:.3f}")

    # calcular a pontuacao da silhueta para cada amostra
    silhueta_amostra = silhouette_samples(dados, cluster_previsoes)

    # configuracao da figura para o grafico da silhueta
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)

    # limites do gráfico da silhueta
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(dados) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhueta_amostra = silhueta_amostra[cluster_previsoes == i]
        ith_cluster_silhueta_amostra.sort()

        tamanho_cluster_i = ith_cluster_silhueta_amostra.shape[0]
        y_upper = y_lower + tamanho_cluster_i

        cor = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhueta_amostra, facecolor=cor, edgecolor=cor, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * tamanho_cluster_i, str(i))
        y_lower = y_upper + 10

    # linha vertical para a média do Silhouette Score
    ax1.axvline(x=silhueta_media, color="red", linestyle="--")

    ax1.set_title(f"gráfico da silhueta para {n_clusters} clusters")
    ax1.set_xlabel("valores do coeficiente de silhueta")
    ax1.set_ylabel("rótulo do cluster")

    ax1.set_yticks([])
    ax1.set_xticks([i/10.0 for i in range(-1, 11)])

    plt.show()

# grafico_silhueta(2, dados)

# método do cotovelo serve para encontrar por meio de um gráfico a quantidade ideal de clusters para o modelo
def plot_cotovelo(inercia):
    plt.figure(figsize=(8,4))
    plt.plot(range(2, 21), inercia, 'bo-')
    plt.xlabel('número de clusters')
    plt.ylabel('inércia')
    plt.title('Método do cotovelo para a determinação de k')
    plt.show()
# plot_cotovelo(inercia) # -> as variacoes estao muito bruscas e a escala da inercia esta muito grande -> pois os dados nao estao normalizados

# normalizando os dados: -> escalando todos os dados de 0 a 1 
from sklearn.preprocessing import MinMaxScaler # isso transforma tudo em uma escala fixa de 0 a 1
scaler = MinMaxScaler()

dados_ecalados = scaler.fit_transform(dados)
dados_ecalados = pd.DataFrame(dados_ecalados, columns=dados.columns)
# print(dados_ecalados.describe())
# pq fiz isso? -> pq alguns dados numéricos estao em escalas diferentes e isso pode confundir a máquina (ex: idade ser de 18 ate 30 e numero de seguidores ir de 1000 ate 10000), fazendo com que interpretacoes erradas do impacto de uma variavel na inércia aonteçam

silhueta, inercia = avaliacao(dados_ecalados)
# print(silhueta) # -> melhor valor da silhueta deu quando tinha 3 clusters

# grafico_silhueta(3, dados_ecalados)

# plot_cotovelo(inercia) # -> a escala da inercia diminuiu drasticamente (o que eh excelente) e o cotovelo se formou em K = 3 (numero ideal de clusters eh 3)


# CRIANDO O MELHOR MODELO
modelo_kmeans = KMeans(n_clusters=3, random_state=67, n_init='auto')
modelo_kmeans.fit(dados_ecalados)



# ANALISAR AS CARACTERISTICAS DE CADA CLUSTER:

# 1- voltando os dados que estao na escala 0 a 1 ao normal
dados_analise = pd.DataFrame()

dados_analise[dados_ecalados.columns] = scaler.inverse_transform(dados_ecalados) # -> dados_analise[dados_ecalados.columns] para ter as mesmas colunas e scaler.inverse_transform reverte a normalizacao de dados_escalados

# print(dados_analise)

# 2- adicionando uma coluna 'cluster' com o respectivo cluster que o modelo KMeans direcionou aquela linha
dados_analise['cluster'] = modelo_kmeans.labels_ # .labels_ retorna o rotulo que o modelo definiu para cada dado

# 3- observar as caracteristicas mais revelantes agrupados por tipo de cluster
cluster_media = dados_analise.groupby('cluster').mean()

# print(cluster_media.T) # esse .T so inverte o que eh linha e o que eh coluna para melhor visualizacao em alguns casos

cluster_media = cluster_media.T

# print(cluster_media[0].sort_values(ascending=False))
# print(cluster_media[1].sort_values(ascending=False))
# print(cluster_media[2].sort_values(ascending=False))


# NOVAS_ENTRADAS
novo_df = pd.read_csv("aulas/clusterizacao/novas_entradas.csv")

# fazer o mesmo processo OneHotEncoder por causa da coluna sexo (obs: sem o fit.transform e apenas transform, ja que ja foi criado)
sexo_novas_entradas = encoder.transform(novo_df[["sexo"]])
novo_df_sexo = pd.DataFrame(sexo_novas_entradas, columns=encoder.get_feature_names_out(['sexo']))
novos_dados = pd.concat([novo_df, novo_df_sexo], axis=1).drop('sexo', axis=1)

# normalizar novos_dados
novos_dados_ecalados = scaler.transform(novos_dados)
novos_dados_ecalados = pd.DataFrame(novos_dados_ecalados, columns=novos_dados.columns)

clusters_novos = modelo_kmeans.predict(novos_dados_ecalados) # -> o .predict eh a funcao a ser utilizada, ja que os clusters ja existem -> retorna um vetor em que cada item representa o clsuter de cada linha
novo_df["cluster"] = clusters_novos
# print(novo_df.head())

print(dados_analise.shape, novo_df.shape)
print(dados_analise.columns)
print(novo_df.columns)


