#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Vendas

# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa investe: TV, Jornal e Rádio
# - Base de Dados: https://drive.google.com/drive/folders/1o2lpxoi9heyQV1hIlsHXWSfDkBPtze-V?usp=sharing
# - TV, Jornal e Rádio estão em milhares de reais
# - Vendas estão em milhões

# ### Passo a Passo de um Projeto de Ciência de Dados:

# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# #### Importar a Base de dados
import pandas as pd

df = pd.read_csv('advertising.csv')
display(df)
print(df.info())

# #### Análise Exploratória

# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# ## Instalar pacotes necessários

# !pip install matplotlib
# !pip install seaborn
# !pip install scikit-learn


# #### Análise Exploratória

# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

import seaborn as sns
import matplotlib.pyplot as plt

# construir o gráfico
sns.heatmap(df.corr(), vmin = -1, cmap='coolwarm', annot=True)

# exibir o gráfico
plt.show()

# #### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning

# - Separando em dados de treino e dados de teste
# separar os dados em x e y
y = df.iloc[:, 3] # = df['Vendas'] -> quem eu quero prever
x = df.iloc[:, 0:3] # quem eu vou usar para fazer a previsão

# separar os dados em base de treino e base de teste
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

# #### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:

# - Regressão Linear
# - RandomForest (Árvore de Decisão)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# criar a inteligência artificicial
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treinar a inteligência articifial
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# #### Teste da AI e Avaliação do Melhor Modelo

# definir melhor modelo de previsão
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste) 

from sklearn import metrics

print(metrics.r2_score(y_teste, previsao_regressaolinear)) # -> resultado: 0.9185502000078154
print(metrics.r2_score(y_teste, previsao_arvoredecisao)) # -> resultado: 0.972110946487927

# #### Visualização Gráfica das Previsões

# a árvore de decisão é o melhor modelo (0.972 > 0.918)
df_auxiliar = pd.DataFrame()
df_auxiliar['y_teste'] = y_teste
df_auxiliar['Previsão Regressão Linear'] = previsao_regressaolinear
df_auxiliar['Previsçao Árvore de Decisão'] = previsao_arvoredecisao

plt.figure(figsize=(15, 6))
sns.lineplot(data=df_auxiliar)
plt.show()

# #### Como fazer uma nova previsão?

# importar a nova base de dados

novos = pd.read_csv('novos.csv')
display(novos)

# escolhendo o melhor modelo:

previsao = modelo_arvoredecisao.predict(novos)
print(previsao)
