#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
# 
# - Base de Dados: https://drive.google.com/drive/folders/1o2lpxoi9heyQV1hIlsHXWSfDkBPtze-V?usp=sharing

# ### Passo a Passo de um Projeto de Ciência de Dados
# 
# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
# - TV, Jornal e Rádio estão em milhares de reais
# - Vendas estão em milhões

# #### Importar a Base de dados

# In[5]:


import pandas as pd


df = pd.read_csv('advertising.csv')
display(df)
print(df.info())


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# ## Instalar pacotes necessários

# In[ ]:


# !pip install matplotlib
# !pip install seaborn
# !pip install scikit-learn


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

# construir o gráigo
sns.heatmap(df.corr(), vmin = -1, cmap='coolwarm', annot=True)

# exibir o gráfico
plt.show()


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# #### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning
# 
# - Separando em dados de treino e dados de teste

# In[87]:


# separar os dados em x e y
y = df.iloc[:, 3] # = df['Vendas'] -> quem eu quero prever
x = df.iloc[:, 0:3] # quem eu vou usar para fazer a previsão

# separar os dados em base de treino e base de teste
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)


# #### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:
# 
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# In[88]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# criou a inteligência artificicial
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treinou a inteligência articifial
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)


# #### Teste da AI e Avaliação do Melhor Modelo
# 
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# In[97]:


# definir melhor modelo de previsão
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

from sklearn import metrics

print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))


# #### Visualização Gráfica das Previsões

# In[104]:


# a árvore de decisão é o melhor modelo
df_auxiliar = pd.DataFrame()
df_auxiliar['y_teste'] = y_teste
df_auxiliar['Previsão Regressão Linear'] = previsao_regressaolinear
df_auxiliar['Previsçao Árvore de Decisão'] = previsao_arvoredecisao

plt.figure(figsize=(15, 6))
sns.lineplot(data=df_auxiliar)
plt.show()


# #### Como fazer uma nova previsão?

# In[108]:


# importar a nova base de dados

novos = pd.read_csv('novos.csv')
display(novos)

# escolhendo o melhor modelo:

previsao = modelo_arvoredecisao.predict(novos)
print(previsao)

