#1. Carregue o ficheiro .csv para um DataFrame, e de seguida crie um novo
#DataFrame com apenas a informação da Região “Asia”. Grave este novo
#DataFrame num novo ficheiro .csv.

import pandas as pd

# carrega o arquivo CSV
df = pd.read_csv('C:/Users/André Pereira/Desktop/Universidade/3º Ano/2º Semestre/Introdução à Ciência dos Dados/TRABALHO EXPERIMENTAL 1/TRABALHO EXPERIMENTAL 1 - GRUPO 3/Life-Expectancy-Data-Updated.csv')

# cria um novo DataFrame apenas com a informação da Região "Asia"
df_asia = df[df['Region'] == 'Asia']

# guarda o novo DataFrame num novo arquivo CSV
df_asia.to_csv('C:/Users/André Pereira/Desktop/Universidade/3º Ano/2º Semestre/Introdução à Ciência dos Dados/TRABALHO EXPERIMENTAL 1/TRABALHO EXPERIMENTAL 1 - GRUPO 3/Life-Expectancy-Data-Asia.csv', index=False)


# 2. A partir do novo DataFrame, faça um gráfico que lhe permita visualizar
# convenientemente a evolução das mortes de crianças menores de cinco anos por
# 1000 habitantes (“Under_five_deaths”) nos países China, India, Japan e Thailand.

import matplotlib.pyplot as plt

# cria um DataFrame com os países desejados

paises = ['China', 'India', 'Japan', 'Thailand']
df_paises = df_asia[df_asia['Country'].isin(paises)] 

#Filtra o DataFrame df_asia para manter apenas as linhas dos
#países presentes na lista paises e armazena o resultado em df_paises e usa a função isin (bolean)

# cria a coluna com a população em milhões
#['Population_mln'] = df_paises['Population'] / 1_000_000  # nova linha a ser adicionada

# cria o gráfico
fig, ax = plt.subplots(figsize=(10, 6)) #largura e altura
for pais in paises:
    df_pais = df_paises[df_paises['Country'] == pais]
    ax.plot(df_pais['Year'], df_pais['Under_five_deaths'], label=pais)
ax.legend()
ax.set_xlabel('Ano')
ax.set_ylabel('Mortes de crianças menores de cinco anos por 1000 habitantes')
ax.set_title('Evolução das mortes de crianças menores de cinco anos nos países selecionados')
plt.show()


# 3. Usando a biblioteca Matplotlib, crie um gráfico circular (‘pie chart’) que represente a
# média nos anos 2000 a 2015 da população total em milhões (“Population_mln”),
# nos países Afghanistan, Indonesia, Philippines e Vietnam. Coloque as legendas
# adequadas.

# cria um DataFrame com os países desejados e o período desejado
paises = ['Afghanistan', 'Indonesia', 'Philippines', 'Vietnam']
df_paises_periodo = df_asia[(df_asia['Country'].isin(paises)) & (df_asia['Year'] >= 2000) & (df_asia['Year'] <= 2015)]

# cria o gráfico
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(df_paises_periodo.groupby('Country')['Population_mln'].mean(), labels=paises, autopct='%1.1f%%')
ax.set_title('Média da população total nos países selecionados entre 2000 e 2015')
plt.show()


# 4. Crie uma função que, dado o nome do país da Região “Asia”, apresente o ano em
# que a esperança média de vida (“Life_expectancy”) foi maior, bem como o
# respetivo valor.

# definição da função
def maior_esperanca_vida(pais):
    df_pais = df_asia[df_asia['Country'] == pais]
    ano = df_pais.loc[df_pais['Life_expectancy'].idxmax()]['Year']
    # retorna o índice do valor máximo em uma série ou coluna de um
    #DataFrame.Em seguida, usamos o método loc[] para localizar 
    #a linha correspondente ao índice retornado pelo idxmax()
    esperanca_vida = df_pais['Life_expectancy'].max()
    print(f"A maior esperança média de vida no {pais} foi em {ano}: {esperanca_vida} anos.")

# exemplo da função com a India
maior_esperanca_vida('India')


# 5. Usando a biblioteca Seaborn, crie um gráfico de dispersão (‘scatter plot’) que
# permita visualizar o relacionamento entre os incidentes de HIV por 1000 habitantes
# dos 15 a 49 anos (“Incidents_HIV”) e a esperança média de vida
# (“Life_expectancy”). Apresente também no gráfico uma regressão linear que
# relacione as duas variáveis. Explique convenientemente o seu significado. Pode
# fazer análise para um país, uma região ou o globo.


import seaborn as sns

# seleciona os países e as variáveis de interesse
paises = ['Afghanistan', 'Indonesia', 'Philippines', 'Vietnam']
variaveis = ['Country','Incidents_HIV', 'Life_expectancy']

#df_paises = df_paises.assign(Country=paises)
#df_paises = df_paises[(df_paises['Age'] >= 15) & (df_paises['Age'] <= 49)]

# cria um novo DataFrame com as informações dos países selecionados
df_paises = df_asia[df_asia['Country'].isin(paises)][variaveis]

# faz o gráfico de dispersão com a regressão linear
sns.lmplot(x='Life_expectancy', y='Incidents_HIV', data=df_paises, hue='Country', fit_reg=True)

# define os rótulos dos eixos
plt.xlabel('Esperança média de vida')
plt.ylabel('Incidentes de HIV por 1000 habitantes dos 15 a 49 anos')

# define o título do gráfico
plt.title('Relação entre a esperança média de vida e os incidentes de HIV na Ásia')

# mostra o gráfico
plt.show()




# # 6. Explore técnicas de Machine Learning que lhe permita fazer uma previsão da
# # esperança média de vida (“Life_expectancy”) no futuro. Poderá usar quaisquer
# # colunas do Dataset que ache conveniente. Documente bem as técnicas utilizadas
# # e decisões tomadas

#verifica quantos campos a null existem
pd.isnull(df_asia).sum()

#Elimina os campos a null da coluna "Alcohol_consumption"
life_fixd = df_asia[df_asia['Alcohol_consumption'].notna()] #método do pandas utilizado para verificar se os valores de uma coluna ou DataFrame não são nulos

#Elimina os campos a null da coluna"Adult_mortality"
life_fixd = life_fixd[life_fixd['Adult_mortality'].notna()]

#Seleciona apenas os dados relativos a Philippines, por exemplo
life_fixd = life_fixd.loc[life_fixd.Country == "Philippines"]

#y é o prediction target e o que pretendemos prever é o "Life_expectancy"
y = life_fixd.Life_expectancy

#O modelo faz previsões com base no ano e nos valores da coluna 'Adult_mortality'
life_features = ['Year', 'Adult_mortality']

#Dados dos campos que alteram a previsão
X = life_fixd[life_features]

from sklearn.tree import DecisionTreeRegressor

#define model. Especifica um número para random_state para garantir os mesmos resultados em cada execução.
life_model = DecisionTreeRegressor(random_state=1)

#fit model
life_model.fit(X, y)

print("Making predictions dor the following 5 years: ")
print(X.head())
print("The predictions are: ")
#faz prefisão com base nos parametross de entrada (features)
print(life_model.predict(X.head()))

#6.2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Seleciona as colunas que serão usadas como variáveis independentes e a coluna alvo
df_asia = df_asia.dropna()
X = df_asia[['Year', 'Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Alcohol_consumption', 'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV', 'GDP_per_capita', 'Population_mln', 'Thinness_ten_nineteen_years', 'Thinness_five_nine_years', 'Schooling']]
y = df_asia['Life_expectancy']

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Cria e treina o modelo de regressão linear múltipla
regression = LinearRegression()
regression.fit(X_train, y_train)

# Faz previsões usando o conjunto de teste
y_pred = regression.predict(X_test)

# Avalia o desempenho do modelo usando a métrica de erro quadrático médio e o coeficiente de determinação (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")  #comparado com o observado e o expectávél
print(f"R2: {r2}")    #métrica que indica o quão bem o modelo de regressão linear se ajusta aos dados observados S




















# # Para fazer a previsão da esperança média de vida, utilizarei técnicas de regressão, 
# # que são um tipo de algoritmo de Machine Learning que permite prever um valor numérico
# # com base em um conjunto de variáveis de entrada. Vou utilizar o 
# # dataset "Life Expectancy (WHO)" disponibilizado pelo Kaggle, 
# # que contém informações sobre a esperança média de vida em diversos países
# # e fatores que podem influenciá-la.
# # Antes de começar a análise, é importante fazer uma limpeza nos dados e 
# # remover valores faltantes. Além disso, vou transformar as variáveis
# # categóricas em variáveis numéricas, utilizando a técnica de "one-hot encoding".
# # Começarei importando as bibliotecas necessárias e carregando os dados:
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import OneHotEncoder

# data = pd.read_csv('C:/Users/André Pereira/Desktop/TRABALHO EXPERIMENTAL 1 - CIENCIA DOS DADOS/TRABALHO EXPERIMENTAL 1 - GRUPO 3/Life-Expectancy-Data-Updated.csv')

# # Verifica se há valores em falta e remove as colunas que não serão utilizadas:
# missing_values = data.isnull().sum()
# print("Valores faltantes:\n", missing_values)
# data.drop(['Country', 'Year', 'Alcohol_consumption', 'Thinness_ten_nineteen_years', 'Hepatitis_B', 'Measles', 'Under_five_deaths', 'Diphtheria', 'GDP_per_capita', 'Population_mln', 'Thinness_five_nine_years'], axis=1, inplace=True)
# data.dropna(inplace=True)

# # Converte a variável "Status" em variáveis numéricas utilizando "one-hot encoding":
# encoder = OneHotEncoder()
# status_encoded = encoder.fit_transform(data[['Economy_status_Developed', 'Economy_status_Developing']])
# status_encoded_df = pd.DataFrame(status_encoded.toarray(), columns=encoder.get_feature_names(['Economy_status']))
# data.drop(['Economy_status_Developed', 'Economy_status_Developing'], axis=1, inplace=True)
# data = pd.concat([data, status_encoded_df], axis=1)

# # Separa os dados em conjuntos de treinamento e teste:
# X = data.drop(['Life_expectancy'], axis=1)
# y = data['Life_expectancy']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Treina um modelo de regressão linear simples:
# reg = LinearRegression().fit(X_train, y_train)

# # Avalia o modelo com o conjunto de teste:
# y_pred = reg.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Imprime as métricas de avaliação:
# print('Mean squared error (MSE): %.2f years^2' % mse)
# print('Coeficiente de Determinação (R²): %.2f' % r2)

    

      
 