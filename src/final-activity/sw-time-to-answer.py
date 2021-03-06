# -*- coding: utf-8 -*-
"""ADS (2022/1) - Tópicos Avançados [ Trabalho Final ].ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FE8jsIjp5B89k8-hP1HUnkgCQUrKngVe

**Aplicação de Machine Learning para predição de tempo de resposta no Stack Overflow**

Mateus Orlandin Lorenzatti e Rodrigo Lumbieri

# 1. Preparando os Dados
"""

import pandas as pd
import numpy as np

# Coletando o dataset
df = pd.read_csv('https://www.ics.uci.edu/~duboisc/stackoverflow/answers.csv')

# Renomeando os campos
df.rename(columns={'Unnamed: 0':'Ordem', 'qid':'question_id', 'i': 'questioner_id', 'qs': 'question_score', 'qt': 'question_time', 'tags': 'question_tags', 'qvc': 'number_views', 'qac':'number_answers', 'aid': 'answer_id', 'j': 'answerer_id', 'as': 'answer_score', 'at': 'answer_time'}, inplace=True )
del df['Ordem']

df.head()

# df.to_csv('stack_overflow_dataset.csv', index=False)

# df.info()

import time

# Transformando o tempo de epochs para datetime
df['question_datetime'] = pd.to_datetime(df['question_time'], unit = 's')
df['answer_datetime'] = pd.to_datetime(df['answer_time'], unit = 's')

# Calculando o tempo em minutos entre as datas
df['minutes_to_answer'] = (df['answer_datetime'] - df['question_datetime']).dt.total_seconds() / 60

# df.head()

# Extraindo o dia da semana do dia da pergunta
df['dow_question'] = df['question_datetime'].dt.dayofweek

# Monday =0, Tuesday=1, Wednesday=2,Thursday =3,  Friday=4 ,  Saturday =5, Sunday =6

# df.head()

# Divisão das tags em diferentes campos
df[['tag1', 'tag2', 'tag3', 'tag4', 'tag5']] = df['question_tags'].str.split(',',expand=True)

# df.head()

# Remover os valores que estiverem abaixo de zero

df = df[(df['minutes_to_answer'] > 0)]

df.to_csv('/content/sample_data/sw_qa_original_data.csv', index=False)

df_original = df

# Agrupar por questão, priorizando as respostas melhores
df = df.sort_values(by=['answer_score'], ascending=False)
df = df.drop_duplicates(['question_id'])

# df.head()

# Exclusão de campos originais, que não são necessários
df.drop(['question_id', 'questioner_id', 'question_score', 'question_time',	'question_tags', 'number_views', 'number_answers',	'answer_id',	'answerer_id',	'answer_score',	'answer_time', 'question_datetime', 'answer_datetime'], axis=1, inplace=True)

# Cria o encoding para cada Tag
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['tag1'] = labelencoder.fit_transform(df['tag1'])
df['tag2'] = labelencoder.fit_transform(df['tag2'])
df['tag3'] = labelencoder.fit_transform(df['tag3'])
df['tag4'] = labelencoder.fit_transform(df['tag4'])
df['tag5'] = labelencoder.fit_transform(df['tag5'])

# Cria o Standarization para o campo de tempo de resposta
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# df['minutes_to_answer'] = scaler.fit_transform(df[['minutes_to_answer']]).flatten()

# df['minutes_to_answer'] = labelencoder.fit_transform(df[['minutes_to_answer']])

from sklearn.preprocessing import KBinsDiscretizer

n_bins = 500

qt = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')

df['minutes_to_answer'] = qt.fit_transform(df['minutes_to_answer'].to_numpy().reshape(-1, 1))

df.to_csv('/content/sample_data/sw_qa_training_data.csv', index=False)

from sklearn.model_selection import train_test_split

preditores = df['minutes_to_answer']
classe = df.drop(['minutes_to_answer'], axis=1)

X_train,X_test,y_train,y_test = train_test_split(classe,preditores,test_size=0.25)
print('X train: ', X_train.shape, ', Y train: ', y_train.shape)

"""# 2. Analisando os Dados"""

df_original.head()

df.head()

df.shape

df_original.info()

df_original[(df_original['minutes_to_answer'] < 30000)]['minutes_to_answer'].hist(bins=100)

import matplotlib.pyplot as plt
import seaborn as sns 

# plt.figure(figsize=(10,10))
# sns.heatmap(df.corr())

x = df[(df['minutes_to_answer'] < 30000)]['minutes_to_answer']
sns.set_style('whitegrid')
sns.distplot(x)
plt.show()

df['minutes_to_answer']

"""# 3. Testando diferentes Algoritmos

Árvore de Decisão
"""

from sklearn.tree import DecisionTreeClassifier

arvore = DecisionTreeClassifier(criterion='entropy')

arvore.fit(X_train, y_train)

arvore.score(X_test, y_test)

previsoes = arvore.predict(X_test)
print(previsoes)
print(y_test)

from sklearn.metrics import mean_absolute_error, accuracy_score

mean_absolute_error(y_test, previsoes)

"""Random Forest"""

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200)

rf.fit(X_train, y_train)

rf.score(X_test, y_test)

previsoes = rf.predict(X_test)
print(previsoes)
print(y_test)

from sklearn.metrics import mean_absolute_error, accuracy_score

mean_absolute_error(y_test, previsoes)

"""Regressão Linear"""

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

previsoes = lr.predict(X_test)
print(previsoes)
print(y_test)

from sklearn.metrics import mean_absolute_error, accuracy_score

print(lr.score(X_test, y_test))
print(mean_absolute_error(y_test, previsoes))

"""Regressão Linear Múltipla"""

from sklearn.model_selection import train_test_split
X_sw_treinamento, X_sw_teste, y_sw_treinamento, y_sw_teste = train_test_split(X_train, y_train, test_size = 0.5, random_state = 1)
from sklearn.linear_model import LinearRegression
regressor_multiplo_sw = LinearRegression()
regressor_multiplo_sw.fit(X_sw_treinamento, y_sw_treinamento)

print(regressor_multiplo_sw.intercept_)
print(regressor_multiplo_sw.coef_)
print(regressor_multiplo_sw.score(X_sw_teste, y_sw_teste))

previsoes = regressor_multiplo_sw.predict(X_sw_teste)
print("Valores preditos", previsoes)
print("Valores Reais", y_sw_teste)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_sw_teste, previsoes)

"""SVM"""

from sklearn.svm import SVC
svm = SVC(kernel = 'poly', random_state = 1, C = 1.0)
    
svm.fit(X_train, y_train)

previsoes = svm.predict(X_test)
print(previsoes)
print(y_test)

from sklearn.metrics import mean_absolute_error, accuracy_score

mean_absolute_error(y_test, previsoes)

"""Rede Neural"""

import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(48, activation='sigmoid', input_shape=(6, )),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(96, activation='sigmoid'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='relu')
])

model.summary()

model.compile( optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss='mae',
              metrics=['mse'])

# Treinando o modelo
h = model.fit(X_train, y_train, epochs=5, batch_size=1, validation_data=(X_test, y_test))

from matplotlib import pyplot

pyplot.clf()
pyplot.plot(h.history['loss'], label='train')
pyplot.plot(h.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Verifica o score do modelo treinado
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(X_test)

import matplotlib.pyplot as plt
plt.clf()
plt.hist(predictions, label='Tempo Previsto')
plt.hist(y_test, label='Tempo Real')
plt.legend()
plt.show()

y_test