"""
2 - Decision Tree Classifier 

Source: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease?resource=download

"""

import pandas as pd
import matplotlib.pyplot as plt
import platform
import os.path as path

# Abre o arquivo local em ../../data
source_file = 'heart-desease.csv'

file_path = ''
if ( 'Linux' in platform.system() ):
    file_path = path.abspath(path.join(__file__ ,"../"*3)) + '/data/' + source_file
elif ( 'Windows' in platform.system()):
    file_path = path.abspath(path.join(__file__ ,"../"*3)) + '\\data\\' + source_file

base = pd.read_csv(file_path)

# print(base.head())
# print(base.corr())

# --------------------------------------------------
# Separa os previsores e classe
previsores = base.iloc[:,1:].values
classe = base.iloc[:,0].values

# print(previsores)
# print(classe)

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

# --------------------------------------------------
# Todos os campos que vão sofrer transformação do label enconder
classe = labelencoder.fit_transform(classe)

labeled_previsores = [1,2,3,6,7,8,9,10,11,12,14,15,16]
for i in labeled_previsores:
    previsores[:,i] = labelencoder.fit_transform(previsores[:,i])

# print(previsores)
# print(classe)

# --------------------------------------------------
# Prepara as bases de preditores de classes para treino e teste
from sklearn.model_selection import train_test_split
test_size = 0.25
X_train,X_test,y_train,y_test=train_test_split(previsores,classe,test_size=test_size)

# --------------------------------------------------
# Instancia o objeto da árvore de decisão
from sklearn.tree import DecisionTreeClassifier
arvore_doenca_cardiaca = DecisionTreeClassifier(criterion='entropy')

# --------------------------------------------------
# O treina e testa a árvore
arvore_doenca_cardiaca.fit(X_train, y_train)
y_pred_hd = arvore_doenca_cardiaca.predict(X_test)

# --------------------------------------------------
# Exibe os resultados finais
from sklearn.metrics import confusion_matrix, accuracy_score
print('-'*50)
print('Accuracy Score -> ', accuracy_score(y_test, y_pred_hd))
print('Test Size -> ', test_size*100, '%')
print('-'*50)
print('Confusion Matrix -> \n ', confusion_matrix(y_test,y_pred_hd))
print('-'*50)

# print(arvore_doenca_cardiaca.feature_importances_)
# print(arvore_doenca_cardiaca.classes_)