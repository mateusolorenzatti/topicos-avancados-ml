"""
6 - Linear Regression

Author: Mateus Orlandin Lorenzatti (https://github.com/mateusolorenzatti)
Source: https://www.kaggle.com/datasets/impapan/student-performance-data-set

Levantamento da nota final dos alunos

"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import platform
import os.path as path

def abrirArquivoLocal(source_file):
    file_path = ''
    if ( 'Linux' in platform.system() ):
        file_path = path.abspath(path.join(__file__ ,"../"*3)) + '/data/' + source_file
    elif ( 'Windows' in platform.system()):
        file_path = path.abspath(path.join(__file__ ,"../"*3)) + '\\data\\' + source_file

    return pd.read_csv(file_path, sep=';')

def preparaBases(base):
    # print(base.head())

    classe = base.iloc[:,32].values

    # Elimina a classe da base
    preditores = base.iloc[:,0:30].values

    # print(preditores)
    # print(classe)

    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()

    labeled_p = [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]
    for i in labeled_p:
        preditores[:,i] = labelEncoder.fit_transform(preditores[:,i])

    # print(preditores)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(preditores,classe,test_size=0.25, random_state=42)
    return X_train,X_test,y_train,y_test

def regressao_linear_simples(X_train,X_test,y_train, y_test):
    from sklearn.linear_model import LinearRegression
    regressor_simples = LinearRegression()
    regressor_simples.fit(X_train, y_train)

    previsoes = regressor_simples.predict(X_train)
    
    print(regressor_simples.score(X_train, y_train))
    # print(previsoes)
    # print(y_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    # print(mean_absolute_error(y_test, previsoes))
    # print(mean_squared_error(y_test, previsoes))
    # print(np.sqrt(mean_squared_error(y_test, previsoes)))

def main():
    base = abrirArquivoLocal('student-performance-por.csv')

    X_train,X_test,y_train,y_test=preparaBases(base)

    regressao_linear_simples(X_train,X_test,y_train,y_test)

if __name__ == '__main__':
    main()