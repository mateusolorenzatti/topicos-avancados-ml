"""
6 - Linear Regression

Author: Mateus Orlandin Lorenzatti (https://github.com/mateusolorenzatti)
Source: https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction?select=train-data.csv

Predição de valor de automóveis usados

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

    return pd.read_csv(file_path)

def preparaBasesSimples(base):
    # print(base.head())

    classe = base.iloc[:,13:14].values
    # print(classe)

    # Campo Year
    preditores = base.iloc[:,3:4].values
    # print(preditores)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(preditores,classe,test_size=0.25, random_state=42)
    return X_train,X_test,y_train,y_test

def preparaBases(base):
    # print(base.head())
    
    classe = base.iloc[:,13:14].values
    # print(classe)

    preditores = base.iloc[:,2:12].values
    # print(preditores)

    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()

    labeled_p = [0,3,4,5]
    for i in labeled_p:
        preditores[:,i] = labelEncoder.fit_transform(preditores[:,i])

    preditores['Mileage'] = preditores['Mileage'].str.replace(' km/kg', '')
    print(preditores['Mileage'])

    # ToDo: Descobir como remover a unidade do campo

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(preditores,classe,test_size=0.25, random_state=42)
    return X_train,X_test,y_train,y_test

def regressao_linear(X_train,X_test,y_train, y_test):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    previsoes = regressor.predict(X_test)
    tipo = ''

    if X_train.shape[1] == 1: 
        tipo = 'Simples' 
    else: 
        tipo = 'Multipla'

    print(" ------- Regressão Linear", tipo)
    
    print("Valores preditos: ", previsoes[0:5,])
    print("Valores reais: ", y_test[0:5,])

    print("Score: ", regressor.score(X_test, y_test))

    from sklearn.metrics import mean_absolute_error
    print("MAE: ", mean_absolute_error(y_test, previsoes))
    print("")

def regressao_polinomial(X_train,X_test,y_train, y_test):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree = 2)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train_poly, y_train)

    previsoes = regressor.predict(X_test_poly)

    print(" ------- Regressão Polinomial")
    
    print("Valores preditos: ", previsoes[0:5,])
    print("Valores reais: ", y_test[0:5,])

    # print("Score: ", regressor.score(X_test_poly, y_test))
    print("Score: ", regressor.score(X_train_poly, y_train))

    from sklearn.metrics import mean_absolute_error
    print("MAE: ", mean_absolute_error(y_test, previsoes))
    print("")
    
def regressao_arvores_decisao(X_train,X_test,y_train, y_test):
    from sklearn.tree import DecisionTreeRegressor
    regressor_arvore = DecisionTreeRegressor()
    regressor_arvore.fit(X_train, y_train)    

    previsoes = regressor_arvore.predict(X_test)

    print(" ------- Regressão por Árvore de Decisão")
    
    print("Valores preditos: ", previsoes[0:5,])
    print("Valores reais: ", y_test[0:5,])

    print("Score: ", regressor_arvore.score(X_test, y_test))

    from sklearn.metrics import mean_absolute_error
    print("MAE: ", mean_absolute_error(y_test, previsoes))
    print("")

def regressao_random_forest(X_train,X_test,y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor
    regressor_random_forest = RandomForestRegressor(n_estimators=20)
    regressor_random_forest.fit(X_train, y_train)

    previsoes = regressor_random_forest.predict(X_test)

    print(" ------- Regressão por Random Forest")
    
    print("Valores preditos: ", previsoes[0:5,])
    print("Valores reais: ", y_test[0:5,])

    print("Score: ", regressor_random_forest.score(X_test, y_test))

    from sklearn.metrics import mean_absolute_error
    print("MAE: ", mean_absolute_error(y_test, previsoes))
    print("")

def regressao_svm(X_train,X_test,y_train, y_test):
    from sklearn.preprocessing import StandardScaler

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))
    
    from sklearn.svm import SVR
    regressor_svr = SVR(kernel='rbf')
    regressor_svr.fit(X_train_scaled, y_train_scaled.ravel())

    previsoes = regressor_svr.predict(X_test_scaled)

    y_test_inverse = scaler_y.inverse_transform(y_test_scaled)
    previsoes_inverse = scaler_y.inverse_transform(previsoes.reshape(-1, 1))

    print(" ------- Regressão por SVM")
    
    print("Valores preditos: ", previsoes_inverse[0:5,])
    print("Valores reais: ", y_test_inverse[0:5,])

    print("Score: ", regressor_svr.score(X_test_scaled, y_test_scaled))

    from sklearn.metrics import mean_absolute_error
    print("MAE: ", mean_absolute_error(y_test_inverse, previsoes_inverse))
    print("")

def main():
    base = abrirArquivoLocal('used-cars-price-prediction.csv')

    X_train,X_test,y_train,y_test=preparaBasesSimples(base)
    regressao_linear(X_train,X_test,y_train,y_test)

    X_train,X_test,y_train,y_test=preparaBases(base)
    # regressao_linear(X_train,X_test,y_train,y_test)
    # regressao_polinomial(X_train,X_test,y_train,y_test)
    # regressao_arvores_decisao(X_train,X_test,y_train,y_test)
    # regressao_random_forest(X_train,X_test,y_train,y_test)
    # regressao_svm(X_train,X_test,y_train,y_test)

if __name__ == '__main__':
    main()