"""
4 - Logistic Regression

Author: Mateus Orlandin Lorenzatti (https://github.com/mateusolorenzatti)
Source: https://www.kaggle.com/datasets/prasannakm/car-crash-dataset

Predição de sobrevivências em acidentes de trânsito

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import platform
import os.path as path

def abrirArquivoLocal(source_file):
    file_path = ''
    if ( 'Linux' in platform.system() ):
        file_path = path.abspath(path.join(__file__ ,"../"*3)) + '/data/' + source_file
    elif ( 'Windows' in platform.system()):
        file_path = path.abspath(path.join(__file__ ,"../"*3)) + '\\data\\' + source_file

    return pd.read_csv(file_path)

def corr(base):
    import seaborn as sns
    from matplotlib import pyplot as plt
    correlations=base.corr()

    plt.figure(figsize=(12,8))
    plot = sns.heatmap(correlations, cmap="Greens",annot=True)
    fig = plot.get_figure()
    fig.savefig("out.png") 

def preparaBases(base):
    # print(base)
    classe = base.iloc[:,2].values

    base.pop('dead')
    preditores = base.iloc[:,0:12].values
    # correlations=base.corr()

    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    
    labeled_p = [0,2,3,5,9,10]
    for i in labeled_p:
        preditores[:,i] = labelEncoder.fit_transform(preditores[:,i])

    classe=labelEncoder.fit_transform(classe)

    base['c#dead'] = classe

    corr(base)
    # print(base)
    # print(preditores)
    # print(classe)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(preditores,classe,test_size=0.25, random_state=42)
    return X_train,X_test,y_train,y_test

def naive_Bayes(X_train,X_test,y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    naive = GaussianNB()
    naive.fit(X_train, y_train)
    y_pred=naive.predict(X_test)
    print("###Resultados Naive Bayes###")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test,y_pred))

def arvore_decisao(X_train,X_test,y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    arvore = DecisionTreeClassifier(criterion='entropy')
    arvore.fit(X_train, y_train)
    y_pred=arvore.predict(X_test)
    print("###Resultados Árvore de Decisão###")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test,y_pred))

def floresta_randomica(X_train,X_test,y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    floresta = RandomForestClassifier(n_estimators=21, max_features=10, criterion='gini', random_state=1)
    floresta.fit(X_train, y_train)
    y_pred=floresta.predict(X_test)
    print("###Resultados Random Forest###")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test,y_pred))    

def regressao_logistica(X_train,X_test,y_train, y_test):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    regressaoL = LogisticRegression(random_state=1,max_iter=1000)
    regressaoL.fit(X_train, y_train)
    y_pred=regressaoL.predict(X_test)
    print("###Resultados Regressão Logística###")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test,y_pred))    

def main():
    base = abrirArquivoLocal('car-crash-train-test.csv')   

    X_train,X_test,y_train,y_test=preparaBases(base)

    naive_Bayes(X_train,X_test,y_train,y_test)
    arvore_decisao(X_train,X_test,y_train,y_test)
    floresta_randomica(X_train,X_test,y_train, y_test)
    regressao_logistica(X_train,X_test,y_train, y_test)

if __name__ == '__main__':
    main()