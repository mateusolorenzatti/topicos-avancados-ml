"""
3 - Random Forest

Author: Mateus Orlandin Lorenzatti (https://github.com/mateusolorenzatti)
Source: https://www.kaggle.com/datasets/impapan/student-performance-data-set

Levantamento da condição da nota final dos alunos

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

    return pd.read_csv(file_path, sep=';')

def corr(base):
    import seaborn as sns
    from matplotlib import pyplot as plt
    correlations=base.corr()

    plt.figure(figsize=(12,8))
    plot = sns.heatmap(correlations, cmap="Greens",annot=True)
    fig = plot.get_figure()
    fig.savefig("out.png") 

def preparaBases(base):
    # Criar um campo categoria originado de G3 (média final), com notas boas e ruins
    base['g-final'] = pd.qcut(base['G3'], q=2, precision=0, labels=[0,1])
    # print(base.head())
    
    classe = base['g-final'].values
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

def naive_bayes(X_train,X_test,y_train, y_test):
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
    floresta = RandomForestClassifier(n_estimators=3, max_features=9, criterion='gini', random_state=1)
    floresta.fit(X_train, y_train)
    y_pred=floresta.predict(X_test)
    print("###Resultados Random Forest###")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test,y_pred))    


def main():
    base = abrirArquivoLocal('student-performance-por.csv')    

    X_train,X_test,y_train,y_test=preparaBases(base)
    # corr(base)

    naive_bayes(X_train,X_test,y_train,y_test)
    arvore_decisao(X_train,X_test,y_train,y_test)
    floresta_randomica(X_train,X_test,y_train, y_test)

if __name__ == '__main__':
    main()