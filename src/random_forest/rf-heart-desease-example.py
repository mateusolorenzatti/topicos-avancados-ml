# Feito por Prof. Dr. Ricardo A. Manfredini

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def preparaBases(base):
    base_heart = pd.read_csv(base)
    preditores = base_heart.iloc[:,1:18].values
    classe = base_heart.iloc[:,0].values
    correlations=base_heart.corr()
    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    preditores[:,1]= labelEncoder.fit_transform(preditores[:,1])
    preditores[:,2]= labelEncoder.fit_transform(preditores[:,2])
    preditores[:,3]= labelEncoder.fit_transform(preditores[:,3])
    preditores[:,6]= labelEncoder.fit_transform(preditores[:,6])
    preditores[:,7]= labelEncoder.fit_transform(preditores[:,7])
    preditores[:,8]= labelEncoder.fit_transform(preditores[:,8])
    preditores[:,9]= labelEncoder.fit_transform(preditores[:,9])
    preditores[:,10]= labelEncoder.fit_transform(preditores[:,10])
    preditores[:,11]= labelEncoder.fit_transform(preditores[:,11])
    preditores[:,12]= labelEncoder.fit_transform(preditores[:,12])
    preditores[:,14]= labelEncoder.fit_transform(preditores[:,14])
    preditores[:,15]= labelEncoder.fit_transform(preditores[:,15])
    preditores[:,16]= labelEncoder.fit_transform(preditores[:,16])
    classe=labelEncoder.fit_transform(classe)
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
    floresta = RandomForestClassifier(n_estimators=400, max_features= 13, criterion='gini', random_state=1)
    floresta.fit(X_train, y_train)
    y_pred=floresta.predict(X_test)
    print("###Resultados Random Forest###")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test,y_pred))    


def main():
    base_path='https://raw.githubusercontent.com/rmanfredini/data/main/heart_2020_cleaned.csv'
    X_train,X_test,y_train,y_test=preparaBases(base_path)
    naive_Bayes(X_train,X_test,y_train,y_test)
    arvore_decisao(X_train,X_test,y_train,y_test)
    floresta_randomica(X_train,X_test,y_train, y_test)

if __name__ == '__main__':
    main()