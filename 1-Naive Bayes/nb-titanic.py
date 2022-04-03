"""
1 - Naive Bayes 

Source: https://www.kaggle.com/datasets/brendan45774/test-file?resource=download

"""
import pandas as pd
import numpy as np

base = pd.read_csv("C:\\Users\\5220203\Documents\\topicos-avancados-ml\\Data\\titanic-tested.csv")
# print(base.head())

base['Age'].fillna((base['Age'].mean()), inplace=True)

# Colunas Pclass, Sex e Age
previsores = base.iloc[:, [2,4,5]].values

# Coluna Survived
classe = base.iloc[:,1].values 

# print(previsores)
# print(classe)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0]) #Pclass
previsores[:,1] = labelencoder.fit_transform(previsores[:,1]) #Sex

# print(previsores)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

resultado = classificador.predict([[1,1,45], [2, 0, 76.5]])
print('-'*50)
print('Results -> ',resultado)
print('-'*50)
print('Classes -> ', classificador.classes_)
print('Classes Count -> ',classificador.class_count_)
print('Prob. -> ',classificador.class_prior_)
