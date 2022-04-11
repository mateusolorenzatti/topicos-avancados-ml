import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('https://github.com/rmanfredini/data/raw/main/risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
base                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

                 
from sklearn.tree import DecisionTreeClassifier
# Instancia o objeto da árvore de decisãp
arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')
#O método fit treina a árvore de decisap
arvore_risco_credito.fit(previsores, classe)


print(arvore_risco_credito.feature_importances_)
print(arvore_risco_credito.classes_)

from sklearn import tree
from sklearn.metrics import confusion_matrix
listaPrevisores = ['história', 'dívida', 'garantias', 'renda']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(arvore_risco_credito, feature_names=listaPrevisores, class_names = arvore_risco_credito.classes_, filled=True);

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15                

previsoes = arvore_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
print(previsoes)
print(confusion_matrix(previsoes, previsoes))