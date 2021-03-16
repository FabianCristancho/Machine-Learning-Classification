# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:46:45 2021

@author: asusu
"""

import pandas as pd # Analisis y manipulacion de datos

import pydot # Creacion de archivo descriptivo en texto plano (DOT)

from sklearn import tree # Aprendizaje automatico
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import numpy as np

# 1. Obtener datos y almacenarlos en un dataframe
neospora = pd.read_excel('Neospora.xlsx', 'Hoja1')
print(neospora.groupby('NEOSPORA').size())


# 2. Creacion de filtros, graficos y analisis de datos en crudo
grp = neospora[['RAZA', 'NEOSPORA']].groupby(['RAZA'], as_index=True).agg(['mean', 'count'])
print(grp)


# 3. Mapear datos del dataframe para categorizarlos
# Mapping de la edad
neospora['cat_edad'] = neospora['EDAD'].map({'> 2 AÑOS': 1,
                                            '> 3 AÑOS': 2,
                                            '> 4 AÑO': 3}).astype(int)

# Mapping de la raza
neospora['cat_raza'] = neospora['RAZA'].map({'AYR': 1,
                                             'HOL': 2,
                                             'NOR': 3,
                                             'JER': 4}).astype(int)

# Mapping del toro
neospora['cat_toro'] = neospora['TORO'].map({False: 0,
                                              True: 1}).astype(int)

# Mapping de inseminaciÃ³n
neospora['cat_inseminacion'] = neospora['INSEMINACION'].map({False: 0,
                                              True: 1}).astype(int)

# Mapping de aborto
neospora['cat_aborto'] = neospora['ABORTO'].map({False: 0,
                                              True: 1}).astype(int)

# Mapping de repeticion
neospora['cat_repeticion'] = neospora['REPETICION'].map({False: 0,
                                              True: 1}).astype(int)

# Mapping de no carga
neospora['cat_no_carga'] = neospora['NO_CARGA'].map({False: 0,
                                              True: 1}).astype(int)

# Mapping de la DISTOCIAS
neospora['cat_distocias'] = neospora['DISTOCIAS'].map({False: 0,
                                             True:1}).astype(int)

# Mapping de la TERNEROS DEBILES
neospora['cat_terneros_debiles'] = neospora['TERNEROS_DEBILES'].map({False: 0,
                                             True:1}).astype(int)

# Mapping de la muerte embrionaria
neospora['cat_muerte_embrionaria'] = neospora['MUERTE_EMBRIONARIA'].map({False: 0,
                                             True:1}).astype(int)


# 4. Eliminacion de datos repetidos u obsoletos, y verificacion de los nuevos datos
drop_elements = ['EDAD', 'RAZA', 'TORO', 'INSEMINACION', 'ABORTO', 'REPETICION', 'NO_CARGA', 'DISTOCIAS', 'TERNEROS_DEBILES', 'MUERTE_EMBRIONARIA']
neospora_mapping = neospora.drop(drop_elements, axis=1)

grp = neospora_mapping[['cat_raza', 'NEOSPORA']].groupby(['cat_raza'], as_index=False).agg(['count'])


# 5. Entrenamiento de los datos para determinar el comportamiento del algoritmo
cv = KFold(n_splits=5)
accuracies = list()
max_attributes = len(list(neospora_mapping))
depth_range = range(1, max_attributes+1)

for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                              min_samples_split=10,
                                              min_samples_leaf=2,
                                              max_depth=depth,
                                              class_weight={True:1.2})
    
    for train_fold, valid_fold in cv.split(neospora_mapping):
        f_train = neospora_mapping.loc[train_fold]
        f_valid = neospora_mapping.loc[valid_fold]
        
        model = tree_model.fit(X = f_train.drop(['NEOSPORA'], axis=1), 
                                y = f_train['NEOSPORA'])
        valid_acc = model.score(X = f_valid.drop(['NEOSPORA'], axis=1),
                                y = f_valid['NEOSPORA'])
        fold_accuracy.append(valid_acc)
    
    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)

df = pd.DataFrame({'Max Depth': depth_range, "Average Accuracy": accuracies})
df = df[['Max Depth', 'Average Accuracy']]
print(df.to_string(index=False))
print('Entrenamiento finalizado')


# 6. Aplicacion del algoritmo de arbol de clasificacion por entropia
X = neospora_mapping.drop(['NEOSPORA'], axis=1).values
y = neospora_mapping['NEOSPORA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# y_train = neospora_mapping['NEOSPORA']
# x_train = neospora_mapping.drop(['NEOSPORA'], axis=1).values

decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=10,
                                            min_samples_leaf=2,
                                            max_depth=11,
                                            class_weight={True:1.2})
decision_tree.fit(X_train, y_train)
print('Arbol de decision implementado')


print('Result')
y_pred = decision_tree.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('Matriz de confusión: \ n ' , cm)

fig, ax = plt.subplots(figsize=(10,5))
ax.matshow(cm)
plt.title('Matriz de confusion', fontsize=20)
plt.ylabel('Etiqueta Predicha', fontsize=15)
plt.xlabel('Etiqueta Verdadera', fontsize=15)

for(i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')


# y_pred = decision_tree.predict(y_train)
# print(neospora['NEOSPORA'])
# y_test = neospora['NEOSPORA'].map({False: 0,
#                                               True: 1}).astype(int)

# print('-----------------------')
# print(y_test)

# new_neospora_predict = neospora.drop(drop_elements, axis=1)
# print(new_neospora_predict['NEOSPORA'])
# predict_test = new_neospora_predict.drop('NEOSPORA', 1).values


# print('Obteniendo y_predict')
# y_pred = decision_tree.predict(x_train)
# print(y_pred)


