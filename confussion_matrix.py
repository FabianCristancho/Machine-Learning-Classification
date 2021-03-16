# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:31:23 2021

@author: asusu
"""

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

y_test = [1 , 0 , 0 , 1 , 0 , 0 , 1 , 0 , 0 , 1]
y_pred = [1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 0]

cm = confusion_matrix(y_test, y_pred)
print('Matriz de confusión: \ n ' , cm)

fig, ax = plt.subplots(figsize=(10,5))
ax.matshow(cm)
plt.title('Matriz de confusion', fontsize=20)
plt.ylabel('Etiquete Verdadera', fontsize=15)
plt.xlabel('Etiqueta Predicha', fontsize=15)

for(i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
