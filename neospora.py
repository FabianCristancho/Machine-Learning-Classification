#Importaciones necesarias
import  numpy as np
import  pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from subprocess import call

from PIL import Image, ImageDraw


neospora = pd.read_excel('Neospora.xlsx', 'Hoja1')
print(neospora.groupby('NEOSPORA').size())
sb.factorplot('RAZA',data=neospora,kind="count")
grp = neospora[['RAZA', 'NEOSPORA']].groupby(['RAZA'], as_index=True).agg(['mean', 'count'])
print(grp)

# Mapear datos del dataframe para categorizarlos
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

# Mapping de inseminación
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


drop_elements = ['EDAD', 'RAZA', 'TORO', 'INSEMINACION', 'ABORTO', 'REPETICION', 'NO_CARGA', 'DISTOCIAS', 'TERNEROS_DEBILES', 'MUERTE_EMBRIONARIA']
neospora_mapping = neospora.drop(drop_elements, axis=1)


grp = neospora_mapping[['cat_raza', 'NEOSPORA']].groupby(['cat_raza'], as_index=False).agg(['count'])


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

y_train = neospora_mapping['NEOSPORA']
x_train = neospora_mapping.drop(['NEOSPORA'], axis=1).values

decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=10,
                                            min_samples_leaf=2,
                                            max_depth=15,
                                            class_weight={True:1.2})

decision_tree.fit(x_train, y_train)

with open(r"tree2.dot", 'w') as f:
    f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 11,
                              impurity = True,
                              feature_names = list(neospora_mapping.drop(['NEOSPORA'], axis=1)),
                              class_names = ['No', 'N1 Neospora'],
                              rounded = True,
                              filled= True )
import pydot
(graph,) = pydot.graph_from_dot_file('tree2.dot')
graph.write_png('image2.png')
print('Terminado')
# Convertir el archivo .dot a png para poder visualizarlo
# check_call(['dot','-Tpng',r'tree1.dot','-o',r'tree1.png'])
# PImage("tree1.png")