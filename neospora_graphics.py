# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:25:47 2021

@author: asusu
"""
import pandas as pd # Analisis y manipulación de datos

import seaborn as sb # Creacion de graficos estadisticos informativos


# 1. Obtener datos y almacenarlos en un dataframe
neospora = pd.read_excel('Neospora.xlsx', 'Hoja1')

# 2. Creacion de graficos simples a partir de columnas de las tablas
sb.catplot(x='RAZA', data=neospora, kind="count")
sb.catplot(x='EDAD', data=neospora, kind="count")
sb.catplot(x='NEOSPORA', data=neospora, kind="count")
sb.catplot(x='TORO', data=neospora, kind="count")
sb.catplot(x='INSEMINACION', data=neospora, kind="count")
sb.catplot(x='ABORTO', data=neospora, kind="count")
sb.catplot(x='REPETICION', data=neospora, kind="count")
sb.catplot(x='NO_CARGA', data=neospora, kind="count")
sb.catplot(x='DISTOCIAS', data=neospora, kind="count")
sb.catplot(x='TERNEROS_DEBILES', data=neospora, kind="count")
sb.catplot(x='MUERTE_EMBRIONARIA', data=neospora, kind="count")


# 3. Agrupar por neospora (https://stackoverflow.com/questions/34615854/seaborn-countplot-with-normalized-y-axis-per-group)
# 3.1. Agrupar por raza/neospora
x,y = 'RAZA', 'NEOSPORA'

neospora.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index().pipe((sb.catplot,'data'), x=x,y='percent',hue=y,kind='bar')

df1 = neospora.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sb.catplot(x=x,y='percent',hue=y,kind='bar',data=df1)
g.ax.set_ylim(0,100)

for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)