# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:23:18 2022

@author: pablo
"""

reset -f

# Cargar Librerías Básicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
%matplotlib inline
sns.set_style('whitegrid')

# Change working directory
os.chdir(r'C:/Users/pablo/Desktop/EDEM/estadisticapython/code_and_data')
os.getcwd()

df = pd.read_csv ('insurance.csv',sep=',',decimal='.')

df.shape
df.head()
df.tail()

# QC OK --> Quality control OK --> He confirmado que está bien

###Data Analysis###
df.info()
df.isnull().sum()
df.describe().T

###Exploratory Data Analysis###

#ver columnas
df.columns
#ver ciertos valores de la edad
df['age'].value_counts()[:30]
#histograma
df['age'].plot(kind = 'hist')
#datos de la variable edad
age_des = df['age'].describe()
age_des.mean
#histograma pro
plt.figure(figsize=(10, 16))
ax = sns.displot(data = df, x = 'age', kde = True)

plt.axvline(39, linestyle = '--', color = 'green', label = 'mean Age')
plt.title('Age Distribution')
#
labels = df['age'].value_counts().keys().to_list()
values = df['age'].value_counts().to_list()

fig = go.Figure(go.Pie(labels=labels, 
                      values=values,
                      hole=0.5))
fig.show()
#variable sexo
df['sex'].value_counts()
#
fig = px.box(df['age'],
            title = "Age Distribution")
fig.show()

df['sex'].value_counts()
#distribución por sexo
sns.countplot(df['sex'])
plt.title("Gender of the peoples")

#Analysis of BMI

df['bmi'].plot(kind = 'hist')

#histograma pro
bmi_des = df['bmi'].describe()

plt.figure(figsize=(16,6))
sns.displot(df['bmi'], kde = True)
plt.axvline(bmi_des['mean'], linestyle = "--", color = "red")
plt.title('BMI distribution')

#Analysis of children
df['children'].plot(kind = "hist")

child_m = df['children'].describe()
child_m['mean']


#histograma max pro
print('Mean value of children: {}'.format(child_m['mean']))

plt.figure(figsize=(10, 6))
sns.distplot( df['children'])
plt.axvline(child_m['mean'], linestyle = "--", color = "red", )
plt.title("Distribution of Children")


#Analysis of smoker
df['smoker'].value_counts()

sns.countplot(df['smoker'])


sns.countplot(df['region'])

#Análisis Bivariante

#Tabla de contingencia
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr().T, annot=True)
