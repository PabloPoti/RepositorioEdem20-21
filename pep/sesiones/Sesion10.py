# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 19:05:15 2021

@author: pablo
"""

reset -f

#load basiclibraries
import os
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype #For definition of custom categorical data types (ordinal if necesary)
import matplotlib.pyplot as plt
import seaborn as sns  # For hi level, Pandas oriented, graphics
import scipy.stats as stats  # For statistical inference 
import scipy.stats as stats


# Change working directory
os.chdir(r'C:\Users\pablo\Desktop\EDEM\estadisticapython\code_and_data')
os.getcwd()

#Reads data from CSV file and stores it in a dataframe called rentals_2011
# Pay atention to the specific format of your CSV data (; , or , .)
wbr = pd.read_csv ("WBR_11_12_denormalized_temp.csv", sep=';', decimal=',')
wbr.shape
wbr.head()
#QC OK

###MODELO DE REGRESIÓN SIMPLE 1 --> SOBRE LA TEMPERATURA###

#Primero un histograma SIEMPRE, este es lo más sencillo posible
wbr.cnt.hist()
wbr.temp_celsius.hist()
#Una bivariante
plt.scatter (wbr.temp_celsius,wbr.cnt)

#Nueva librería
from statsmodels.formula.api import ols

#OLS - Ordinary List Square
model1 = ols('cnt ~ temp_celsius', data=wbr).fit()
#       'vble target ~ vble predictora'

print(model1.summary2()) #Para ver que hay dentro de la regresión

#Nos interesa:
    #No. Observations:
    #R-squared:El % de predicción solo con la variable estimada (en este caso, el 40% de la variabilidad de las ventas es debido a la temperatura, el 60% aún no se a qué se debe)
    #Intercept: (el punto de partida --< temperatura igual a 0)valor predicho para la vble target cuando hay VALOR=0 del predictor --> SU P-VALUE NO SE INTERPRETA
    #temp_celsius(2º coef): Cuantas uds. se incrementa la vble dependiente por cada incremento en 1 unidad de esta vble predictora --> pendiente del modelo --> SU P-VALUE SÍ SE INTERPRETA --> Para ver si me atrevo a generalizar


###MODELO DE REGRESIÓN SIMPLE 2 --> SOBRE EL VIENTO###


#Primero un histograma SIEMPRE, este es lo más sencillo posible
wbr.cnt.hist()
wbr.windspeed_kh.hist()
#Una bivariante para confirmar que no hay nada raro
plt.scatter (wbr.windspeed_kh,wbr.cnt)

#Nueva librería
from statsmodels.formula.api import ols

#OLS - Ordinary List Square
model1b = ols('cnt ~ windspeed_kh', data=wbr).fit()
#       'vble target ~ vble predictora'

print(model1b.summary2()) #Para ver que hay dentro de la regresión

#Nos interesa(INTERPRETACIÓN):
    #No. Observations:
    #R-squared:En este caso, el 5,5% de la variabilidad de las ventas es debido a la velocidad del viento, el 94,5% es debido a otras variables.
    #Intercept: (el punto de partida --< temperatura igual a 0)valor predicho para la vble target cuando hay VALOR=0 del predictor --> SU P-VALUE NO SE INTERPRETA
    #temp_celsius(2º coef): Por cada km/h que aumenta el viento las ventas dismuyen en 87 uds. --> pendiente del modelo --> SU P-VALUE SÍ SE INTERPRETA --> Para ver si me atrevo a generalizar


###MODELO DE REGRESIÓN SIMPLE 3 --> SOBRE LA HUMEDAD###

#Primero un histograma SIEMPRE, este es lo más sencillo posible
wbr.cnt.hist()
wbr.hum.hist()
#Una bivariante para confirmar que no hay nada raro
plt.scatter (wbr.hum,wbr.cnt)

#Nueva librería
from statsmodels.formula.api import ols

#OLS - Ordinary List Square
model1c = ols('cnt ~ hum', data=wbr).fit()
#       'vble target ~ vble predictora'

print(model1c.summary2()) #Para ver que hay dentro de la regresión


###MODELO DE REGRESIÓN SIMPLE 4 --> SOBRE WORKINGDAY###

#Primero un histograma SIEMPRE, este es lo más sencillo posible
wbr.cnt.hist()
wbr.workingday.hist()
#Una bivariante para confirmar que no hay nada raro
plt.scatter (wbr.workingday,wbr.cnt)

#Nueva librería
from statsmodels.formula.api import ols

#OLS - Ordinary List Square
model1d = ols('cnt ~ workingday', data=wbr).fit()
#       'vble target ~ vble predictora'

print(model1d.summary2()) #Para ver que hay dentro de la regresión


###MODELO DE REGRESIÓN SIMPLE 4 --> SOBRE AÑO###

#Primero un histograma SIEMPRE, este es lo más sencillo posible
wbr.cnt.hist()
wbr.yr.hist()
#Una bivariante para confirmar que no hay nada raro
plt.scatter (wbr.yr,wbr.cnt)

#Nueva librería
from statsmodels.formula.api import ols

#OLS - Ordinary List Square
model1e = ols('cnt ~ yr', data=wbr).fit()
#       'vble target ~ vble predictora'

print(model1e.summary2()) #Para ver que hay dentro de la regresión



###MODELO DE REGRESIÓN MÚLTIPLE 1--> AJUSTAR UN MODELO TENIENDO EN CUENTA TEMPERATURA Y VIENTO###

model2 = ols('cnt ~ temp_celsius + windspeed_kh', data=wbr).fit()
print(model2.summary2())



###MODELO DE REGRESIÓN MÚLTIPLE 2--> AJUSTAR UN MODELO TENIENDO EN CUENTA TEMPERATURA, VIENTO Y HUMEDAD###

model3 = ols('cnt ~ temp_celsius + windspeed_kh + hum', data=wbr).fit()
print(model3.summary2())

###MODELO DE REGRESIÓN MÚLTIPLE 3--> AJUSTAR UN MODELO TENIENDO EN CUENTA TEMPERATURA, VIENTO Y HUMEDAD Y WORKINGDAY###

model4 = ols('cnt ~ temp_celsius + windspeed_kh + hum + workingday', data=wbr).fit()
print(model4.summary2()) #workingday NO ES NADA SIGNIFICATIVO; MEJOR QUITAR Y QUE NO GENERE RUIDO

###MODELO DE REGRESIÓN MÚLTIPLE 4--> AJUSTAR UN MODELO TENIENDO EN CUENTA TEMPERATURA, VIENTO Y HUMEDAD Y WORKINGDAY###

model5 = ols('cnt ~ temp_celsius + windspeed_kh + hum + workingday + yr', data=wbr).fit()
print(model5.summary2())

#¿Cómo interpretar variable nominal en modelo de regresión múltiple?:
    #Cuando todo vale 0 espero vender 2515 bicis, si aumento 1 año aumenta, ceteris paribus,en 2007 bicis vendidas.
    #Si no fuera dicotómica y hubiera más de dos opciones(p.e.: sunny, cloudy, rainy) habría que recodificar.
