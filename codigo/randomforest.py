####################################################
#   Trabajo final: Communities And Crimes
####################################################
#   Autores:
#      - Alejandro Borrego Megías
#      - Blanca Cano Camarero
#   Fecha: Principios junio 2021
####################################################

from main import *

#############################
#######  BIBLIOTECAS  #######
#############################
# Biblioteca lectura de datos
# ==========================
import pandas as pd

# matemáticas
# ==========================
import numpy as np

# Modelos a usar   
# =========================================
from sklearn.ensemble import RandomForestRegressor

# Validación cruzada
# ==========================
from sklearn.model_selection import cross_val_score


# metricas
# ==========================
from sklearn.metrics import r2_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Otros
# ==========================
from operator import itemgetter #ordenar lista
import time


#redundante
np.random.seed(1)
######### CONSTANTES #########  
NOMBRE_FICHERO_REGRESION = './datos/train.csv'

################ funciones auxiliares

### Validación
def Evaluacion_validacion( modelo, x, y, x_val, y_val, nombre_modelo):
    '''
    Función para calcular error en entrenamiento y validación  
    INPUT:
    - modelo: Modelo con el que calcular los errores
    - X datos entrenamiento. 
    - Y etiquetas de los datos de entrenamiento
    - x_val, y_val conjunto de entrenamiento y etiquetas de validación
    '''

    ##########################
    
    print('\n','-'*60)
    print (f' Evaluando {nombre_modelo}')
    print('-'*60)
    # Error cuadrático medio
    # predecimos test acorde al modelo
    modelo.fit(x, y)
    prediccion = modelo.predict(x)
    prediccion_val = modelo.predict(x_val)

    Eval=r2_score(y_val, prediccion_val)
    Ein=r2_score(y, prediccion)
    print("E_in en entrenamiento: ",Ein)
    print("E_val en validación: ",Eval)
  
def Evaluacion_test_randomforest( modelo, x, y, x_test, y_test, nombre_modelo):
    '''
    Función para calcular error en entrenamiento y test  
    INPUT:
    - modelo: Modelo con el que calcular los errores
    - X datos entrenamiento. 
    - Y etiquetas de los datos de entrenamiento
    - x_val, y_val conjunto de entrenamiento y etiquetas de validación
    '''

    ##########################
    
    print('\n','-'*60)
    print (f' Evaluando {nombre_modelo}')
    print('-'*60)
    # Error cuadrático medio
    # predecimos test acorde al modelo
    modelo.fit(x, y)
    prediccion = modelo.predict(x)
    prediccion_test = modelo.predict(x_test)

    Etest=r2_score(y_test, prediccion_test)
    Ein=r2_score(y, prediccion)
    print("E_in en entrenamiento: ",Ein)
    print("E_test en validación: ",Etest)
    return  modelo.feature_importances_
'''      
def GraficaError(num_estimadores, resultados):
    plt.plot( num_estimadores, resultados['mean_test_score'], c = 'red', label='R2') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
    plt.legend();
    plt.title("Evolución del coeficiente R2 para n_estimators")
    plt.xlabel('n_estimators')
    plt.ylabel('R2')
    plt.figure()
    plt.show()
'''
 
def GraficaRegularizacion(E_in,E_val,alpha):
    plt.plot( alpha, E_in, c = 'orange', label='E_in') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
    plt.plot( alpha, E_val, c = 'blue', label='E_val') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
    plt.legend();
    plt.title("Influencia de la regularización en train y validación")
    plt.xlabel('alpha')
    plt.ylabel('R2')
    plt.figure()
    plt.show()
#################################################################
###################### Modelos a usar ###########################

#Modelo: Random Forest aplicado a Regresión
#Grid de Parámetros
print('\nRandom Forest aplicado a Regresión \n')

num_estimadores =[]
for i in range(50,350,50):
    num_estimadores.append(i)
    
parametros = {
     'max_features' :['sqrt'],
    'n_estimators' : num_estimadores
    }


modelo=RandomForestRegressor(random_state=0)

MuestraResultadosVC(modelo,parametros, x_train, y_train)


#------------------- Ajuste con outliers --------------------------------

Parada('Ajustamos ahora con Outliers')
  


resultados=MuestraResultadosVC(modelo,parametros, x_train_outliers_normalizado, y_train_con_outliers)

#En esta gráfica vemos que el coeficiente R^2 es creciente
GraficaError(num_estimadores,resultados)

num_estimadores =[]
for i in range(200,325,25):
    num_estimadores.append(i)
    
parametros = {
     'max_features' :['sqrt'],
    'n_estimators' : num_estimadores
    }

resultados=MuestraResultadosVC(modelo,parametros, x_train_outliers_normalizado, y_train_con_outliers)

#En esta gráfica vemos que el coeficiente R^2 parece alcanzar el máximo entre 260-290 estimadores
GraficaError(num_estimadores, resultados, 'n_estimators')

num_estimadores =[]
for i in range(260,295,5):
    num_estimadores.append(i)
    
parametros = {
     'max_features' :['sqrt'],
    'n_estimators' : num_estimadores
    }

resultados=MuestraResultadosVC(modelo,parametros, x_train_outliers_normalizado, y_train_con_outliers)

#En esta gráfica vemos que el coeficiente R^2 es creciente
GraficaError(num_estimadores, resultados, 'n_estimators')
Parada()

num_estimadores =[]
for i in range(280,292,2):
    num_estimadores.append(i)
    
parametros = {
     'max_features' :['sqrt'],
    'n_estimators' : num_estimadores
    }

resultados=MuestraResultadosVC(modelo,parametros, x_train_outliers_normalizado, y_train_con_outliers)

#En esta gráfica vemos que el coeficiente R^2 es creciente
GraficaError(num_estimadores, resultados, 'n_estimators')

Parada()
print("\n Veamos si hay Sobrajuste: ")

x_entrenamiento, x_validacion, y_entrenamiento, y_validacion=train_test_split(
    x_train_outliers_normalizado, y_train_con_outliers,
    test_size= 0.2,
    shuffle = True, 
    random_state=1)


modelo=RandomForestRegressor(max_features='sqrt',n_estimators=290,random_state=0)
Evaluacion_validacion(modelo,x_entrenamiento,y_entrenamiento,x_validacion,y_validacion,'Random Forest')

Parada()

print("\n\nTratamos de reducir el sobreajuste: ")

alpha =np.arange(0.0, 0.001, 0.0001).tolist()
  

for i in alpha:
    print("\n alpha=",i)
    modelo=RandomForestRegressor(max_features='sqrt',n_estimators=290,ccp_alpha=i, random_state=0)
    Evaluacion_validacion(modelo,x_entrenamiento,y_entrenamiento,x_validacion,y_validacion,'Random Forest')


E_in=[0.9531783709010848,0.8776642587554613,0.822134867592024,0.7791942044642272,0.7480678478671681,0.7251329558659247,0.7070096167353577,0.6934033698830031, 0.6827806319388955,0.6733809257943127]
E_val=[0.6851165863802047,0.6828587662709389,0.6783327423991174,0.6700327984832091,0.6619702856867934,0.6550114680674126,0.6489352273229924,0.6426441681200656,0.6374229820589261,0.6321236114445636]

GraficaRegularizacion(E_in,E_val,alpha)
Parada()

#El modelo final será con alpha=0.0006
modelo=RandomForestRegressor(max_features='sqrt',n_estimators=290,ccp_alpha=0.0006, random_state=0)

importancias=Evaluacion_test_randomforest(modelo,x_train_outliers_normalizado,y_train_con_outliers,x_test_outliers_normalizado,y_test,'Random Forest')
#Evaluacion(modelo,x_train_con_outliers,y_train_con_outliers,x_test,y_test,'Random Forest')

caracteristicas =np.arange(1, 101, 1).tolist()
plt.bar(caracteristicas,importancias, color='darkblue', align='center')
plt.title ('Importancia de cada característica')
plt.show()

# Cogemos los elementos más importantes del árbol del decisión 
importancias=importancias.tolist()
maximo_1=max(importancias)
max_index1=importancias.index(maximo_1)
importancias.pop(max_index1)

maximo_2=max(importancias)
max_index2=importancias.index(maximo_2)
importancias.pop(max_index2)

maximo_3=max(importancias)
max_index3=importancias.index(maximo_3)
importancias.pop(max_index3)

print("Los atributos más importantes son: ", max_index1, max_index2, max_index3)


score = np.mean(cross_val_score(modelo, x, y, cv = k_folds, scoring="r^2",n_jobs=-1)

