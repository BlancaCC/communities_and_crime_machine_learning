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

# Modelos lineales de Regresión a usar   
# =========================================
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR


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

### Validación cruzada
def Evaluacion( modelo, x, y, x_val, y_val, nombre_modelo):
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
 
    
### Evaluación test
def Evaluacion_test( modelo, x, y, x_test, y_test, nombre_modelo):
    '''
    Función para calcular error en entrenamiento y validación  
    INPUT:
    - modelo: Modelo con el que calcular los errores
    - X datos entrenamiento. 
    - Y etiquetas de los datos de entrenamiento
    - x_test, y_test conjunto de entrenamiento y etiquetas de validación
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
    print("E_test en test: ",Etest)
 
  
def GraficaError(param, resultados,hiperparametro):
    plt.plot( param, resultados['mean_test_score'], c = 'red', label='R2') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
    plt.legend();
    plt.title("Evolución del coeficiente R2")
    plt.xlabel(hiperparametro)
    plt.ylabel('R2')
    plt.figure()
    plt.show()              
   
#################################################################
###################### Modelos a usar ###########################
print('\nPrimer Modelo: SVM aplicado a Regresión con kernel polinómico\n')
parametros = {
     'degree':[2],
    #'gamma' : ['scale', 'auto']
    #'C' : [0.1,0.2,0.5,1],
    #'epsilon': [0.01,0.05,0.1,0.2]
    }


modelo=SVR(kernel='poly')
MuestraResultadosVC(modelo,parametros, x_train, y_train)

Parada("Pulse una tecla para continuar")

print ("Ajustamos ahora con Outliers")

MuestraResultadosVC(modelo,parametros, x_train_outliers_normalizado, y_train_con_outliers)

Parada("Pulse una tecla para continuar")

print('\nSegundo Modelo: SVM aplicado a Regresión con kernel RBF\n')
#Segundo Modelo: SVM aplicado a Regresión con kernel RBF
#Hago un vector con modelos del mismo tipo pero variando los parámetros
modelo=SVR(kernel='rbf')
MuestraResultadosVC(modelo,parametros, x_train, y_train)

Parada("Pulse una tecla para continuar")

print ("Ajustamos ahora con Outliers")
MuestraResultadosVC(modelo,parametros, x_train_outliers_normalizado, y_train_con_outliers)


print("\nComo con Outliers da mejor resultado, vamos a estimar los parámetros para el ajuste con Outliers para los SVM con kernel polinómico y rbf")
modelo2=SVR(kernel='rbf',degree=2)

print("\n\nPrimero estimamos gamma")
parametros = {
    'gamma' : ['scale', 'auto']
    }
print("\n------------------SVM Kernel rbf------------------\n")
MuestraResultadosVC(modelo2,parametros, x_train_outliers_normalizado, y_train_con_outliers)

print("\n\nUna vez estimado el mejor valor de gamma para cada modelo veamos el valor de C\n")
modelo2=SVR(kernel='rbf',degree=2,gamma='scale')
C=[0.1,0.2,0.5,1]
parametros = {
    'C' : C,
    }

print("\n------------------SVM Kernel rbf------------------\n")
resultados=MuestraResultadosVC(modelo2,parametros, x_train_outliers_normalizado, y_train_con_outliers)
GraficaError(C,resultados,"C")

C =np.arange(0.1, 0.31, 0.01).tolist()
parametros = {
    'C' : C,
    }
print("\n------------------SVM Kernel rbf------------------\n")
resultados=MuestraResultadosVC(modelo2,parametros, x_train_outliers_normalizado, y_train_con_outliers)
GraficaError(C,resultados,"C")


print("\n\nFinalmente ajustamos epsilon")
modelo1=SVR(kernel='poly',degree=2,gamma='auto',C=0.28)
e= [0.01,0.05,0.1,0.2]
parametros = {
    'epsilon':e
}

print("\n------------------SVM Kernel rbf------------------\n")
resultados=MuestraResultadosVC(modelo2,parametros, x_train_outliers_normalizado, y_train_con_outliers)
GraficaError(e,resultados,"epsilon")


e =np.arange(0.001, 0.051, 0.001).tolist()
parametros = {
    'epsilon':e
}
print("\n------------------SVM Kernel rbf------------------\n")
resultados=MuestraResultadosVC(modelo2,parametros, x_train_outliers_normalizado, y_train_con_outliers)
GraficaError(e,resultados,"epsilon")



modelo_definitivo=SVR(kernel='poly',degree=2,gamma='auto',C=0.28,epsilon=0.03)
Evaluacion_test(modelo_definitivo, x_train_outliers_normalizado, y_train_con_outliers, x_test_outliers_normalizado, y_test, "SVR con Kernel rbf")












