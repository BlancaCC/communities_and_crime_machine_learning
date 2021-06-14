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
from sklearn.svm import LinearSVR


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


# ESTO SOBRA 
np.random.seed(1)
######### CONSTANTES #########  
NOMBRE_FICHERO_REGRESION = './datos/train.csv'

################ funciones auxiliares
def Evaluacion_test_modelos_lineales( modelo, x, y, x_test, y_test, nombre_modelo):
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
    
    
      
def GraficaError(param, resultados, Hiperparametro):
    plt.plot( param, resultados['mean_test_score'], c = 'red', label='R2') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
    plt.legend();
    plt.title("Evolución del coeficiente R2")
    plt.xlabel(Hiperparametro)
    plt.ylabel('R2')
    plt.figure()
    plt.show()
      
  
Parada('Pulsa una tecla para continuar')
#################################################################
###################### Modelos a usar ###########################

print('\nModelo: Regresión lineal con SVM\n')
'''
#Modelo: Regresión Lineal con SVM
#Sin Outliers
modelo=LinearSVR(random_state=0, max_iter=15000)
x_entrenamiento=TransformacionPolinomica(2,x_train)


print("Ajustamos el término de regularización C")
C=[0.5,1,1.5,2]
parametros = {
     'C' : C
    }

resultados=MuestraResultadosVC(modelo,parametros, x_entrenamiento, y_train)
GraficaError(C,resultados,'C')

C=[0.1,0.2,0.3,0.4,0.5]
parametros = {
     'C' : C
    }

resultados=MuestraResultadosVC(modelo,parametros, x_entrenamiento, y_train)
GraficaError(C,resultados,'C')

e=[0.0,0.1,0.2,0.3]
parametros = {
     'epsilon' : e
    }

resultados=MuestraResultadosVC(modelo,parametros, x_entrenamiento, y_train)
GraficaError(e,resultados,'epsilon')

print("Probamos epsilon entre 0 y 0.9")
e=[0.0,0.03,0.06,0.09]
parametros = {
     'epsilon' : e
    }

resultados=MuestraResultadosVC(modelo,parametros, x_entrenamiento, y_train)
GraficaError(e,resultados,'epsilon')
'''
x_entrenamiento=TransformacionPolinomica(2,x_train)

modelo=LinearSVR(epsilon=0.03, C=0.1,random_state=0, max_iter=15000)

x_test_polinomios=TransformacionPolinomica(2,x_test)
Evaluacion_test_modelos_lineales( modelo, x_entrenamiento, y_train, x_test_polinomios, y_test,'SVM aplicado a Regresión')
