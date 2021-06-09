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
def Evaluacion( modelos, x, y, x_test, y_test, k_folds, nombre_modelo):
    '''
    Función para automatizar el proceso de experimento: 
    1. Ajustar modelo.
    2. Aplicar validación cruzada.
    3. Medir tiempo empleado en ajuste y validación cruzada.
    4. Medir Error cuadrático medio.   
    INPUT:
    - modelo: Modelo con el que buscar el clasificador
    - X datos entrenamiento. 
    - Y etiquetas de los datos de entrenamiento
    - x_test, y_test
    - k-folds: número de particiones para la validación cruzada
    OUTPUT:
    '''

    ###### constantes a ajustar
    numero_trabajos_paralelos_en_validacion_cruzada = 2 
    ##########################
    
    print('\n','-'*60)
    print (f' Evaluando {nombre_modelo}')
    print('-'*60)


    print('\n------ Comienza Validación Cruzada------\n')        

    #validación cruzada
    np.random.seed(0)
    tiempo_inicio_validacion_cruzada = time.time()

    best_score = -np.infty
    for model in modelos:
        print(model)
        score = np.mean(cross_val_score(model, x, y, cv = k_folds, scoring="r2",n_jobs=-1))
        print('Error cuadrático medio del modelo con cv: ',score)
        print()

        if best_score < score:
            best_score = score
            best_model = model

    tiempo_fin_validacion_cruzada = time.time()
    tiempo_validacion_cruzada = (tiempo_fin_validacion_cruzada
                                 - tiempo_inicio_validacion_cruzada)

    print(f'\nTiempo empleado para validación cruzada: {tiempo_validacion_cruzada}s\n')
    
    print('\n\nEl mejor modelo es: ', best_model)
    print('E_val calculado en cross-validation: ', best_score)

    # Error cuadrático medio
    # predecimos test acorde al modelo
    best_model.fit(x, y)
    prediccion = best_model.predict(x)
    prediccion_test = best_model.predict(x_test)

    Etest=r2_score(y_test, prediccion_test)
    Ein=r2_score(y, prediccion)
    print("Error cuadratico medio en entrenamiento: ",Ein)
    print("Error cuadratico medio en test: ",Etest)

    return best_model
  
              
   
#################################################################
###################### Modelos a usar ###########################
k_folds=10 #Número de particiones para cross-Validation
'''
print('\nPrimer Modelo: SVM aplicado a Regresión con kernel polinómico\n')
#Primer Modelo: Regresión Lineal con SGD para obtener vector de pesos
#Hago un vector con modelos del mismo tipo pero variando los parámetros
modelos2=[SVR(kernel='poly', degree=d, gamma=g, coef0=coef0, C=c, epsilon=e) for e in [0.1,0.3,0.5] for g in ['scale','auto'] for coef0 in [0.0,1.0] for d in [2,3] for c in [0.5,1,1.5]]

#Usando cross-Validation tomo el modelo con los parámetros que mejor comportamiento tiene
modelo_elegido2=Evaluacion( modelos2, x_train, y_train, x_test, y_test, k_folds, 'SVM aplicado a Regresión')
print ("Ajustamos ahora con Outliers")
modelo_elegido2=Evaluacion( modelos2, x_train_outliers_normalizado, y_train_con_outliers, x_test, y_test, k_folds, 'SVM aplicado a Regresión')
'''


print('\nSegundo Modelo: SVM aplicado a Regresión con kernel RBF\n')
#Primer Modelo: Regresión Lineal con SGD para obtener vector de pesos
#Hago un vector con modelos del mismo tipo pero variando los parámetros
modelos2=[SVR(kernel='rbf', degree=d, gamma=g, coef0=coef0, C=c, epsilon=e) for e in [0.1,0.3,0.5] for g in ['scale','auto'] for coef0 in [0.0,1.0] for d in [2,3] for c in [0.5,1,1.5]]

#Usando cross-Validation tomo el modelo con los parámetros que mejor comportamiento tiene
modelo_elegido2=Evaluacion( modelos2, x_train, y_train, x_test, y_test, k_folds, 'SVM aplicado a Regresión')
print ("Ajustamos ahora con Outliers")
modelo_elegido2=Evaluacion( modelos2, x_train_outliers_normalizado, y_train_con_outliers, x_test, y_test, k_folds, 'SVM aplicado a Regresión')





















