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
    print("E_in en entrenamiento: ",Ein)
    print("E_test en test: ",Etest)

    return best_model
  
              
   
#################################################################
###################### Modelos a usar ###########################
k_folds=10 #Número de particiones para cross-Validation

#Modelo: Random Forest aplicado a Regresión
#Grid de Parámetros
print('\nRandom Forest aplicado a Regresión \n')
parametros = {
     'max_features' :['auto','sqrt'],
    'n_estimators' : [50,100,150,200]
    }


modelo=RandomForestRegressor(random_state=0)
MuestraResultadosVC(modelo,parametros, x_train, y_train)

Parada("Pulse una tecla para continuar")

print ("Ajustamos ahora con Outliers")

MuestraResultadosVC(modelo,parametros, x_train_outliers_normalizado, y_train_con_outliers)

