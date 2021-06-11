####################################################
#   Trabajo final: Communities And Crimes
####################################################
#   Autores:
#      - Alejandro Borrego Megías
#      - Blanca Cano Camarero
#   Fecha: Principios junio 2021
####################################################


#############################
#######  BIBLIOTECAS  #######
#############################
# Biblioteca lectura de datos
# ==========================
import pandas as pd

# matemáticas
# ==========================
import numpy as np


# lectura de datos
# ==========================
from pandas import read_csv

# Modelos lineales de clasificación a usar   
# =========================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor

# Modelos no lineales a usar
# =============================================
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier

# Preprocesado 
# ==========================
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# visualización de datos
# ==========================
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns # utilizado para pintar la matriz de correlación 

# Validación cruzada
# ==========================
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score


# metricas
# ==========================
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Otros
# ==========================
from operator import itemgetter #ordenar lista
import time

np.random.seed(1)

############# Constantes básicas  ############

NOMBRE_FICHERO  = './datos/communities.data'
SEPARADOR = ','

NUMERO_CPUS_PARALELO = 4 


######################## Funciones básicas ##############

def Parada(mensaje = None):
    '''
    Hace parada del código y muestra un mensaje en tal caso 
    '''
    print('\n-------- fin apartado, enter para continuar -------\n')
    #input('\n-------- fin apartado, enter para continuar -------\n')
    
    if mensaje:
        print('\n' + mensaje)


        
def LeerDatos (nombre_fichero, separador):
    '''
    Input: 
    - file_name: nombre del fichero path relativo a dónde se ejecute o absoluto
    La estructura de los datos debe ser: 
       - Cada fila un vector de características con su etiqueta en la última columna.

    Outputs: x,y
    x: matriz de filas de vector de características
    y: vector fila de la etiquetas 
    
    '''
    datos = np.genfromtxt(nombre_fichero,delimiter=separador)
    y = datos[:,-1].copy() #cogemos las etiquetas 
    x = datos[:,5:-1].copy() #obviamos las 5 primeras (no predictoras) y la última (etiquetas)

    return x,y

def NoNan(a, media, desviacion_tipica):
    '''
    si a == nan entonces devuelve media + random(desviacion_tipica
    '''
    if np.isnan(a):
        extremo = 1.5*desviacion_tipica
        return media +  np.random.uniform(-extremo, extremo,1)[0]

    else:
        return a

def TratamientoDatosPerdidos(x, porcentaje_eliminacion = 20):
    '''
    INPUT
    x: atributos
    porcentaje_eliminacion: a partir de qué porcentaje se eliminan del conjutno de datos
    OUTPUT x con el siguiemnte criterio de datos perdidos:

    eliminamos los atributos que tengan una pérdida mayor o igual del $20\%$
    para el resto los completamos con la media de valores válidos de ese atributo más un valor aleatorio dentro del intervalo $[-1.5 \sigma, 1.5 \sigma ]$ siendo $\sigma$ la desviación típica de la variable dicha.  
    '''

    # numero umbral de datos perdidos para eliminar dicho atributo
    umbral_perdido = len(x) * porcentaje_eliminacion/100
   
    # eliminamos los atributos que tengan más del porcentaje_eliminacion de datso perdidos                  
    x_eliminada = x.T[[ sum(np.isnan(atributo))
                        < umbral_perdido for atributo in x.T ]]

    
    
    # conservamos el resto por criterio de media más valor random
    for i, atributo in enumerate(x_eliminada):
        suma = sum(np.isnan(atributo))
        if (suma  > 0):
        #calculamos media de los valores que son válido
            filtrados = atributo[list(map(lambda x: not x, np.isnan(atributo)))]
            media = filtrados.mean()
            desviacion_tipica = filtrados.std()
            
            
            x_eliminada[i] =  [NoNan(j, media, desviacion_tipica)
                               for j in x_eliminada[i]]
            
            
     
    return x_eliminada.T

    

x,y = LeerDatos(NOMBRE_FICHERO, SEPARADOR)
x = TratamientoDatosPerdidos(x, porcentaje_eliminacion = 20)

###### separación test y entrenamiento  #####
ratio_test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size= ratio_test_size,
    shuffle = True, 
    random_state=1)



####  Comprobació balanceo de los datos


def BalanceadoRegresion(y, divisiones = 20):
    '''
    INPUT: 
    y: Etiquetas
    divisiones: número de agrupaciones en las que dividir el rango de etiquetas

    OUTPUTS: 
    void
    imprime en pantalla detalles y gráfica
 
    '''
    min_y = min(y)
    max_y = max(y)

    longitud = (max_y - min_y)/divisiones    
    extremo_inferior = min_y
    extremo_superior = min_y + longitud

    datos_en_rango = np.arange(divisiones)
    cantidad_minima = np.infty
    cantidad_maxima = - np.infty
    indice_minimo = None
    indice_maximo = None
    
    
    for i in range(divisiones):
        datos_en_rango[i] = np.count_nonzero(
            (extremo_inferior <= y ) &
            (y <= extremo_superior)
        )
        extremo_inferior = extremo_superior
        extremo_superior += longitud

        if cantidad_minima > datos_en_rango[i]:
            cantidad_minima = datos_en_rango[i]
            indice_minimo = i
        if cantidad_maxima < datos_en_rango[i]:
            cantidad_maxima = datos_en_rango[i]
            indice_maximo = i

    # imprimimos valores
    print('\nDistribución de las etiquetas de y en rango valores de [%.4f, %.4f] \n'%(min_y, max_y))
    
    print('Número total de etiquetas ', len(y))
    
    print('Cantidad mínima de datos ', cantidad_minima)
    extremo_inferior = min_y + longitud * indice_minimo
    print(f'Alcanzada en intervalo [%.4f , %.4f]'%
          (extremo_inferior , (extremo_inferior + longitud)))
    
    print('Cantidad máxima de datos ', cantidad_maxima)
    extremo_inferior = min_y + longitud * indice_maximo
    print(f'Alcanzada en intervalo [%.4f , %.4f]'%
          (extremo_inferior , (extremo_inferior + longitud)))
    print('La media de datos por intervalo es %.4f'% datos_en_rango.mean())
    print('La desviación típica de datos por intervalos es %.4f' % datos_en_rango.std())
    print('La mediana de y es %4.f' % np.median(y))
    print('La media de datos %.4f'% y.mean())
    print('La desviación típica de datos %.4f'% y.std())
    
    Parada('Gráfico de balanceo')
    # gráfico  de valores
    plt.title('Número de etiquetas por rango de valores')
    plt.bar([i*longitud + min_y for i in range(len(datos_en_rango))],
            datos_en_rango, width = longitud * 0.9)
    plt.xlabel('Valor de la etiqueta y (rango de longitud %.3f)'%longitud)
    plt.ylabel('Número de etiquetas')
    plt.show()
    
# DESCOMENTAR
#BalanceadoRegresion(y_train, divisiones = 25)


## quitamso outliers

def EliminaOutliers(y, proporcion_distancia_desviacion_tipica = 3.0):
    '''
    OUTPUT
    (muestra en pantalla alguna información sobre el cálculo de la máscara)
    mascara_y_sin_outliers
    INPUT
    y: etiquetas a las que quitar el outliers

    proporcion_distancia_desviacion_tipica = 3.0
   
    Información sobre cómo elegir la proporcion_distancia_desviacion_tipica:
    Alguna relaciones: 
    distancia | intervalo de confianza:
    1         | 0.683
    1.282     | 0.8
    1.644     | 0.9
    2         | 0.954
    3         | 0.997
    3.090     | 0.998
    4         | 0.9999367
    5         | 0.99999942
    
    

    https://es.wikipedia.org/wiki/Distribuci%C3%B3n_normal#Desviaci%C3%B3n_t%C3%ADpica_e_intervalos_de_confianza
    '''
    media = y.mean()
    desviacion_tipica = y.std()

    corte = desviacion_tipica * proporcion_distancia_desviacion_tipica
    limite_inferior = media - corte
    limite_superior = media + corte

    mascara_y_sin_outliers = (limite_inferior <= y ) & (y <= limite_superior)
       
    # imprimimos información
    print('\n___ Información eliminando outliers___')
    print('Proporcion de deistancia a la desviación típica tomada ',
          proporcion_distancia_desviacion_tipica)
    print('Media de los datos %.4f'% media)
    print('Desviación típica %.4f'% desviacion_tipica)
    print('Rango de valores aceptado [%.4f, %.4f]' %
          (limite_inferior, limite_superior ))

    print(f'Número de outliers eliminados { np.count_nonzero(mascara_y_sin_outliers == False)}')

    return mascara_y_sin_outliers


mascara_sin_outliers = EliminaOutliers(y_train, proporcion_distancia_desviacion_tipica = 3)

#mantenemos copia para después compararlos  
x_train_con_outliers = np.copy(x_train)
y_train_con_outliers = np.copy(y_train)

x_train = x_train[mascara_sin_outliers]
y_train = y_train[mascara_sin_outliers]


Parada('Normalizamos los datos')
#Normalizamos los datos para que tengan media 0 y varianza 1

x_train_sin_normalizar = np.copy(x_train)
x_test_sin_normalizar = np.copy(x_test)

scaler = StandardScaler()
x_train = scaler.fit_transform( x_train )
x_test = scaler.transform( x_test) 

scaler_outliers = StandardScaler()
x_train_outliers_normalizado = scaler_outliers.fit_transform( x_train_con_outliers )
x_test_outliers_normalizado = scaler_outliers.transform(x_test_sin_normalizar)


print("\nDimensiones de los datos con las distintas transformaciones: \n")
print("\nMatriz x de características de entrenamiento con outliers sin normalizar: ", x_train_con_outliers.shape)
print("\nVector y de etiquetas de entrenamiento con outliers: ", y_train_con_outliers.shape)
print("-------------------------------------------------------------------------------------")
print("\nMatriz x de características de entrenamiento sin outliers normalizada: ", x_train_sin_normalizar.shape)
print("\nVector y de etiquetas de entrenamiento sin outliers: ", y_train.shape)


#######################################################################
#
#       Función para evaluar los distintos métodos
#
##########################################################################



def Evaluacion( clasificador,
                x, y, 
                k_folds,
                nombre_modelo,
                metrica_error):
    '''
    Función para automatizar el proceso de experimento: 
    1. Ajustar modelo.
    2. Aplicar validación cruzada.
    3. Medir tiempo empleado en ajuste y validación cruzada.
    4. Medir la precisión.   

    INPUT:
    - Clasificador: Modelo con el que buscar el clasificador
    - X datos entrenamiento. 
    - Y etiquetas de los datos de entrenamiento
    - k-folds: número de particiones para la validación cruzada
    - metrica_error: debe estar en el formato sklearn (https://scikit-learn.org/stable/modules/model_evaluation.html)

    OUTPUT:
    '''

    ###### constantes a ajustar
    numero_trabajos_paralelos_en_validacion_cruzada = NUMERO_CPUS_PARALELO
    ##########################
    
    print('\n','-'*20)
    print (f' Evaluando {nombre_modelo}')
    
    #print(f'\n------ Ajustando modelo------\n')        
    tiempo_inicio_ajuste = time.time()
    
    #ajustamos modelo 
    ajuste = clasificador.fit(x,y) 
    tiempo_fin_ajuste = time.time()

    tiempo_ajuste = tiempo_fin_ajuste - tiempo_inicio_ajuste

     

    #validación cruzada
    tiempo_inicio_validacion_cruzada = time.time()

    score_validacion_cruzada = cross_val_score(
        clasificador,
        x, y,
        scoring = metrica_error,
        cv = k_folds,
        n_jobs = numero_trabajos_paralelos_en_validacion_cruzada
    )
    tiempo_fin_validacion_cruzada = time.time()
    
    tiempo_validacion_cruzada = tiempo_fin_validacion_cruzada - tiempo_inicio_validacion_cruzada

    print('\tscore_validacion_cruzada')
    print(score_validacion_cruzada)
    
    print('--------------------')
    print ('\tMedia error de validación cruzada {:.5f} '.format(score_validacion_cruzada.mean()))
    print('--------------------')
    
    print('\tDesviación típica del error de validación cruzada {:.5f} '.format(score_validacion_cruzada.std()))
    print('\tTiempo empleado para el ajuste: {:.4f}s '.format(tiempo_ajuste))
    print('\tTiempo empleado para el validacion cruzada {:.4f}s'.format(tiempo_validacion_cruzada))



    return ajuste
