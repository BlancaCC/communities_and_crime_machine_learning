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
    OUTPUT x con el siguiemnte criterio de datos perdidos
    '''

    # umbra datos perdidos para eliminar
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
    

BalanceadoRegresion(y_train, divisiones = 25)
