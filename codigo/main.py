####################################################
#   Trabajo final: Communities And Crimes
####################################################
#   Autores:
#      - Alejandro Borrego Megías
#      - Blanca Cano Camarero
#   Fecha: Principios junio 2021
####################################################

## Estructura del fichero
# - Biblioteca
# - Funciones generales
# - Modelos 
#############################
#######  BIBLIOTECAS  #######
#############################
# Biblioteca lectura de datos
# ==========================
import pandas as pd

# matemáticas
# ==========================
import numpy as np
from math import floor 


# lectura de datos
# ==========================
from pandas import read_csv

# Modelos lineales de clasificación a usar   
# =========================================
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.dummy import DummyRegressor


# Modelos no lineales a usar
# =============================================
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor

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
from sklearn.model_selection import GridSearchCV
from numpy.ma import getdata # para rescatar elementos del grid 


# metricas
# ==========================
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


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
    print('\n-------- fin apartado, enter para continuar -------\n\n')
    #input('\n-------- fin apartado, enter para continuar -------\n\n')
    
    if mensaje:
        print('-'*40)
        print('\n' + mensaje )
        print('-'*40)


        
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

x_train_con_outliers = np.copy(x_train) #Sin normalizar con outliers
y_train_con_outliers = np.copy(y_train)

x_train = x_train[mascara_sin_outliers]
y_train = y_train[mascara_sin_outliers]


Parada('Normalizamos los datos')
#Normalizamos los datos para que tengan media 0 y varianza 1

x_train_sin_normalizar = np.copy(x_train) # Sin normalizar sin outliers
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

     
##########################################################################


'''
def EvaluacionSimple( clasificador,
                x, y, 
                k_folds,
                nombre_modelo,
                metrica_error):
'''
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
'''



 
def MuestraResultadosVC( estimador, parametros, x_entrenamiento, y_entrenamiento):
    '''
    Dados una serie de parametros y un estimador muestra en pantalla los mejores hiperparámetros junto con su error, así como una tabla que resuma todo 
    '''
    grid = GridSearchCV(
        estimator = estimador,
        param_grid = parametros,
        scoring = 'r2',
        n_jobs = -1,
        verbose = 0 # cero to have no verbose 
        #cv = croosvalidation
    )

    grid.fit(x_entrenamiento, y_entrenamiento)
    
    print ('Ya se ha terminado el croosValidation')
    #Parada('Procesamos a ver los mejores estimadores: ')
    print('Procesamos a ver los mejores estimadores: ')
    


    print('Mejores parámetros: ', grid.best_params_)
    print('Con una $R^2$ de: ', grid.best_score_ , '\n' )


    ## Función para evaluar los resultados del cross validation por orden
    grid.cv_results_['rank_test_score'] # devuelve por orden 
    
    
    l = len(grid.cv_results_['rank_test_score'])
    rank_indice = list(
        zip(
            grid.cv_results_['rank_test_score'],
            [i for i in range(l)]
        )
    )

    rank_indice.sort(key = lambda x: x[0]) # ordenamos por ranking
    parametros = grid.param_grid.keys()

    print(' |Parámetros | $R^2$ medio | Desviación tipica $R^2$| Ranking | tiempo medio ajuste |      ')
    print('|---|---|---|---|---|    ')
    
    for ranking, indice in rank_indice:
        # imprimimos las caracterísitcas de los parámetros evaluados
        print ('| ', end = '')
        for p in parametros:
            print ( p ,
                    getdata( grid.cv_results_['param_'+p])[indice], end = ' ')

        print('|', end = ' ')
        print( '{:.4f}'.format(grid.cv_results_['mean_test_score'][indice]),
               end = ' | ')
        print( '{:.4f}'.format(grid.cv_results_['std_test_score'][indice]),
                   end = ' | ')
        print( '{:.0f}'.format(grid.cv_results_['rank_test_score'][indice]),
               end = ' | ')
        print( '{:.4f}'.format(grid.cv_results_['mean_fit_time'][indice]),
               end = '|     \n')
    

    return grid.cv_results_


############## Función para dibujar los resultados #########

def GraficaError(parametros, resultados, nombre_parametros = None):
    '''
    INPUT: 
    - parametros: lista con los parámetros con los que estamos probando
    - resutlados: varieble que almacena el grid.cv_results_, el output de la función MuestraResultadosVC
    - nombre_parametro que poner en el título y leyenda

    OUTPUT: Void
    Muestra en pantalla la figura deseada
    '''
    plt.clf()
    plt.plot( parametros, resultados['mean_test_score'], c = 'red', label='R2') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
    if (nombre_parametros == None):
        plt.title("Evolución del coeficiente R2")
        plt.xlabel('Valores parámetros')
    else:
        plt.title(f'Evolución del coeficiente R2 {nombre_parametros}')
        plt.xlabel(nombre_parametros)
        
    plt.ylabel('R2')
    plt.legend()
    plt.show()



def GraficaComparativaEinEval( ejex, E_in,E_val, etiqueta):
    '''
    INPUT:
    - ejex: parámetro que deseamos comparar
    - E_in , E_val: lista de los respectivos errores
    - etiqueta: string con el nombre de lo que estamos comparando
    '''
    plt.clf()
    plt.plot( ejex, E_in, c = 'orange', label='$E_{in}$') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
    plt.plot( ejex, E_val, c = 'blue', label='$E_{val}$') #Para representarlo, despejo x2 de la ecuación y represento la función resultante en 2D
    plt.legend();
    plt.title(f'Influencia de {etiqueta}  en train y validación')
    plt.xlabel(etiqueta)
    plt.ylabel('R2')
    plt.show()

def GraficaRegularizacion(E_in,E_val,alpha):
    '''Versón particular de GraficaComparativaEinEval
    '''
    GraficaComparativaEinEval( alpha, E_in,E_val, 'regularización')


    

########################################################################
## Función de transformación de datos
def TransformacionPolinomica( grado,x):
    '''
    Devuelve un vector con transformaciones polinómicas 
    '''
    x_aux = np.copy(x)
    for i in range(1,grado):
        x_aux = x_aux*x_aux
        x = np.c_[x_aux, x]
    return x



####################################################
# Comparativa de evolución del error frente tamaño de entrenamiento
###################################################



def TablasComparativasEvolucion (tam, ein, e_val, dos_separadas = True):
    '''
    Muestra gráficas comparativas de la variación 
de los errores en función del tamaño de entrenamiento.

    - Si dos_separadas es True se muestra también la variación de los valores separados.
    - si no solo una.

    Esta función se utiliza exclusivamente en la función: 
    EvolucionDatosEntrenamiento 
    '''

    if dos_separadas:
        Parada('Comarativas evolución error por tamaño muestra separadas')
        plt.figure(figsize = (9,9))

        plt.subplot(121)
        plt.plot(tam,ein)
        plt.title('Variación $R^2$ $E_{in}$')
        plt.xlabel('Tamaño set entrenamiento')
        plt.ylabel('$R^2$')

        plt.subplot(122)
        plt.plot(tam, e_val)
        plt.title('Variación $R^2$ $E_{eval}$')
        plt.xlabel('Tamaño set entrenamiento')
        plt.ylabel('$R^2$')

        plt.show()
    

    # juntos
    Parada('Comarativas separadas')
    plt.title('Comparativas $R^2$')
    plt.plot(tam,ein, label = '$R^2$ $E_{in}$' )
    plt.plot(tam, e_val, label = '$R^2$ $E_{val}$' )

    plt.xlabel('Tamaño set entrenamiento')
    plt.ylabel('$R^2$')
    plt.legend()
    plt.show()
    
    
    
    
def EvolucionDatosEntrenamiento(modelo,
                                x, y,
                                numero_particiones,
                                porcentaje_validacion = 0.2,
                                dos_separadas = True):
    '''
    Dado un modelos muestra la evolución del Error in y Error out
    En función del tamaño de entrenamiento 

    INPUT:
    modelo: modelo a ajustar
    numero_particiones: numero natural positvo 
    porcentaje_validacion: porcentaje de x_entrenamiento que se usará para validación  
    x, y

    OUTPUT
    El error se calcula a partir de un subconjunto de tamaño porcentaje de 
    validación. 

    Las tablas que muestra depende de las definidas en la función 
    TablasComparativasEvolucion (tam, ein, eval)
    '''

    # retiramos subconjunto para test, para no hacer data snopping
    x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size= porcentaje_validacion,
    shuffle = True, 
    random_state=1)

    incremento = floor(len(y_train)/ numero_particiones)
    
    size_set_entrenamiento = [incremento*i for i in range(1,numero_particiones+1)]
    score_in = np.zeros( numero_particiones)
    score_out = np.zeros( numero_particiones)

    print('| Tamaño | $R^2$ $E_{in}$ | $R^2$ $E_{eval}$ |    ')
    print('|---'*3 , '|    ')
    
    for i,tam in enumerate(size_set_entrenamiento):
        #no considero necesario
        
        modelo.fit(x_train[:tam], y_train[:tam])

        score_in[i] = modelo.score(x_train[:tam], y_train[:tam])
        score_out[i] = modelo.score(x_test, y_test)
        print( '|{:.0f} | {:.4f} | {:.4f}'.format(
            tam,
            score_in[i],
            score_out[i]
        )
               ,
               end = '|     \n'
            )

    
    TablasComparativasEvolucion (
        size_set_entrenamiento,
        score_in,
        score_out,
        dos_separadas
        )


#######################################################
# Resultado finales de la selección final del modelo
#######################################################

def ConclusionesFinales( modelo,
                         x_train, y_train,
                         x_test, y_test,
                         mostrar_coeficientes = True,
                         leave_one_out = False):

    
    modelo.fit(x_train,y_train)

    E_in = modelo.score(x_train,y_train)
    E_test = modelo.score(x_test,y_test)

    # Entrenamos con todos los datos y devolvemos coeficientes:
    x = np.append(x_train, x_test).reshape(len(x_train) + len(x_test),
                                      len(x_train[0]))
    y = np.append(y_train, y_test)

    modelo.fit(x,y)
    E_in_total = modelo.score(x,y)

    # Imprimimos resultados
    print('R^2_in : {:.4f} , R^2_test :{:.4f}'.format(
        E_in, E_test ) )
    print('Tras entrenar con todos los datos: R^2_in : {:.4f} '.format(E_in_total))


    if mostrar_coeficientes:
        print( modelo.coef_)

    if leave_one_out == True:
        e_out = np.mean(cross_val_score(modelo,
                                x, y,
                                cv = len(y)//50,
                                scoring="r2",
                                n_jobs=-1))
        print('E_out {:.4f}'.format(e_out))
        


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
'''
    
# fin-modelos
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
    
#                         MODELOS


#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################



#################################################
# Lineal SVM
###################################################
print('#################################################')
Parada('Modelo: Regresión lineal con SVM')
print('#################################################')

#################################################################
###################### Modelos a usar ###########################
print('Esta parte tiene un costo computacional elevado y ha sido comentada')
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

Parada('Modelo final lineal')
x_entrenamiento=TransformacionPolinomica(2,x_train)

modelo=LinearSVR(epsilon=0.03, C=0.1,random_state=0, max_iter=15000)

x_test_polinomios=TransformacionPolinomica(2,x_test)
Evaluacion_test_modelos_lineales( modelo, x_entrenamiento, y_train, x_test_polinomios, y_test,'SVM aplicado a Regresión')




#################################################
# Random forest 
###################################################

#Modelo: Random Forest aplicado a Regresión
#Grid de Parámetros
''' DESCOMENTAR
Parada('\nRandom Forest aplicado a Regresión \n')

num_estimadores =[]
for i in range(50,350,50):
    num_estimadores.append(i)
    
parametros = {
     'max_features' :['sqrt'],
    'n_estimators' : num_estimadores
    }


modelo=RandomForestRegressor(random_state=0)

MuestraResultadosVC(modelo,parametros, x_train, y_train)

'''# DESCOMENTAR
#------------------- Ajuste con outliers --------------------------------
'''# DESCOMENTAR
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

Parada()

#En esta gráfica vemos que el coeficiente R^2 parece alcanzar el máximo entre 260-290 estimadores
GraficaError(num_estimadores, resultados, 'n_estimators')

Parada('Ahora outliers normalizados')

num_estimadores =[]
for i in range(260,295,5):
    num_estimadores.append(i)
    
parametros = {
     'max_features' :['sqrt'],
    'n_estimators' : num_estimadores
    }

resultados=MuestraResultadosVC(modelo,parametros, x_train_outliers_normalizado, y_train_con_outliers)

Parada()
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

Parada()
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
'''# DESCOMENTAR
#El modelo final será con alpha=0.0006
MODELO_RANDOM_FOREST=RandomForestRegressor(max_features='sqrt',n_estimators=290,ccp_alpha=0.0006, random_state=0)
'''# DESCOMENTAR
importancias=Evaluacion_test_randomforest(MODELO_RANDOM_FOREST,x_train_outliers_normalizado,y_train_con_outliers,x_test_outliers_normalizado,y_test,'Random Forest')
#Evaluacion(MODELO_RANDOM_FOREST,x_train_con_outliers,y_train_con_outliers,x_test,y_test,'Random Forest')

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


Parada('Estudio preliminar de los parámetros (tarda un minuto aprox)')
# Parámetros por defecto
MLP_1 = MLPRegressor(
    random_state=1,
    max_iter=500,
    shuffle = True,
    activation = 'logistic',

)

tam_capas = [50, 75, 100]

parametros = {
    'hidden_layer_sizes' : [ (i,j) for i in tam_capas for j in tam_capas],
    'solver':['sgd', 'adam']
}

 
resultados_1 = MuestraResultadosVC(
    MLP_1,
    parametros,
    x_train,
    y_train
    
)


## De este experimento los mejores resultados han sido
#Mejores parámetros:  {'hidden_layer_sizes': (100, 50), 'solver': 'adam'}
#Con una $R^2$ de:  0.6205065254947334   


# -----------------------------------------------------

# Procedamos ahora a hacer una exploración del learning rate y la regularización

Parada('Estudio sobre el learning rate (tarda medio minuto aprox)')
# Parámetros por defecto
MLP_2 = MLPRegressor(
    random_state=1,
    max_iter=200,
    shuffle = True,
    early_stopping = False,
    activation = 'logistic',
    hidden_layer_sizes = (100, 50),
    solver = 'adam'
    
)

# valores con que experimentar 
learnig_rates = [0.0001, 0.001, 0.01, 0.1, 1]

parametros = {
    'learning_rate_init':learnig_rates

}


resultados_2 = MuestraResultadosVC(
    MLP_2,
    parametros,
    x_train,
    y_train
    
)


Parada( 'Muestro gráfico ')
print('Gráfico que muestra la evolución del coefiente de terminación frente la tasa de aprendizaje')
GraficaError(learnig_rates, resultados_2, 'tasa de aprendizaje')


## Experimento sobre el método de adaptación

Parada ('Experimento variando regularización')


# valores con que experimentar
MLP_3 = MLPRegressor(
    random_state=1,
    max_iter=200,
    shuffle = True,
    early_stopping = False,
    activation = 'logistic',
    hidden_layer_sizes = (100, 50),
    solver = 'adam'
    
)
regularizacion = [0,0.0001,0.001,0.01, 1]

parametros_3 = {
    'alpha': regularizacion

}

resultados_3 = MuestraResultadosVC(
    MLP_3,
    parametros_3,
    x_train,
    y_train
    
)

Parada( 'Muestro gráfico comparación regularización ')
print('Grafica que compara la evolución de R^2 frente a la regularización')
GraficaError( regularizacion, resultados_3, 'regularización')

# -------------------------------------

    
Parada('Experimento de número de iteraciones ')


# valores con que experimentar
MLP_4 = MLPRegressor(
    random_state=1,
    shuffle = True,
    early_stopping = False,
    activation = 'logistic',
    hidden_layer_sizes = (100, 50),
    solver = 'adam',
    alpha = 0.01
    
)
maximas_iteraciones = [10, 50, 100, 200, 350]

parametros_4 = {
    'max_iter':maximas_iteraciones

}
   
resultados_4 = MuestraResultadosVC(
    MLP_4,
    parametros_4,
    x_train,
    y_train
    
)
'''# DESCOMENTAR
''' Comento porque tiene poco sentido mostrar esta gráfica
Parada( 'Muestro gráfico comparación número de iteraciones')
GraficaError( maximas_iteraciones, resultados_4)
'''
'''# DESCOMENTAR
# ------- tras todo esto el modelo seleccionado por cross validation es ---

MLP_mejor = MLPRegressor(
    random_state=1,
    shuffle = True,
    early_stopping = False,
    activation = 'logistic',
    hidden_layer_sizes = (100, 50),
    solver = 'adam',
    alpha = 0.01,
    learning_rate_init = 0.001
    
)

# añadir función de evaluación de errores   

Parada( 'MEJOR RESULTADO PARA MLP')
'''# DESCOMENTAR
print('''Los hiperparámetros seleccionados han sido:  
    random_state=1,
    shuffle = True,
    early_stopping = False,
    activation = 'logistic',
    hidden_layer_sizes = (100, 50),
    solver = 'adam',
    alpha = 0.01,
    learning_rate_init = 0.001
    
''')


'''# DESCOMENTAR
ConclusionesFinales( MLP_mejor,
                     x_train,
                     y_train,
                     x_test,
                     y_test,
                     mostrar_coeficientes = False  #importante, porque AdaBoos no tiene esta función y daría error
                    )





###########################################################
# DUMMY
###########################################################

Parada('Evaluación de croos validation para dummy')
dummy_regr = DummyRegressor()
parametros = {'strategy':['mean', 'median']}

MuestraResultadosVC( dummy_regr, parametros, x_train, y_train)

'''# DESCOMENTAR

''' Tarda un par de mínutos en ejecutarse, ya que calculamos 
el EOUT a partir de leaveCincuentaOut (una versión de leaveOneOut pero con 50)
Parada('Evaluación del modelo final seleccionado')
print('\n Evaluación final del del modelo ingenuo ')
ConclusionesFinales( MODELO_RANDOM_FOREST,
                     x_train, y_train,
                     x_test, y_test,
                     mostrar_coeficientes = False,
                     leave_one_out = True
                    )

'''
#######################################################
#
                
                
