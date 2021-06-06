'''
PRÁCTICA 3 Regresión 
Blanca Cano Camarero   
'''
#############################
#######  BIBLIOTECAS  #######
#############################
# Biblioteca lectura de datos
# ==========================
import pandas as pd

# matemáticas
# ==========================
import numpy as np

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

########## CONSTANTES #########
NOMBRE_FICHEROS_REGRESION  = ['./datos/train.csv','./datos/unique_m.csv'] # solo se va a leer el primero 
SEPARADOR_REGRESION = ','

NUMERO_CPUS_PARALELO = 4
####################################################
################### funciones auxiliares 
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
    data = np.genfromtxt(nombre_fichero,delimiter=separador)
    y = data[1:,-1].copy()
    x = data[1:,:-1].copy()

    return x,y


def VisualizarClasificacion2D(x,y, titulo=None):
    """Representa conjunto de puntos 2D clasificados.
    Argumentos posicionales:
    - x: Coordenadas 2D de los puntos
    - y: Etiquetas"""

    _, ax = plt.subplots()
    
    # Establece límites
    xmin, xmax = np.min(x[:, 0]), np.max(x[:, 0])
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)

    # Pinta puntos
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", alpha=0.8)

    # Pinta etiquetas
    etiquetas = np.unique(y)
    for etiqueta in etiquetas:
        centroid = np.mean(x[y == etiqueta], axis=0)
        ax.annotate(int(etiqueta),
                    centroid,
                    size=14,
                    weight="bold",
                    color="white",
                    backgroundcolor="black")

    # Muestra título
    if titulo is not None:
        plt.title(titulo)
    plt.show()


def Separador(mensaje = None):
    '''
    Hace parada del código y muestra un menaje en tal caso 
    '''
    #print('\n-------- fin apartado, enter para continuar -------\n')
    input('\n-------- fin apartado, enter para continuar -------\n')
    
    if mensaje:
        print('\n' + mensaje)





#######################################################################
#######################################################################
#######################################################################


# Lectura de los datos

print(f'Vamos a proceder a leer los datos de los ficheros {NOMBRE_FICHEROS_REGRESION}')


x,y = LeerDatos( NOMBRE_FICHEROS_REGRESION[0], SEPARADOR_REGRESION)

# Comprobación de si hay algún valor perdido
print('Hay algún valor perdido:',np.isnan(x).any())


Separador('Separamos test y entrenamiento')

###### separación test y entrenamiento  #####
ratio_test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size= ratio_test_size,
    shuffle = True, 
    random_state=1)


#vemos que están balanceados

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
    

    # gráfico  de valores
    plt.title('Número de etiquetas por rango de valores')
    plt.bar([i*longitud + min_y for i in range(len(datos_en_rango))],
            datos_en_rango, width = longitud * 0.9)
    plt.xlabel('Valor de la etiqueta y (rango de longitud %.3f)'%longitud)
    plt.ylabel('Número de etiquetas')
    plt.show()
    

    
        
### Comprobación de balanceo 
Separador('___ Distribución de las etiquetas ___')
BalanceadoRegresion(y, divisiones = 30)

restricciones_y = [100, 140]
for restriccion_y in restricciones_y: 
    Separador(f'Veamos para datos que cumplan y>{restriccion_y}')
    BalanceadoRegresion(y[y>restriccion_y], 30)


## Quitamos outliers

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

#########




Separador('Normalizamos los datos')
#Normalizamos los datos para que tengan media 0 y varianza 1
scaler = StandardScaler()
x_train = scaler.fit_transform( x_train )
x_test = scaler.transform( x_test) 

scaler_outliers = StandardScaler()
x_train_outliers_normalizado = scaler_outliers.fit_transform( x_train_con_outliers )
    
#------- correlacion ----
def PlotMatrizCorrelacion(matriz_correlacion):
    '''
    Muestra en pantalla la matriz de correlación de x
    usa la biblioteca de seaborn 
    '''
    plt.figure(figsize=(12,8))
    plt.title('Matriz de correlación')
    sns.heatmap(matriz_correlacion)
    plt.show()


    
def Pearson( x, umbral, traza = False):
    '''INPUT 
    x vector de caracteríscas 
    umbral: valor mínimo del coefiente para ser tenido en cuenta
    traza: Imprime coeficiente de Pearson e índices que guardan esa relación.   
    muestra gráfico: determina si se muestra una tabla con los coeficinetes que superen el umbral

    OUTPUT
    indice_explicativo: índice de columnas linealmente independientes (con ese coeficiente)
    relaciones: lista de tuplas (correlacion, índice 1, índice 2)

    '''
    r = np.corrcoef(x.T)
    longitud_r  = len(r[0])
    # Restamos la matriz identidad con la diagonal
    # Ya que queremos encontrar donde alcanza buenos niveles de correlación no triviales 
    sin_diagonal = r - np.identity(len(r[0])) 
    relaciones = [] # guardaremos tupla y 


    # Devolveré un vector con lo índices que no puedan ser explicado,
    # Esto es, si existe una correlación mayor que el umbra entre i,j
    # y I es el conjunto de características el nuevo I = I -{j}
    # Denotarelos a I con la variable índice explicativo 
    indice_explicativo = np.arange( len(x[0]))


    # muestra tupla con el coefiente de pearson y los dos índices con ese vector de características
    for i in range(longitud_r):
        for j in range(i+1, longitud_r):
            if abs(sin_diagonal[i,j]) > umbral:
            
                relaciones.append((sin_diagonal[i,j], i,j))
                #print(sin_diagonal[i,j], i,j)

                indice_explicativo [j] = 0 # Indicamos que la columna j ya no es explicativa
                #para ello la pongo a cero, ya que el 0 siempre explicará, por ir  de menor a mayor los subíndices

    indice_explicativo = np.unique(indice_explicativo) # dejamos solo un cero

    
    relaciones.sort(reverse=True, key =itemgetter(0))

    # imprimimos las relaciones en orden
    if(traza):
        print(f'\nCoeficiente pearson para umbral {umbral}')
        print('| Coeficiente | Índice 1 | Índice 2 |     ')
        print( '| --- | --- | ---|     ')
        for i,j,k in relaciones:
            print('| ',i,' | ' , j, ' | ', k , '|     ')

    return indice_explicativo, relaciones


Separador('Matriz de correlación asociada a los datos de entrenamiento')
PlotMatrizCorrelacion(np.corrcoef(x_train.T))


Separador('Índice de las características a mantener')

### Cálculos para distinto umbrales
umbrales = [0.999, 0.99, 0.98, 0.97, 0.95, 0.9] 
indice_explicativo = dict()
relaciones = dict()

for umbral in umbrales:
    indice_explicativo[umbral], relaciones[umbral] = Pearson( x_train,
                                                              umbral,
                                                              traza = True,
                                                            )
    Separador()
numero_caracteristicas = len(x_train[0])
print(f'\nEl número inical de características es de { numero_caracteristicas}\n' )
print('Las reducciones de dimensión total son: \n')
print('| umbral | tamaño tras reducción | reducción total |    ')
print('|:------:|:---------------------:|:---------------:|    ')
for  umbral, ie in indice_explicativo.items():
    len_ie = len(ie)
    print(f'| {umbral} | {len_ie} | {numero_caracteristicas - len_ie} |    ')
    



    
umbral_seleccionado = 0.97 # debe de estar definidio en la lista umbrales  

def ReducirCaracteristicas(x,indices_representativos):
    '''
    x vector características
    indices_representativos: índices características que mantener

    OUTPUT 
    x_reducido 
    '''
    x_reducido = (x.T[indices_representativos]).T

    return x_reducido


## Reducimos los datos  por regresión    
x_train_reducido = ReducirCaracteristicas(x_train, indice_explicativo[ umbral_seleccionado])
x_test_reducido = ReducirCaracteristicas(x_test, indice_explicativo[ umbral_seleccionado])
n_x_test_reducido =  len(x_test_reducido[0])

##  reducimos datos por PCA
Separador('PCA y su máxima verosimilitud logaritmica')

n_x_train = len(x_train[0]) #número de características del x_train 

numero_componentes = [1,2, n_x_test_reducido//2,
                      int(n_x_test_reducido* 3/4),
                      int(n_x_train * 9/10),
                      n_x_test_reducido,
                      n_x_train]

pca_sin_pearson = dict() # tomamos x_train
x_train_pca_sin_pearson = dict()
pca_con_pearson = dict() # tomamos x_train_reducido
x_train_pca_con_pearson = dict()

print('| N componentes | score sin haber reducido | score habiendo reducido |   ')
print('|:---:'*3 + '|    ')
for n_componentes in numero_componentes:

    pca_sin_pearson[n_componentes] = PCA(n_components = n_componentes)
    pca_sin_pearson[n_componentes].fit(x_train)
    x_train_pca_sin_pearson[n_componentes] = pca_sin_pearson[n_componentes].transform(x_train)
    score_sin_pearson = pca_sin_pearson[n_componentes].score(x_train)
    
    if(n_componentes <= n_x_test_reducido):
        pca_con_pearson[n_componentes] = PCA(n_components = n_componentes)
        pca_con_pearson[n_componentes].fit(x_train_reducido)
        x_train_pca_con_pearson[n_componentes] = pca_con_pearson[n_componentes].fit_transform(x_train_reducido)
        score_con_pearson = pca_con_pearson[n_componentes].score(x_train_reducido)
    else:
        score_con_person = 'no calculable'

    print('| {}|{:.4} |{:.4}|    '.format(
        n_componentes, score_sin_pearson, score_con_pearson))
        

Separador('Visualizamos PCA 2d')

plt.title('PCA sin haber reducido antes con Pearson')
plt.scatter(x_train_pca_sin_pearson[1], y_train)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


### Validación cruzada
def MostrarMatrizConfusion(clasificador, x, y, titulo, normalizar):
    '''
    normalizar: 'true' o 'false', deben de ser los valores de normalice en mostrar_plot
    '''
    
    mostrar_plot = plot_confusion_matrix(clasificador,
                                         x , y,
                                         normalize = normalizar)
    mostrar_plot.ax_.set_title(titulo)
    plt.show()


def Evaluacion2( clasificador,
                x, y, x_test, y_test,
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
    - x_test, y_test
    - k-folds: número de particiones para la validación cruzada
    - metrica_error: debe estar en el formato sklearn (https://scikit-learn.org/stable/modules/model_evaluation.html)

    OUTPUT:
    '''

    ###### constantes a ajustar
    numero_trabajos_paralelos_en_validacion_cruzada = NUMERO_CPUS_PARALELO
    ##########################
    
    print('\n','-'*20)
    print (f' Evaluando {nombre_modelo}')
    #print('-'*20)

    
    #print(f'\n------ Ajustando modelo------\n')        
    tiempo_inicio_ajuste = time.time()
    
    #ajustamos modelo 
    ajuste = clasificador.fit(x,y) 
    tiempo_fin_ajuste = time.time()

    tiempo_ajuste = tiempo_fin_ajuste - tiempo_inicio_ajuste
    print(f'Tiempo empleado para el ajuste: {tiempo_ajuste}s')

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
    print(f'Tiempo empleado para el validacion cruzada: {tiempo_validacion_cruzada}s')

    
    '''
    print('score_validacion_cruzada')
    print(score_validacion_cruzada)
    print (f'Media error de validación cruzada {score_validacion_cruzada.mean()}')
    print(f'Varianza del error de validación cruzada: {score_validacion_cruzada.std()}')
 
    print(f'Ein_train {ajuste.score(x,y)}')
    
    print('______Test____')
    print(f'En_test {metrica_error} { ajuste.score(x_test, y_test)}' )
   '''

    return ajuste



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
    print ('\tMedia error de validación cruzada {:.5f} '.format(score_validacion_cruzada.mean()))
    print('\tDesviación típica del error de validación cruzada {:.5f} '.format(score_validacion_cruzada.std()))
    print('\tTiempo empleado para el ajuste: {:.4f}s '.format(tiempo_ajuste))
    print('\tTiempo empleado para el validacion cruzada {:.4f}s'.format(tiempo_validacion_cruzada))



    return ajuste



############################################################
############ EVALUACIÓN DE LOS MODELOS #####################
############################################################


ITERACION_MAXIMAS = 2000
# ¿sería interesante ver la variabilidad con los folds ?
k_folds = 5 # valor debe de estar entre 5 y 10


#_________ regresión lineal __________
Separador('____ Regresión lineal______')

## experimentos con PCA, reducción pearon y transformaciones

LINEAL_REGRESSION = LinearRegression(normalize = False, 
                    n_jobs = NUMERO_CPUS_PARALELO)

regresion_lineal_con_outliers = Evaluacion(  LINEAL_REGRESSION,
                                x_train_con_outliers,
                                y_train_con_outliers,
                                k_folds,
                                'Regresión lineal con datos sin preprocesar',
                                metrica_error  = 'r2'
                              )

regresion_lineal_con_outliers_normalizados =Evaluacion(
    LINEAL_REGRESSION,
    x_train_outliers_normalizado,
    y_train_con_outliers,
    k_folds,
    'Regresión lineal con datos normalizados sin preprocesar, normalizados',
    metrica_error  = 'r2'
)

regresion_lineal_sin_outliers_normalizados =Evaluacion(
    LINEAL_REGRESSION,
    x_train,
    y_train,
    k_folds,
    'Regresión lineal con datos normalizados y sin outliers',
    metrica_error  = 'r2'
)
regresion_lineal_pearson = Evaluacion(  LINEAL_REGRESSION,
                                x_train_reducido,
                                        y_train,
                                k_folds,
                                'Regresión lineal x reducida por Pearson',
                                metrica_error  = 'r2'
                              )
numero_componentes_pca = [int(n_x_train * 9/10), n_x_test_reducido] # deben de estar definidas en `numero_componentes`
regresion_lineal_mejor_pca = dict()

for n_pca in numero_componentes_pca :
    regresion_lineal_mejor_pca = Evaluacion(
        LINEAL_REGRESSION,
        x_train_pca_sin_pearson[n_pca],
        y_train,                                
        k_folds,
        f'Regresión lineal, PCA n={n_pca} sin reducción por pearson',
        metrica_error  = 'r2'
    )

# ________ experimentos transformación de los datos  ____________
# Probamos con una transformación cuadrática

Separador('Regresión lineal transformación cuadrática')

def TransformacionPolinomica( grado,x):
    x_aux = np.copy(x)
    for i in range(1,grado):
        x_aux = x_aux*x_aux
        x = np.c_[x_aux, x]
    return x


x_inversa = np.array([ [1/x for x in fila]
                       for fila in x_train
])

regresion_lineal_inversa = Evaluacion(
    LINEAL_REGRESSION,
    np.c_[x_inversa, x_train],
    y_train,
    k_folds,
    'Regresión lineal transformación en inversa',
    metrica_error  = 'r2'
)
grado = 2
regresion_lineal_p2 = Evaluacion(
    LINEAL_REGRESSION,
    TransformacionPolinomica( grado, x_train_reducido),
    y_train,
    k_folds,
    'Regresión lineal transformación lineal cuadrática',
    metrica_error  = 'r2'
)
regresion_lineal_sin_transformacion = Evaluacion(
    LINEAL_REGRESSION,
    x_train,
    y_train,
    k_folds,
    'Regresión lineal sin transformaciones',
    metrica_error  = 'r2'
)
# No hay mejora considerable, descartamos el método de transformar las variables

#_______ Experimentos coeficientes  _________ 

Separador('Coeficientes')

def EvaluacionCoeficientes( w, title, mostrar_coeficientes = False):
    print(title)
    print(x_train )

    if(mostrar_coeficientes):
        print(w)

    print('Media de los coeficientes {:.4f}'.format(w.mean()))
    print('Desviación típicade los coeficientes {:.4f}'.format(w.std()))
    print('Rango valores de los coeffientes [{:.4f} , {:.4f}]'.format(
        min(w), max(w)))
    
    
EvaluacionCoeficientes(
    regresion_lineal_sin_transformacion.coef_,
    'Coeficientes del ajuste lineal para regresión lineal sin regularizar',
    mostrar_coeficientes = True)


EvaluacionCoeficientes(
    regresion_lineal_p2.coef_,
    'Coeficientes del ajuste final para regresión lineal con transformacíon cuadrática')



## Número máximo de iteraciones



#NUMERO_MAXIMO_ITERACIONES = 10000

# Evaluación tabla

def EvaluacionTablaCoeficientes ( clasificador,
                x, y, 
                k_folds,
                nombre_modelo,
                metrica_error):

    ###### constantes a ajustar
    numero_trabajos_paralelos_en_validacion_cruzada = NUMERO_CPUS_PARALELO
    ##########################
    
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

    w = ajuste.coef_
    #| Modelo               | Media $R^2$ cv | Desviación típica cv | Media coeficientes | Desv. coef | Intervalo coeficientes   | t.ajuste | t t vc  |
    print('|{:s} |{:.5f} |{:.5f} |{:.5f} |{:.5f} | $[{:.3f},{:.3f}]$ | {:.3f} | {:.3f} |      '.format(
        nombre_modelo,
        score_validacion_cruzada.mean(),
        score_validacion_cruzada.std(),
        w.mean(),
        w.std(),
        min(w), max(w),
        tiempo_ajuste,
        tiempo_validacion_cruzada
              
    ))

    return ajuste
    



##_________ método Ridge y Lasso______
Separador('Comparativas regularización entre Lasso y Ridge')

# COMENTARIO PARA AHORAR TIEMPO DE EJECUCIÓN
#alphas = [0.0001, 0.01, 1, 100]
alphas = [ 0.01]

print('ATENCIÓN: En la memoria se ha calculado para alpha tomando los valores : ',
      '[0.0001, 0.01, 1, 100]',
      '\nPara esta ejecución se ha utilizado ', alphas ,
      '\npara agilizar la ejecución')

RIDGE = dict() # clasificadores 
ridge = dict() # ajustes

LASSO = dict() # clasificadores 
lasso = dict() # ajustes

print(' | Modelo               | Media $R^2$ cv | Desviación típica cv | Media coeficientes | Desv. coef | Intervalo coeficientes   | t.ajuste | tiempo vc |    ')
print('|:--------------------:|:--------------:|:--------------------:|:------------------:|:----------:|--------------------------|----------|:---------:|     ')
for a in alphas:
    #Separador(f'Ridge alpha = {a}')
    RIDGE[a] = Ridge(alpha = a,
                  max_iter = ITERACION_MAXIMAS,
                  
                  )


    ridge[a] =  EvaluacionTablaCoeficientes(  RIDGE[a],
                          x_train, y_train,
                          k_folds,
                          f'Ridge alpha = {a}',
                          metrica_error  = 'r2'
                      
                        )

    LASSO[a] = Lasso(alpha = a,
                  max_iter = ITERACION_MAXIMAS,
                  
                  )
    

    lasso[a] =  EvaluacionTablaCoeficientes(LASSO[a],
                          x_train, y_train,
                          k_folds,
                          f'Lasso alpha = {a}',
                          metrica_error  = 'r2'                 
                        )

    
# La variación es muy poca, y el error en cross validation se mantien, luego descartamso esta opción


#tenemso los datos sufiecientes para aplicar 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

'''
SUPPORT_VECTOR_REGRESSION = SVR(C=1.0, epsilon=0.2)
grado = 1
svr = Evaluacion(
    SUPPORT_VECTOR_REGRESSION,
    TransformacionPolinomica( grado, x_train_reducido),
    y_train,
    TransformacionPolinomica( grado, x_test_reducido,),
    y_test,
    k_folds,
    'Suppot vector regression',
    metrica_error  = 'r2'
    #metrica_error  = 'neg_mean_squared_error'
)


# datos obtenidos

 Evaluando Regresión lineal transformación lineal cuadrática
Tiempo empleado para el ajuste: 37.32402229309082s
score_validacion_cruzada
[0.32158177 0.25984677 0.31485914 0.28903334 0.32404925]
Media error de validación cruzada 0.3018740548850907
Varianza del error de validación cruzada: 0.02441279456279483
Ein_train 0.3148199169185655
______Test____
En_test r2 0.29736389886545533

Conclusión: no merece la pena

'''

### ______ sgd regresor ________
Separador('Otros modelos sgd')

print()
''' Datos usados en experimentos, dejamos los más significativos
algoritmos = ['squared_loss', 'epsilon_insensitive']
penalizaciones = ['l2'] 
tasa_aprendizaje = ['optimal', 'adaptive']
alphas = [0.001, 0.0001, 1]
eta = 0.0001
'''
algoritmos = [ 'epsilon_insensitive']
penalizaciones = ['l2'] 
tasa_aprendizaje = ['optimal']
alphas = [1]
eta = 0.0001

''' para gráfica memoria
print(' | Modelo               | Media $R^2$ cv | Desviación típica cv | Media coeficientes | Desv. coef | Intervalo coeficientes   | t.ajuste | tiempo vc |    ')
print('|:--------------------:|:--------------:|:--------------------:|:------------------:|:----------:|--------------------------|----------|:---------:|     ')
'''
cnt = 0 # contado de número de algoritmos lanzados
ajustes = list()

for a in alphas:
    for algoritmo in algoritmos:
        for penalizacion in penalizaciones:
            for aprendizaje in tasa_aprendizaje:
                Separador()
                
                SGD_REGRESSOR = SGDRegressor(
                    alpha = a,
                    max_iter = ITERACION_MAXIMAS,
                    eta0 = eta,
                    learning_rate = aprendizaje,
                    penalty = penalizacion,
                    loss = algoritmo,
                    shuffle = True,
                    early_stopping = True
                )

                
                titulo = str(
                    f'\n___SGD regresión ({cnt})___\n' +
                    'algoritmo: ' + algoritmo  + '\n' +
                    'penalización: '+ penalizacion  + '\n' +
                    'aprendizaje: ' +  aprendizaje + '\n' +
                    'eta: ' + str(eta) +  '\n' +
                    'alpha: ' + str(a) + '\n'
                )
                   

                sgd =  Evaluacion(  SGD_REGRESSOR,
                                    x_train, y_train,
                                    k_folds,
                                    titulo,
                                    metrica_error  = 'r2'
                      
                                  )
                ''' PARA DIBUJAR LA GRÁFICA DE LA MEMORIA
                titulo = str(
                    f'SGD {cnt}, ' + algoritmo  + ', ' + aprendizaje +', ' +
                    'a=' + str(a) 
                )
                
                sgd =  EvaluacionTablaCoeficientes(  SGD_REGRESSOR,
                                      x_train, y_train,
                                      k_folds,
                                      titulo,
                                      metrica_error  = 'r2'
                      
                                  )
                '''

                ajustes.append(sgd)
                cnt += 1


# _______- conclusiones finales ________

Separador('Datos mejor ajuste')
# mejor ajuste

print('El score en test de regresion_lineal_sin_outliers_normalizados es {:.5f}'.format(regresion_lineal_sin_outliers_normalizados.score(x_test, y_test) ))

print('Su error dentro de la muestra era de {:.5f}'.format(regresion_lineal_sin_outliers_normalizados.score(x_train, y_train) ))

Separador('Vamos a entrenar ahora con todos los datos y devolver los coeficientes')

x = np.append(x_train, x_test).reshape(len(x_train) + len(x_test),
                                       len(x_train[0]))
y = np.append(y_train, y_test)


regresion_lineal_sin_outliers_normalizados.fit(x,y)

print('Su error dentro de la muestra era de {:.5f}'.format(regresion_lineal_sin_outliers_normalizados.score(x, y) ))

print('Los coeficientes finales son: \n',
      regresion_lineal_sin_outliers_normalizados.coef_)

# _______ matriz de confusión _____

Separador('Vamos a comparar la predicción usando la matris de confusión')
y_predecida = regresion_lineal_sin_outliers_normalizados.predict(x)

escalas = [1,1/5,1/20]
for escala in escalas:
    Separador('Matriz de confusión escala {:.2f}'.format(escala))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(  confusion_matrix( (y_predecida*escala).round(),(y*escala).round()))
    plt.title('Matriz de confusión escala {:.2f}'.format(escala))
    fig.colorbar(cax)
    plt.xlabel('Predición')
    plt.ylabel('Valor real')
    plt.show()




 ## Comparación con un aproximador dummy   
