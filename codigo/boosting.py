'''
Boosting 

'''
from main import *
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from numpy.ma import getdata # para rescatar el orden que devuelve eso
from math import floor

'''
Del fichero main se han exportado los datos: 

x_train_con_outliers | sin normalizar con outliers  
y_train_con_outliers 

x_train |  normalizados sin outliers
y_train 

x_train_sin_normalizar | si outlieres sin normalizar 
x_test_sin_normalizar 

x_train_outliers_normalizado  | son outliers normalizados
x_test_outliers_normalizado 
'''


'''
boostingRegresion =  AdaBoostClassifier(
    n_estimators=n_estimadores,
    learning_rate = tasa_aprendizaje,
    loss = funcion_perdida, )
'''

Parada('Comenzaremos el cálculo de cross validation, tarda aproximadamente un minuto')
'''
parametros = {
     'n_estimators' :[50,70],#[50, 75, 100, 125],
    'learning_rate' : [0.01]#[0.001, 0.01, 0.1, 1, 1.1]#,
    #'loss' : ('linear')#[funcion_perdida]
    }
'''
parametros = {
     'n_estimators' :[50, 60, 80, 100],
    'learning_rate' : [0.001, 0.01, 0.1, 1]
    }


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
boostingRegresion =  AdaBoostRegressor(random_state = 2)

#DESCOMENTAR
#MuestraResultadosVC( boostingRegresion, parametros, x_train, y_train)


Parada('Cambiamos conjuntos de entrenamiento')


parametros_seleccionados = {
     'n_estimators' :[50, 100],
    'learning_rate' : [0.01, 0.1,]
    }
conjuntos = ['Sin normalizar con outliers',
             'Normalizado con outliers',
             'Sin normalizar sin outliers',
             'Normalizado con outliers'
             ]

x_conjuntos = [
    x_train_con_outliers,
    x_train_outliers_normalizado,
    x_train_sin_normalizar,
    x_train
]

y_conjuntos = [ 
    y_train_con_outliers, y_train_con_outliers,
    y_train,y_train
    ]


for i in range(len(conjuntos)):
    print( f'\n--- {conjuntos[i]}----')
    #DESCOMENTAR
    '''
    MuestraResultadosVC( boostingRegresion,
                         parametros_seleccionados,
                         x_conjuntos[i],
                         y_conjuntos[i]
                        )
    '''
#####_____Comprobamos si existe sobre ajuste con el número de estimadores ____
Parada('Experimentos sobre ajuste en función del número de estimadores')
# reservamos un conjunto de datos de evaluación
x_train_aux, x_eval, y_train_aux, y_eval = train_test_split(
    x_train, y_train,
    test_size= 0.15,
    shuffle = True, 
    random_state=1)


# Errores
Ein = []
Eval = []

# Cálculos de los errores 
ESTIMADORES = [i for i in range(50, 101, 5)]

print('| Nº estimadores | $E_{in}$ | $E_{eval}$|     ')
print('|---'*3, '|     ')

for n_estimadores in ESTIMADORES:
    boosting =  AdaBoostRegressor(
        n_estimators=n_estimadores,
        learning_rate = 0.1,
        random_state=1,
        #shuffle = True
    )
    boosting.fit(x_train_aux, y_train_aux)
    Ein.append(boosting.score(x_train_aux, y_train_aux))
    Eval.append(boosting.score(x_eval, y_eval))
    print('| {} | {:.4f} | {:0.4f}|     '.format
          (
              n_estimadores,
              Ein[-1],
              Eval[-1]
          )
    )
Parada('Mostramos gráfico de la evolución de de Ein y Eval')

GraficaComparativaEinEval( ESTIMADORES, Ein,Eval, 'nº estimadores')
    


##### ______ aumentamos la dimensión del espacio de búsqueda  _________

Parada(' Transformación de los datos ')

parametros_seleccionados = {
     'n_estimators' :[50, 100], # estos valores por el tiempo
    'learning_rate' : [ 0.1, 0.01]
    }
grados = [1,2,3]

for g in grados:
    print(f'\n --- Validación cruzada para grado {g} --- ')
    x_polinomio = TransformacionPolinomica(
        g,
        x_train_outliers_normalizado)

    ''' DESCOMENTAR
    MuestraResultadosVC( boostingRegresion,
                         parametros_seleccionados,
                         x_polinomio,
                         y_train_con_outliers
                        )
    '''
    





## _______ comprobación tamaños del conjunto de entrenamiento  ______

#### modelo seleccionado
Parada('Evolución de los errores en función del tamaño de entrenamiento')

boosting_regresion_1 =  AdaBoostRegressor(
    n_estimators = 50,
    learning_rate = 0.1,
    random_state = 2
)



EvolucionDatosEntrenamiento(boosting_regresion_1,
                            x_train_con_outliers,
                            y_train_con_outliers,
                            numero_particiones = 20,
                            porcentaje_validacion = 0.2)


