'''
Boosting 

'''
from main import *
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from numpy.ma import getdata # para rescatar el orden que devuelve eso


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


'''xf
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
boostingRegresion =  AdaBoostRegressor()

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

##### ______- aumentamso la dimensión del espacio de búsqueda  _________

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

    MuestraResultadosVC( boostingRegresion,
                         parametros_seleccionados,
                         x_polinomio,
                         y_train_con_outliers
                        )
    


