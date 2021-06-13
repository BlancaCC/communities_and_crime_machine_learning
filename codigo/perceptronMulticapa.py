'''
Este fichero se unirá a main, recordar inluir las bibliotecas
respectivas en el main   
'''

from main import *
#from randomforest import GraficaError, GraficaRegularizacion
from sklearn.neural_network import MLPRegressor
# x_train y y_train son las variables que nos interesan que tenemos   

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html


k_folds = 5
metrica_error = 'r2'

TAMANOS_CAPAS = [100]#[50, 75, 100]
METODOS = ['adam']#['adam', 'sgd']
ALPHAS = [0.1]#[0.0001, 0.001, 0.1, 1]
LEARNING_RATES = ['constant'] #['constant', 'invscaling', 'adaptive'] # se paran en el mismo error




Parada(' ____ Estudio preliminar de los parámetros (tarda un minuto aprox)___')
# Parámetros por defecto
'''
MLP_1 = MLPRegressor(
    random_state=1,
    max_iter=500,
    shuffle = True,
    activation = 'logistic',
    radom_state = False
)
'''
tam_capas = [50, 75, 100]

parametros = {
    'hidden_layer_sizes' : [ (i,j) for i in tam_capas for j in tam_capas],
    'solver':['sgd', 'adam']
}

'''  Descomentar    
resultados_1 = MuestraResultadosVC(
    MLP_1,
    parametros,
    x_train,
    y_train
    
)
'''

## De este experimento los mejores resultados han sido
#Mejores parámetros:  {'hidden_layer_sizes': (100, 50), 'solver': 'adam'}
#Con una $R^2$ de:  0.6205065254947334   


# -----------------------------------------------------

# Procedamos ahora a hacer una exploración del learning rate y la regularización

Parada(' ____ Estudio sobre el learning rate (tarda menos de un minuto aprox)___')
# Parámetros por defecto
MLP_2 = MLPRegressor(
    #random_state=1,
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

'''   descomentar 
resultados_2 = MuestraResultadosVC(
    MLP_2,
    parametros,
    x_train,
    y_train
    
)


Parada( 'Muestro gráfico ')

GraficaError(learnig_rates, resultados_2)
'''

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

'''descomentar
resultados_3 = MuestraResultadosVC(
    MLP_3,
    parametros_3,
    x_train,
    y_train
    
)

Parada( 'Muestro gráfico comparación regularización ')

GraficaError( regularizacion, resultados_3)
'''

    
Parada('Experimento de número de iteraciones ')


# valores con que experimentar
MLP_4 = MLPRegressor(
    random_state=1,
    shuffle = True,
    early_stopping = False,
    activation = 'logistic',
    hidden_layer_sizes = (100, 50),
    solver = 'adam'
    
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

Parada( 'Muestro gráfico comparación número de iteraciones')

GraficaError( maximas_iteraciones, resultados_4)


# ------- tras todo esto el modelo seleccionado por cross validation es ---

MLP_mejor = MLPRegressor(
    random_state=1,
    shuffle = True,
    early_stopping = False,
    activation = 'logistic',
    hidden_layer_sizes = (100, 50),
    solver = 'adam',
    alpha = 0.01
    
)

MLP_mejor.fit(x_train, x_test)

# añadir función de evaluación de errores   
