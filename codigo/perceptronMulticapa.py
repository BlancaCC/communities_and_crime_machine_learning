'''
Este fichero se unirá a main, recordar inluir las bibliotecas
respectivas en el main   
'''

from main import *
#from randomforest import GraficaError, GraficaRegularizacion
from sklearn.neural_network import MLPRegressor
# x_train y y_train son las variables que nos interesan que tenemos   

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html



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

''' Comento porque tiene poco sentido mostrar esta gráfica
Parada( 'Muestro gráfico comparación número de iteraciones')
GraficaError( maximas_iteraciones, resultados_4)
'''

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



ConclusionesFinales( MLP_mejor,
                     x_train,
                     y_train,
                     x_test,
                     y_test,
                     mostrar_coeficientes = False  #importante, porque AdaBoos no tiene esta función y daría error
                    )


Los valores seleccionados han sido:
    random_state=1,
    shuffle = True,
    early_stopping = False,
    activation = 'logistic',
    hidden_layer_sizes = (100, 50),
    solver = 'adam',
    alpha = 0.01


R^2_in : 0.6437 , R^2_test :0.6055
Tras entrenar con todos los datos: R^2_in : 0.6507 
