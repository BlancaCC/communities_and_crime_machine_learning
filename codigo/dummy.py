'''
Estimadores Dummy   
'''


from sklearn.dummy import DummyRegressor


from main import *

Parada('Evaluación de croos validation para dummy')
dummy_regr = DummyRegressor()
parametros = {'strategy':['mean', 'median']}

MuestraResultadosVC( dummy_regr, parametros, x_train, y_train)


Print('\n Evaluación del modelo ingenuo ')
ConclusionesFinales( dummy_regr,
                         x_train, y_train,
                         x_test, y_test,
                         mostrar_coeficientes = False)
