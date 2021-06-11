'''
Boosting 

'''
from main import *
from sklearn.ensemble import AdaBoostClassifier
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



n_estimadores = 100
tasa_aprendizaje = 1
funcion_perdida = 'linear'

'''
boostingRegresion =  AdaBoostClassifier(
    n_estimators=n_estimadores,
    learning_rate = tasa_aprendizaje,
    loss = funcion_perdida, )
'''


parametros = {
     n_estimators :[50, 75, 100, 125],
    learning_rate : [0.001, 0.01, 0.1, 1, 1.1],
    loss : funcion_perdida,
    }
