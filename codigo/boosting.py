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


'''
boostingRegresion =  AdaBoostClassifier(
    n_estimators=n_estimadores,
    learning_rate = tasa_aprendizaje,
    loss = funcion_perdida, )
'''

print('Comenzaremos el cálculo de cross validation, puede tardar unos minutos')
'''
parametros = {
     'n_estimators' :[50,70],#[50, 75, 100, 125],
    'learning_rate' : [0.01]#[0.001, 0.01, 0.1, 1, 1.1]#,
    #'loss' : ('linear')#[funcion_perdida]
    }
'''
parametros = {
     'n_estimators' :[50, 75, 100, 125],
    'learning_rate' : [0.001, 0.01, 0.1, 1, 1.1]
    }


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
boostingRegresion =  AdaBoostRegressor()

MuestraResultadosVC( boostingRegresion, parametros)
