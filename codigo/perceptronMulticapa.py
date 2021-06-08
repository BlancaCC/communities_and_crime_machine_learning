'''
Este fichero se unirá a main, recordar inluir las bibliotecas
respectivas en el main   
'''

from main import *
from sklearn.neural_network import MLPRegressor
# x_train y y_train son las variables que nos interesan que tenemos   

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html


k_folds = 5
metrica_error = 'r2'

TAMANOS_CAPAS = [100]#[50, 75, 100]
METODOS = ['adam']#['adam', 'sgd']
ALPHAS = [0.1]#[0.0001, 0.001, 0.1, 1]
LEARNING_RATES = ['constant'] #['constant', 'invscaling', 'adaptive'] # se paran en el mismo error



print('============================================')
print('============================================')
Parada('EXPERIMENTO MLP')
print('============================================')
for tamano_capa in TAMANOS_CAPAS:
    for metodo in METODOS:
        for a in ALPHAS:
            for lr in LEARNING_RATES:
            
                nombre_modelo = 'Perceptrón multicapa tamaño capa {} metodo {} a {}, learning_rete {}'.format(
                    tamano_capa,
                    metodo,
                    a,
                    lr
                )


                MLP = MLPRegressor(random_state=1,
                                   max_iter=500,
                                   shuffle = True,
                                   solver = metodo,
                                   activation = 'logistic',
                                   hidden_layer_sizes = (tamano_capa),
                                   alpha = a,
                                   #verbose = True,
                                   learning_rate = lr
                                   
                       )


                MLP_ajuste = Evaluacion( MLP,
                                         x_train, y_train, 
                                         k_folds,
                                         nombre_modelo,
                                         metrica_error)



                print('ESCORE EN TEST ', MLP_ajuste.score(x_test, y_test))
                Parada()
