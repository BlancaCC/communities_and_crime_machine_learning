---
header-includes: 
- \usepackage{tikz,pgfplots}   
- \usepackage[spanish,es-tabla]{babel}
- \usepackage[utf8]{inputenc}
- \usepackage{graphicx}
- \usepackage{subcaption}
---

# Perceptrón multicapa regresión  

Para esta implementación vamos a utilizar la función 
```
sklearn.neural_network.MLPRegressor(
hidden_layer_sizes=100, activation='relu', *, solver='adam',
alpha=0.0001, batch_size='auto', learning_rate='constant',
learning_rate_init=0.001, power_t=0.5, max_iter=200, 
shuffle=True, random_state=None, tol=0.0001,
verbose=False, warm_start=False, momentum=0.9,
nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
beta_1=0.9, beta_2=0.999, epsilon=1e-08,
n_iter_no_change=10, max_fun=15000)  
```


de la biblioteca de `sklearn` 

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html  

(Añadir enlace a la bibliografía más adelantes )

Además utilizaremos los siguientes argumentos:  

- `hidden_layer_sizes` número de unidades por capa en el rango 50-100, que afinaremos por validación cruzada.  
- `activation`: `logistic` la función de activación logística NO TENGO ARGUMENTO PARA ELEGIR ESTA U OTRA.  
- `solver`  la técnica para minimizar `adam` ya que según la documentación este método es el que funciona mejor con miles datos como es nuestro caso. 
- `alpha` método de regularización.  
- `learning_rate: {'constant', 'invscaling', 'adaptative'}`.  
- `learning_rate_init` aquí si hay que utilizarl 

## Explicación del método de minimización de `adam`  

Bibliografía: 
Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).  

Es un método basado en la optimización de gradiente descendiente.  Requiere de gradeinte de primer orden.   

Las ventajas que supone son  

```
Our method is designed to combine the advantagesof two recently popular methods: 
AdaGrad (Duchi et al., 2011), which works well with sparse gra-dients, and RMSProp (Tieleman & Hinton, 2012),
which works well in on-line and non-stationarysettings; important connections to 
these and other stochastic optimization methods are clarified insection 5.
Some of Adam’s advantages are that the magnitudes of parameter updates are
invariant torescaling of the gradient, its stepsizes are approximately bounded 
by the stepsize hyperparameter,it does not require a stationary objective, it works with 
sparse gradients, and it naturally performs aform of step size annealing.
```  

TODO : redactar mejor

Además com heurística en la propia documentación del sklearn se recomendaba para tamños de entrenamiento de miles.  


El algoritmo indicado en el artículo de 2015 donde se publicó es el siguiente:    

<!-- Lo siento Alex, no he sido capaz de copiarlo, qué pereza -->  

\begin{figure}[!h]
\centering
\includegraphics[width=1\textwidth]{./imagenes/MLP/algoritmo_adam.png}
\caption{Descripción del algoritmo de minimización estocástico de Adam}
\end{figure}.   



