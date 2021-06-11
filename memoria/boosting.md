# Boosting   

Se hará un estudio utilizando un modelo de regresión de boosting pro gradiente.  

Para ello se utilizará la función 

```python
class sklearn.ensemble.GradientBoostingRegressor(*, loss='ls', learning_rate=0.1,
n_estimators=100, subsample=1.0,
criterion='friedman_mse', min_samples_split=2,
min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
max_depth=3, min_impurity_decrease=0.0,
min_impurity_split=None, init=None, 
random_state=None, max_features=None, 
alpha=0.9, verbose=0, max_leaf_nodes=None, 
warm_start=False, validation_fraction=0.1,
n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
```  

TODO: Escribir esto como buena bibliografía.  

De la biblioteca de sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html  


Los hiperparámetros que ajustaremos y utilizaremos de esta función son los siguientes.   

- `learning_rate` haremos un estudio de cómo varía en función del larning rate.  
- `loss` utilizaremos `huber` por el estudio preliminar que aparece en machine learnng from theory to algorithm (TODO añadir referencia exacta).  
- `n_estimator` númerod de *boosting stages* (TODO buscar qué significa), según la propia documentación es bastante robusto contra el sobre ajuste, por lo que mayor número genera mejores resultados. Realizaremos un estudio.  
- `subsample` La fracción de ejemplo que se usa para ajustar la base individual (TODO consultar bibliografía).  
- `criterion{‘friedman_mse’, ‘mse’, ‘mae’},`  función que meide la calidad de la partición (TODO consultar la bibliografía).  
- `min_samples_split`  número mínimo de ejemplo para la partición del nodo interno.  

- `min_samples_leaf`. 
