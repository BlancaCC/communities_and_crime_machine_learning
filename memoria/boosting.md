# Boosting   

Se hará un estudio utilizando el modelo de AdaBoost para regresión introducido por Freund y Schapire en 1995.  
TODO Añadir la bibliografía   Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of on-Line Learning and an Application to Boosting", 1995.  

Para ello se utilizará la función 

```python
 class sklearn.ensemble.AdaBoostRegressor(base_estimator=None,
 *, n_estimators=50, learning_rate=1.0,
 loss='linear', random_state=None)
```

Bibliografía: 
-  Teoría: https://scikit-learn.org/stable/modules/ensemble.html#adaboost  
- Implementación: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html  

Los hiperparámetros que ajustaremos y utilizaremos de esta función son los siguientes.   

-  `base_estimator` objeto con el estimador base que utilizar, utilizares `DecisionTreeRegressor` inicialos con profundidad máxima de tres. (TODO, justificar, en el guión se nos indica que utilicemos estos.    
- `learning_rate` haremos un estudio de cómo varía en función del larning rate.  
- `loss` La función de pérdida para actualizar los pesos del boosting en cada iteración.   
- `n_estimators` número de estimadores para el cual el boosting termina, en caso de encontrarse un ajuste perfector pararía antes.   
