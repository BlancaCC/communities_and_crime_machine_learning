# Estimadores dummy   
En el mejor de nuestros modelos hemos obtenido una bondad de ajuste de $R^2 = 0.6$ TODO (REVISAR CUANDO EL MODELO ESTÉ YA ELEGIDO).  

Comparemos cuál es la bondad de este ajuste a partid de estimador que sea *dummy*, es decir un modelo de regresión que utiliza predicción a partir de reglas simples.   

Utilizaremos el de la biblioteca de scikit-learn 


```
class sklearn.dummy.DummyRegressor(*, strategy='mean', constant=None, quantile=None)
```  


Las estrategias que vamos a probar son: 

- `mean` que predice la media.  
- `median`  que predice la mediana.  

Los resultados son: 

Mejores parámetros:  {'strategy': 'mean'}
Con una $R^2$ de:  -0.004209704148944527 

Table:  $R^2$ para estimador dummy   

 | Parámetros      | $R^2$ medio | Desviación tipica $R^2$ | Ranking | tiempo medio ajuste |
 |-----------------|-------------|-------------------------|---------|---------------------|
 | strategy mean   | -0.0042     | 0.0057                  | 1       | 0.0109              |
 | strategy median | -0.1249     | 0.0244                  | 2       | 0.0063              |


Como observamos los datos son considerablemente peores.  

