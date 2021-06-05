# Communies and crime   
Machine learning problem. Try to predict crime from FBI data. 


Este es el trabajo final de la asignatura de aprendizaje automático, impartida 
durante el curso 2020-2021 en la universidad de Granada.  


## Descripción del problema  

(Copiar pegar de la memoria cuando la tengamos)  

## Metodología de trabajo (información para Alex <3)  

(En los epígrafes sucesivos te voy explicando las cosas, no te asustes y no pienses que 
soy absolutamente repelente porfiiiii :( . Que  esto lo hago pa facilitarnos el trabajo aunque no lo parezca :smile:   )

1. El trabajo realizado no se subirá directamente al repositorio, sino que se actualizará por medio de pull request.  
2. Las tareas pendientes tendrán asociadas issues.  
3. Resolver una tarea pendiente es equivalente a cerrar un issue con el commit.  
4. Los commites deberán hacerse atómicos  y de manera periódica, indicando entre `[]` el tema general, por ejemplo 
`[preprocesado] Tratamiento de datos faltantes, algoritmo de Pepe el cuajao`. 


### Cómo realizar pull request.  

Hay varias formas, explico una aprovechando que tienes acceso a este repositorio.  

1. Clonar repositorio. (Esto solo hay que hacerlo una vez :)  )
2. Crear rama de trabajo: `git checkout -b ejemplo-rama-pull` esto te cambiará a esta rama automáticamente. 

Para cambiar de rama basta con hacer `git chekout nombre-rama`, por ejemplo `git chekout main`. 
Otro comando útil es `git status`.
Para borrar rama local `git branch -d nombre-rama`, rama remota `git branch -D nombre-rama`.

3. Añadir,commitear y subir los cambios que hayas realizado (esto es como siempre, `git add .` `git commit` `git push origin rama`) a la rama en que te encuentres.  
4. Desde el navegardor, debajo del nombre del repositorio tienes las opciones de pull-request desde rama.  

Más información sobre los pull-request:
https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request

(Otra opción sería que tuvieras un fork del repositorio)  


## Cómo crear issues  

Desde el navegador, en el repositorio, tienes la opción debajo ( a la izquierda de pull-request).  
Al crearlo se le asocia un número único.  
Para cerrar un issue desde un commit basta con escribir por algún lado del mensaje `close #123` (si el número de issue era #123, si son varios separa por comas por ejemplo `close #1 , close #2`.  


## Cómo descargar contenido   

Utilizar `git rebase nombre-rama`
No utilizar `git pull`

Más información: https://www.atlassian.com/es/git/tutorials/rewriting-history/git-rebase








