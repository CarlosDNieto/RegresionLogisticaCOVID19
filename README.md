# Tarea 4: COVID-19 Regresión Logística
---
### Seminario de Estadística
### Semestre 2020-2
### Facultad de Ciencias (UNAM)
#### Carlos David Nieto Loya
#### Liga del projecto en github: https://github.com/CarlosDNieto/RegresionLogisticaCOVID19
---

En este projecto hay 3 archivos principales:

* **tarea4_final.py:** Archivo .py que corre un modelo de regresión logística para predecir (a partir de las características físicas y médicas de una persona) si una persona va a ser hospitalizada o no, el output de este archivo solo imprime la evaluación del modelo.

* **covid19_notebook.ipynb:**  Es el jupyter notebook donde se hace el análisis exploratorio de datos y se corren dos modelos diferentes, uno con la pruba Chi y otro sin ella.

* **script_nano:** En este archivo .py se encuentra el script que se corrió en el ``spark-submit`` dentro de la máquina maestra del clúster que se inicializó en AWS.

> Los outputs del spark-submit se encuentran dentro de la carpeta ``screenshots/spark-submit_AWS``


También hay carpetas secundarias en este projecto:

* **data:** Carpeta con la base de datos de casos de covid de la CDMX unicamente por problemas de almacenamiento.

* **pckgs:** Dentro de esta carpeta, viene contenido un módulo de python donde guardamos todas nuestras funciones de procesamiento de datos dentro de spark. Éste módulo se llama: **``models.py``**.

* **tests:** En esta carpeta se encuentran algunos modelos de prueba que se corrieron para definir un modelo "ganador".

---
### Disclaimer:
Durante el desarrollo de este proyecto cambié de una computadora con un SO Windows, a una computadora con un SO MacOS. Por esto no me dió tiempo de incluir la funcionalidad de agregar datos en Hive desde el clúster de AWS además de que estuve varias horas seguidas tratando de poner un ambiente virtual para spark adecuadamente y fallé. 

Esto se debe a que hace unos días (18-06-2020) Spark lanzó como versión oficial spark3.0.0 y con los comandos de MacOS para descargar spark, éste se descargaba automáticamente sin dejarme decidir entre la versión 3 y la 2.4.

Otro problema es la versión de mi JVM, tengo la 14, instalé la 8 y aún me seguían apareciendo errores.

Todo esto hizo que el archivo ``tarea4_final.py`` lo tuviera que correr otro compañero en su computadora, es por eso que adjunto la captura de pantalla ``tarea4_final OUTPUT`` como evidencia de que el código corre correctamente dentro de un ambiente adecuado para spark y pyspark.






