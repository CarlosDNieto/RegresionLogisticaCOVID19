
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (ChiSqSelector, MinMaxScaler, OneHotEncoder, StringIndexer, VectorAssembler)
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import when
from pyspark.ml.classification import LogisticRegression
from requests.exceptions import ConnectionError

warnings.filterwarnings("ignore")

try:
    from pyspark.sql import SparkSession
except:
    import findspark
    findspark.init()
    from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("COVID").config("hive.exec.dynamic.partition", "true").config(
    "hive.exec.dynamic.partition.mode", "nonstrict").enableHiveSupport().getOrCreate()
sc = spark.sparkContext

def guardardatosenHive(df, modo, table="covid.casoscovid"):
    # insertar codigo
    #     df.write.partitionBy("particion").mode("append").format("parquet").saveAsTable(tableNameResult)
    return None


def filtraPositivos(df):
    """
    Esta función filtra los casos positivos de un DataFrame dado.
    df: spark DataFrame con la columna 'resultado'
    """
    # Devolvemos nuestro df con los casos positivos
    return df.filter(df.RESULTADO == 'Positivo SARS-CoV-2')


def limpiaNulls(df):
    """
    Función que limpia nulls en un DF de spark e imprime
    la cantidad de registros que borró (cantidad de valores
    nulos).
    """
    cleanDF = df.na.drop()

    # print("Numero de registros con algun valor nulo: ",
    #       df.count() - cleanDF.count())
    return cleanDF


def dataProcessing(df, categoricalCols, numericalCols, labelCol="TIPO PACIENTE"):
    """Función que hace todo el preprocesamiento de los datos
    categóricos de un conjunto de datos de entrenamiento (o no).
    :param train spark df: conjunto de datos de entrenamiento.
    :param categoricalCols list,array: conjunto de nombres de columnas categoricas del
        conjunto de datos train.
    :param numericalCols list,array: conjunto de nombres de columnas numéricas del 
        conjunto de datos train.
    :param labelCol str: variable objetivo o etiqueta

    :Returns spark dataframe con las columnas 'label' y 'features'
    """

    # codificamos todas las variables categóricas
    stages = []
    for categoricalCol in categoricalCols:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
        encoder = OneHotEncoder(inputCol=stringIndexer.getOutputCol(), outputCol=categoricalCol + "ohe")
        stages += [stringIndexer, encoder]

    # variable objetivo (etiqueta)
    label_strIdx = StringIndexer(inputCol=labelCol, outputCol="label")
    stages += [label_strIdx]

    # ponemos todas las covariables en un vector
    assemblerInputs = [c + "ohe" for c in categoricalCols] + numericalCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="feat")
    stages += [assembler]

    # escala de 0-1
    scala = MinMaxScaler(inputCol="feature", outputCol="features")
    stages += [scala]

    # pipeline donde vamos a hacer todo el proceso
    pipe = Pipeline(stages=stages)
    pipeModel = pipe.fit(df)
    df = pipeModel.transform(df)

    # regresamos nuestro df con lo que necesitamos
    return train


def modeloLogistico(data, labelCol="label", featuresCol="features", weightCol="classWeights"):
    """
    Función que se encarga de ajustar un modelo logístico
    a partir de un dataframe de spark con el esquema ya procesado
    a partir de la función dataProcessing().

    :param data: spark dataframe.
    :param labelCol: string nombre de la columna con la variable respuesta.
    :param featuresCol: string nombre de la columna con los vectores de las
        covariables.

    :returns modelo ajustado:
    """


    model = LogisticRegression(
        featuresCol=featuresCol, labelCol=labelCol, weightCol=weightCol)
    return model.fit(data)

def predictLogistico(test, model):
    """
    Esta función predice un modelo logístico con columnas categóricas
    y numéricas sobre un conjunto de datos de prueba.
    """
    # predecimos el modelo
    predictions = model.transform(test)
    # regresamos el df con las predicciones
    return predictions

# path del archivo csv
file_local_path = '/Users/carlosnieto/Desktop'

# cargamos el archivo en un spark DataFrame
df_origen = spark.read.csv(file_local_path, header=True, encoding='UTF-8', inferSchema=True)

# covariables que vamos a tomar en cuenta para el modelo
features = ["TIPO PACIENTE", "SEXO", "EDAD", "EMBARAZO", "DIABETES", "EPOC", "ASMA", "INMUNOSUPRESION", "HIPERTENSION",
            "OTRA COMPLICACION", "CARDIOVASCULAR", "OBESIDAD", "RENAL CRONICA", "TABAQUISMO", "RESULTADO"]
df = df_origen.select(features)
# filtramos los casos positivos
df = filtraPositivos(df)
# ya no necesitamos la columna resultado
df = df.drop('RESULTADO')
# limpiamos nuestra información
df = limpiaNulls(df)
# creamos una vista temporal en spark
df.createOrReplaceTempView('covid')

# Primero definimos nuestras columnas categóricas, numéricas y nuestra variable objetivo
categoricalCols = ['SEXO', 'EMBARAZO', 'DIABETES', 'EPOC', 'ASMA', 'INMUNOSUPRESION', 'HIPERTENSION', 'OTRA COMPLICACION',
                   'CARDIOVASCULAR', 'OBESIDAD', 'RENAL CRONICA', 'TABAQUISMO']
numericalCols = ["EDAD"]
labelCol = "TIPO PACIENTE"

# Procesamos nuestros datos con la función dataProcessing()
raw_data = dataProcessing(df, categoricalCols, numericalCols, labelCol)

# conjunto de entrenamiento y de prueba
train, test = raw_data.randomSplit([0.70, 0.30])

# obtenemos el balancing ratio
numHosp = train.filter(train["TIPO PACIENTE"] == "HOSPITALIZADO").count()
numAmb = train.filter(train["TIPO PACIENTE"] == "AMBULATORIO").count()
BalancingRatio = numAmb / (numHosp + numAmb)
print("Balancing Ratio: ", BalancingRatio)

# agregamos una columna con el BalancingRatio respectivo para cada label
train = train.withColumn("classWeights", when(
    train.label == 1, BalancingRatio).otherwise(1-BalancingRatio))

# modelo de regresión logística
model = modeloLogistico(data=train, labelCol="label",
                        featuresCol="features", weightCol="classWeights")
# imprimimos los coeficientes
print("Coeficientes: ", str(model.coefficientMatrix))
print("Intercepto: ", str(model.interceptVector))

# predicciones con el conjunto de prueba
predictions = predictLogistico(test, model)

modelSummary = model.summary

roc = modelSummary.roc.toPandas()
plt.plot(roc['FPR'], roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print('Training set areaUnderROC: ' + str(modelSummary.areaUnderROC))
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

pr = modelSummary.pr.toPandas()
plt.plot(pr['recall'], pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

# Matriz de confusión en las predicciones del test set
predictions.crosstab("label", "prediction").show()

# otras métricas de evaluación
print("CLASES:", modelSummary.labels)
print("MEDIDA-F", modelSummary.fMeasureByLabel(beta=1.0))
print("TASA DE FALSOS-POSITIVOS:", modelSummary.falsePositiveRateByLabel)
print("PRECISION: ", modelSummary.precisionByLabel)
print("EXHAUSTIVIDAD: ", modelSummary.recallByLabel)
print("TABLA DE CONFUSION: ")
print(predictions.crosstab("label", "prediction").show())


def pruebaChi(dataframe, categoricalCols, numericalCols, labelCol="TIPO PACIENTE"):
    """Función que hace todo el preprocesamiento de los datos
    categóricos de un conjunto de datos de entrenamiento (o no).
    :param train spark df: conjunto de datos de entrenamiento.
    :param categoricalCols list,array: conjunto de nombres de columnas categoricas del
        conjunto de datos train.
    :param numericalCols list,array: conjunto de nombres de columnas numéricas del 
        conjunto de datos train.
    :param labelCol str: variable objetivo o etiqueta

    :Returns spark dataframe con las columnas 'label' y 'features'
    """

    # codificamos todas las variables categóricas
    stages = []
    for categoricalCol in categoricalCols:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
        encoder = OneHotEncoder(inputCol=stringIndexer.getOutputCol(), 
                                outputCol=categoricalCol + "ohe")
        stages += [stringIndexer, encoder]

    # variable objetivo (etiqueta)
    label_strIdx = StringIndexer(inputCol=labelCol, outputCol="label")
    stages += [label_strIdx]

    # ponemos todas las covariables en un vector
    assemblerInputs = [c + "ohe" for c in categoricalCols] + numericalCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="feat")
    stages += [assembler]

    # seleccionamos las variables que nos sirven con ChiSqSelector
    selector = ChiSqSelector(featuresCol="feat", outputCol="feature", labelCol="label", fpr=0.05,
                             selectorType='fpr')
    stages += [selector]

    # escala de 0-1
    scala = MinMaxScaler(inputCol="feature", outputCol="features")
    stages += [scala]

    # pipeline donde vamos a hacer todo el proceso
    pipe = Pipeline(stages=stages)
    pipeModel = pipe.fit(dataframe)
    df = pipeModel.transform(dataframe)

    # regresamos nuestro df con lo que necesitamos
    return df


def probSujetodeprueba(df):
    # insertar codigo
    pass
    # return probabilidad


def getModeloPersistente(modelo):
    # insertar codigo
    pass
    # return modeloserializado
