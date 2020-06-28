# Modulo de funciones para nuestro modelo
# Carlos David Nieto Loya
# Seminario de Estadística
# 2020-02

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (ChiSqSelector, MinMaxScaler, OneHotEncoder, StringIndexer, VectorAssembler)
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import when
from pyspark.ml.classification import LogisticRegression

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
    scala = MinMaxScaler(inputCol="feat", outputCol="features")
    stages += [scala]

    # pipeline donde vamos a hacer todo el proceso
    pipe = Pipeline(stages=stages)
    pipeModel = pipe.fit(df)
    df = pipeModel.transform(df)

    # regresamos nuestro df con lo que necesitamos
    return df


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