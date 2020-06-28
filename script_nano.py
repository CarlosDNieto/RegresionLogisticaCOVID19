try:
    from pyspark.sql import SparkSession
except:
    import findspark
    findspark.init()
    from pyspark.sql import SparkSession

import warnings
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (ChiSqSelector, MinMaxScaler, OneHotEncoder, StringIndexer, VectorAssembler)
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import when
from pyspark.ml.classification import LogisticRegression

warnings.filterwarnings("ignore")

spark = SparkSession.builder.appName("COVID").config("hive.exec.dynamic.partition", "true").config(
    "hive.exec.dynamic.partition.mode", "nonstrict").enableHiveSupport().getOrCreate()
sc = spark.sparkContext

file_path = 's3://carlosnieto.s3.aws.com/dataframe/casos-asociados-a-covid-19-CDMX.csv'

# Columnas que vamos a usar para nuestro modelo:
features = ["TIPO PACIENTE", "SEXO", "EDAD", "EMBARAZO", "DIABETES", "EPOC", "ASMA", "INMUNOSUPRESION", "HIPERTENSION",
            "OTRA COMPLICACION", "CARDIOVASCULAR", "OBESIDAD", "RENAL CRONICA", "TABAQUISMO"]

categoricalCols = ['SEXO', 'EMBARAZO', 'DIABETES', 'EPOC', 'ASMA', 'INMUNOSUPRESION', 'HIPERTENSION', 'OTRA COMPLICACION',
                   'CARDIOVASCULAR', 'OBESIDAD', 'RENAL CRONICA', 'TABAQUISMO']

numericalCols = ["EDAD"]

labelCol = "TIPO PACIENTE"

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

df_origen = spark.read.csv(file_path, header=True, encoding='UTF-8', inferSchema=True)
df = df_origen.select(features)
df = limpiaNulls(df)
df.createOrReplaceTempView('covid')

raw_data = pruebaChi(df, categoricalCols, numericalCols, labelCol)

# conjunto de entrenamiento y de prueba
train, test = raw_data.randomSplit([0.70, 0.30])

numHosp = train.filter(train["TIPO PACIENTE"] == "HOSPITALIZADO").count()
numAmb = train.filter(train["TIPO PACIENTE"] == "AMBULATORIO").count()
BalancingRatio = numAmb / (numHosp + numAmb)

train = train.withColumn("classWeights", when(
    train.label == 1, BalancingRatio).otherwise(1-BalancingRatio))

model = modeloLogistico(data=train, labelCol="label",
                        featuresCol="features", weightCol="classWeights")

modelSummary = model.summary

predictions = predictLogistico(test, model)

evaluator = BinaryClassificationEvaluator()

print("################ EVALUACION DEL MODELO ################")
print('AUROC DEL CONJUNTO DE ENTRENAMIENTO: ' + str(modelSummary.areaUnderROC))
print('AUROC DEL CONJUNTO DE PRUEBA: ', evaluator.evaluate(predictions))
print("CLASES:", modelSummary.labels)
print("MEDIDA-F", modelSummary.fMeasureByLabel(beta=1.0))
print("TASA DE FALSOS-POSITIVOS:", modelSummary.falsePositiveRateByLabel)
print("PRECISION: ", modelSummary.precisionByLabel)
print("EXHAUSTIVIDAD: ", modelSummary.recallByLabel)
print("TABLA DE CONFUSION: ")
print(predictions.crosstab("label", "prediction").show())