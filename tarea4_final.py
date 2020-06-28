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

from pckgs.models import (pruebaChi, predictLogistico, limpiaNulls, modeloLogistico, dataProcessing)

warnings.filterwarnings("ignore")

spark = SparkSession.builder.appName("COVID").config("hive.exec.dynamic.partition", "true").config(
    "hive.exec.dynamic.partition.mode", "nonstrict").enableHiveSupport().getOrCreate()
sc = spark.sparkContext

print(spark.version)

# path del archivo csv
file_path = 'data/casos-asociados-a-covid-19-CDMX.csv'
# path del bucket S3 creado en AWS:
#   file_path = 's3://carlosnieto.s3.aws.com/dataframe/casos-asociados-a-covid-19-CDMX.csv'

# Columnas que vamos a usar para nuestro modelo:
features = ["TIPO PACIENTE", "SEXO", "EDAD", "EMBARAZO", "DIABETES", "EPOC", "ASMA", "INMUNOSUPRESION", "HIPERTENSION",
            "OTRA COMPLICACION", "CARDIOVASCULAR", "OBESIDAD", "RENAL CRONICA", "TABAQUISMO"]

categoricalCols = ['SEXO', 'EMBARAZO', 'DIABETES', 'EPOC', 'ASMA', 'INMUNOSUPRESION', 'HIPERTENSION', 'OTRA COMPLICACION',
                   'CARDIOVASCULAR', 'OBESIDAD', 'RENAL CRONICA', 'TABAQUISMO']

numericalCols = ["EDAD"]

labelCol = "TIPO PACIENTE"

# cargamos el archivo en un spark DataFrame
df_origen = spark.read.csv(file_path, header=True, encoding='UTF-8', inferSchema=True)
df = df_origen.select(features)
df = limpiaNulls(df)
df.createOrReplaceTempView('covid')

# Procesamos nuestros datos con la función dataProcessing()
raw_data = dataProcessing(df, categoricalCols, numericalCols, labelCol)

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


