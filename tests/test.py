from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.pipeline import Pipeline

stages = []
pipe = Pipeline(stages=stages)
pipeModel = pipe.fit()

