from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.classification import RandomForestClassificationModel

spark = SparkSession.builder.master("local").appName("creditcard_predict").getOrCreate()

model = RandomForestClassificationModel.load("models/bestdtree") 

features = ["V1","V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19"]

def predict(args):
  account=args["feature"].split(",")
  feature = spark.createDataFrame([list(map(float,account[0:13]))], features)
  from pyspark.ml.feature import VectorAssembler
  feature_selected = ["V1","V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19"]
  assembler = VectorAssembler(inputCols=feature_selected, outputCol="Features")
  assembled = assembler.transform(feature)
  result=model.transform(assembled).collect()[0].prediction
  return {"result" : result}

# predict({"feature":"1,2,3,4,5,6,7,8,9,10,11,12,13"})
