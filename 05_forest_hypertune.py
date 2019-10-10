# # Tuning hyperparameters using grid search

# ## Setup

# Import useful packages, modules, classes, and functions:
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
#import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

# Create a SparkSession:
spark = SparkSession.builder.master("local").appName("creditcard_hypertune").getOrCreate()

# ## Preprocess the modeling data

# Read the explored data from HDFS:
df = spark.read.parquet("creditcardfraud/exploredata/")

# Now we manually select our features and label:
# Features selected
feature_selected = ["V1","V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19"]
df_selected = df.select("Time","V1","V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19","Class")

# The machine learning algorithms in Spark MLlib expect the features to be collected into
# a single column. So we use
# [VectorAssembler](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)
# to assemble our feature vector:
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=feature_selected, outputCol="Features")
df_assembled = assembler.transform(df_selected)
df_assembled.head(5)


# ## Create train and test datasets for machine learning (classification).
# Fit our model on the train DataFrame and evaluate our model on the test DataFrame:
# We want both train and test dataset to have equal proportion of normal and fraud transactions.

df_norm = df_assembled.filter(df_assembled.Class == 0)
df_fraud = df_assembled.filter(df_assembled.Class == 1)

(norm_train, norm_test) = df_norm.randomSplit([0.7, 0.3], 12345)
(fraud_train, fraud_test) = df_fraud.randomSplit([0.7, 0.3], 12345)

df_train = norm_train.union(fraud_train).orderBy("Time")
df_test = norm_test.union(fraud_test).orderBy("Time")


# ## Requirements for hyperparameter tuning

# We need to specify four components to perform hyperparameter tuning using
# grid search:
# * Estimator
# * Hyperparameter grid
# * Evaluator
# * Validation method


# ## Specify the estimator

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol="Features", labelCol="Class")

# Use the `explainParams` method to get a full list of parameters:
print(rf.explainParams())


# ## Try tuning parameter, numTrees.
# ## Specify hyperparameter grid

# Use the
# [ParamGridBuilder](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder)
# class to specify a hyperparameter grid:
from pyspark.ml.tuning import ParamGridBuilder
numTreesList = [5, 10, 20, 30]
grid = ParamGridBuilder().addGrid(rf.numTrees, numTreesList).build()

# The resulting object is simply a list of parameter maps:
grid


# ## Specify the evaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Class", metricName="areaUnderPR")


# ## Tuning the hyperparameters using holdout cross-validation

# For large DataFrames, holdout cross-validation will be more efficient.  Use
# the
# [TrainValidationSplit](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.TrainValidationSplit)
# class to specify holdout cross-validation:
from pyspark.ml.tuning import TrainValidationSplit
validator = TrainValidationSplit(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator, trainRatio=0.75, seed=54321)

# Use the `fit` method to find the best set of hyperparameters:
%time cv_model = validator.fit(df_train)

# **Note:** Our train DataFrame is split again according to `trainRatio`.

# The resulting model is an instance of the
# [TrainValidationSplitModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.TrainValidationSplitModel)
# class:
type(cv_model)

# The cross-validation results are stored in the `validationMetrics` attribute:
cv_model.validationMetrics

# Plotting Validation Metric for each set of hyperparameters (NumTrees).

def plot_holdout_results(model):
  plt.plot(numTreesList, model.validationMetrics)
  plt.title("Hyperparameter Tuning Results")
  plt.xlabel("NumTrees")
  plt.ylabel("Area_PR")
  plt.show()
plot_holdout_results(cv_model)

# Save the best model to HDFS
cv_model.bestModel.write().overwrite().save("models/bestdtree")

# ## Cleanup
# Stop the SparkSession:
# spark.stop()
