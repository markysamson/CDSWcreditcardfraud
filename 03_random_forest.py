# # Building and Evaluating Random Forest Model

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
spark = SparkSession.builder.master("local").appName("creditcard_classify2").getOrCreate()


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

# **Note:** `features` is stored in sparse format.


# ## Create train and test datasets for machine learning (classification).
# Fit our model on the train DataFrame and evaluate our model on the test DataFrame:
# We want both train and test dataset to have equal proportion of normal and fraud transactions.

df_norm = df_assembled.filter(df_assembled.Class == 0)
df_fraud = df_assembled.filter(df_assembled.Class == 1)

(norm_train, norm_test) = df_norm.randomSplit([0.7, 0.3], 12345)
(fraud_train, fraud_test) = df_fraud.randomSplit([0.7, 0.3], 12345)

df_train = norm_train.union(fraud_train).orderBy("Time")
df_test = norm_test.union(fraud_test).orderBy("Time")

# Do some checking on the new DataFrame, see if they look ok.
df_train.select("V1","V2","Features","Class").show(10)
df_test.select("V1","V2","Features","Class").show(10)

# Get some stats on the datasets
df_train.describe("V1","Class").show()
df_test.describe("V1","Class").show()


# ## Specify Random Forest model

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol="Features", labelCol="Class", numTrees=10)

# Use the `explainParams` method to get a full list of parameters:
print(rf.explainParams())

# ## Fit the Random Forest model

# Use the `fit` method to fit the linear regression model on the train DataFrame:
%time rf_model = rf.fit(df_train)

# The result is an instance of the
# [LogisticRegressionModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegressionModel)
# class:
type(rf_model)


# ## Evaluate model performance on the test dataset.

# Use the `evaluate` method of the
# [BinaryClassificationEvaluator](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator)
# class

# Generate predictions on the test DataFrame:
test_with_prediction = rf_model.transform(df_test)
test_with_prediction.select("rawPrediction","probability","prediction","Class").show(5)

test_with_prediction.write.mode("overwrite").saveAsTable("fraud_scored")

# **Note:** The resulting DataFrame includes three types of predictions.  The
# `rawPrediction` is a vector of log-odds, `prediction` is a vector or
# probabilities `prediction` is the predicted class based on the probability
# vector.

# Create an instance of `BinaryClassificationEvaluator` class:
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Class", metricName="areaUnderROC")
print(evaluator.explainParams())
evaluator.evaluate(test_with_prediction)

# Evaluate using another metric:
evaluator.setMetricName("areaUnderPR").evaluate(test_with_prediction)


# ## Show how many missed prediction
mismatches = test_with_prediction.filter(test_with_prediction.Class != test_with_prediction.prediction)
mismatches.select("rawPrediction","probability","prediction","Class").show(5)

mis_count = mismatches.count()
print(mis_count)


# ## Cleanup
# Stop the SparkSession:
# spark.stop()