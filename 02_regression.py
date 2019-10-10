# # Building and Evaluating Binomial Logistic Regression

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
spark = SparkSession.builder.master("local").appName("creditcard_classify").getOrCreate()


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

# Save data for subsequent modules:
df_assembled.write.parquet("creditcardfraud/classifydata", mode="overwrite")


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


# ## Specify a logistic regression model

# Use the
# [LogisticRegression](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression)
# class to specify a logistic regression model:
from pyspark.ml.classification import LogisticRegression
log_reg = LogisticRegression(featuresCol="Features", labelCol="Class")

# Use the `explainParams` method to get a full list of parameters:
print(log_reg.explainParams())

# ## Fit the logistic regression model

# Use the `fit` method to fit the linear regression model on the train DataFrame:
# And use time to measure how long the model fit operation took.
%time log_reg_model = log_reg.fit(df_train)

# The result is an instance of the
# [LogisticRegressionModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegressionModel)
# class:
type(log_reg_model)

# The model parameters are stored in the `intercept` and `coefficients` attributes:
log_reg_model.intercept
log_reg_model.coefficients

# The `summary` attribute is an instance of the
# [BinaryLogisticRegressionTrainingSummary](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.BinaryLogisticRegressionTrainingSummary)
# class:
type(log_reg_model.summary)

# We can query the iteration history:
log_reg_model.summary.totalIterations
log_reg_model.summary.objectiveHistory

# and plot it too:
def plot_iterations(summary):
  plt.plot(summary.objectiveHistory)
  plt.title("Training Summary")
  plt.xlabel("Iteration")
  plt.ylabel("Objective Function")
  plt.show()

plot_iterations(log_reg_model.summary)

# We can also query the model performance, in this case, the area under the ROC curve:
log_reg_model.summary.areaUnderROC

# and plot the ROC curve:
log_reg_model.summary.roc.show(5)

def plot_roc_curve(summary):
  roc_curve = summary.roc.toPandas()
  plt.plot(roc_curve["FPR"], roc_curve["FPR"], "k")
  plt.plot(roc_curve["FPR"], roc_curve["TPR"])
  plt.title("ROC Area: %s" % summary.areaUnderROC)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.show()

plot_roc_curve(log_reg_model.summary)

# **Note:** Redefining the `plot_roc_curve` function in the same session
# results in an error when the function is called.


# ## Evaluate model performance on the test set.

# We have been assessing the model performance on the train DataFrame.  We
# really want to assess it on the test DataFrame.

# **Method 1:** Use the `evaluate` method of the `LogisticRegressionModel` class

test_summary = log_reg_model.evaluate(df_test)

# The result is an instance of the
# [BinaryLogisticRegressionSummary](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.BinaryLogisticRegressionSummary)
# class:
type(test_summary)

# It has attributes similar to those of the
# `BinaryLogisticRegressionTrainingSummary` class:
test_summary.areaUnderROC
plot_roc_curve(test_summary)

# **Method 2:** Use the `evaluate` method of the
# [BinaryClassificationEvaluator](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator)
# class

# Generate predictions on the test DataFrame:
test_with_prediction = log_reg_model.transform(df_test)
# test_with_prediction.show(5)
test_with_prediction.select("Class","rawPrediction","probability","prediction").show(5)

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


# ## Score out a new dataset

# There are two ways to score out a new dataset.

# **Method1:** The `evaluate` method

# The more expensive way is to use the `evaluate` method of the
# `LogisticRegressionModel` class.  The `predictions` attribute of the
# resulting `BinaryLogisticRegressionSummary` instance contains the scored
# DataFrame:
test_with_evaluation = log_reg_model.evaluate(df_test)
test_with_evaluation.predictions.printSchema()
test_with_evaluation.predictions.head(5)

# **Note:** This is more expensive because the `evaluate` method computes all
# the evaluation metrics in addition to scoring out the DataFrame.

# ### Method 2: The `transform` method

# The more direct and efficient way is to use the `transform` method of the
# `LogisticRegressionModel` class:
test_with_prediction = log_reg_model.transform(df_test)
test_with_prediction.printSchema()
test_with_prediction.head(5)


# ## Show how many missed prediction
mismatches = test_with_prediction.filter(test_with_prediction.Class != test_with_prediction.prediction)
mismatches.select("rawPrediction","probability","prediction","Class").show(5)

mis_count = mismatches.count()
print(mis_count)

# ## Cleanup

# Stop the SparkSession:
spark.stop()