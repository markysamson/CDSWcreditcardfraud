# Certification Project by Joon Kim
#
# ## Part 1 - Data Exploration, Feature Selection

# In this module we use Spark in conjunction with some popular Python libraries
# to explore data and select features we will use in the next module which is model training,
# scoring and evaluation.

# ## Setup
# Import some useful packages and modules:
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a SparkSession:
spark = SparkSession.builder.master("local").appName("creditcard_explore01").getOrCreate()

# Copy credit card data from CDSW project to HDFS

#!hdfs dfs -mkdir creditcardfraud
#!hdfs dfs -put data/creditcard.csv creditcardfraud

# Load the creditcard data from HDFS:

df = spark.read.csv("creditcardfraud/", sep=",", header=True, inferSchema=True)
# Show first 5 lines to see if the delimited lines have been read properly
df.show(5)

# And Print schema
df.printSchema()

# Define a new schema
from pyspark.sql.types import *
schema = StructType([
    StructField("Time", DoubleType()),
    StructField("V1", DoubleType()),
    StructField("V2", DoubleType()),
    StructField("V3", DoubleType()),
    StructField("V4", DoubleType()),
    StructField("V5", DoubleType()),
    StructField("V6", DoubleType()),
    StructField("V7", DoubleType()),
    StructField("V8", DoubleType()),
    StructField("V9", DoubleType()),
    StructField("V10", DoubleType()),
    StructField("V11", DoubleType()),
    StructField("V12", DoubleType()),
    StructField("V13", DoubleType()),
    StructField("V14", DoubleType()),
    StructField("V15", DoubleType()),
    StructField("V16", DoubleType()),
    StructField("V17", DoubleType()),
    StructField("V18", DoubleType()),
    StructField("V19", DoubleType()),
    StructField("V20", DoubleType()),
    StructField("V21", DoubleType()),
    StructField("V22", DoubleType()),
    StructField("V23", DoubleType()),
    StructField("V24", DoubleType()),
    StructField("V25", DoubleType()),
    StructField("V26", DoubleType()),
    StructField("V27", DoubleType()),
    StructField("V28", DoubleType()),
    StructField("Amount", DoubleType()),
    StructField("Class", IntegerType())
])

df = spark \
  .read \
  .format("csv") \
  .option("sep", ",") \
  .option("header", True) \
  .schema(schema) \
  .load("creditcardfraud/creditcard.csv")

df.describe("Time","Amount","Class").show()

# Run some basic checks on the data - any NULL values?
df_nonull = df.dropna()
df_nonull.describe("Time","Amount","Class").show()
  
# Add a new Category Column "Fraud"
df2 = df.withColumn("Fraud", df.Class == 1)

# Describe the new DataFrame
df2.select("Time", "V1", "V2", "Amount", "Class", "Fraud").show(5)
df2.describe("Time", "V1", "V2", "Amount", "Class").show()
  
# Load into Panda Dataframe to visualize summary better.  
pdf = df2.toPandas()
pdf.describe()


# Time Column - View distribution
# Plot Time with normal, and plot Time with fraud
# sns.distplot(pdf["Time"], kde=False)
# sns.distplot(pdf["Time"][pdf.Class == 0], kde=False)
# sns.distplot(pdf["Time"][pdf.Class == 1], kde=False)


# Filter "Normal" DataFrame where Class == 0
# and filter "Fraudulent" DataFrame where Class == 1

pdf_normal = pdf[pdf.Class == 0]
# pdf_normal.count()

# Plot distribution of Normal transactions
sns.jointplot(x="Time", y="Amount", data=pdf_normal, size=12, kind="reg")

pdf_fraud = pdf[pdf.Class == 1]

# Plot Distribution of Fraud transactions
sns.jointplot(x="Time", y="Amount", data=pdf_fraud, size=12, kind="reg")

# FacetGrid
def tmp_plot():  # Wrap plot build into function for CDSW
  g = sns.FacetGrid(data=pdf, col="Fraud", sharex=True, size=10)
  g = g.map(plt.scatter, "Time", "Amount")
tmp_plot()


# Explore each "V" features
from pyspark.sql.functions import count, mean

v_list = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10", \
          "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20", \
          "V21","V22","V23","V24","V25","V26","V27","V28"]

def explore(vfeatures):
   for v in vfeatures:
    df.rollup("Class").agg(count(v), mean(v)).orderBy("Class").show()

explore(v_list)


def tmp_plot2(vfeatures):
  for v in vfeatures:
    ax = plt.subplot(111)
    sns.distplot(pdf[v][pdf.Class == 1], bins=50)
    sns.distplot(pdf[v][pdf.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('Feature: ' + str(v))
    plt.show()

tmp_plot2(v_list)
  

# When visualizing the distribution of data between "normal" and "fraud" transactions,
# the following columns (features) show very different distribution between the two
# transaction types.

feature_selected = ["V1","V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19"]

# Save the data for next phase, Machine Learning
df2.write.parquet("creditcardfraud/exploredata", mode="overwrite")

# ## Cleanup
# Stop the SparkSession:
spark.stop()