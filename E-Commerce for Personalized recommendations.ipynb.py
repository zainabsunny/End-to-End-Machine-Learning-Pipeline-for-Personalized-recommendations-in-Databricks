# Databricks notebook source
# MAGIC %md
# MAGIC # Royal Cybers: End-to-End Machine Learning Pipeline for Personalized recommendations in Databricks 

# COMMAND ----------

# MAGIC %md
# MAGIC ## RUN THIS NOTEBOOK

# COMMAND ----------

import importlib
from pyspark.sql import SparkSession
from scripts.data_cleaning import clean_cosmetic_df, clean_mapping_df, clean_reviews_df
!pip install spacy
!python -m spacy download en_core_web_sm
from scripts.feature_engineering import process_reviews_df, add_customer_engagement, add_predictor_features
from scripts.data_transformation import transform_cosmetic_data, transform_reviews_data, transform_mapping_data
from pyspark.sql.functions import col
from scripts.EDA import perform_eda


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset

# COMMAND ----------

# Load Dataset from S3 Bucket
spark = SparkSession.builder.appName("E-Commerce Pipeline").getOrCreate()

cosmetic_store_data_path = "s3://e-commerce-pipeline-dataset/Cosmetic Store Website Data.csv"
reviews_data_path = "s3://e-commerce-pipeline-dataset/nyka_top_brands_cosmetics_product_reviews.csv"
product_mapping_path = "s3://e-commerce-pipeline-dataset/unique_product_id_pairings.csv"

# COMMAND ----------

cosmetic_df = spark.read.csv(cosmetic_store_data_path, header=True, inferSchema=True)
reviews_df = spark.read.csv(reviews_data_path, header=True, inferSchema=True)
mapping_df = spark.read.csv(product_mapping_path, header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Data

# COMMAND ----------

cosmetic_df = clean_cosmetic_df(cosmetic_df)
mapping_df = clean_mapping_df(mapping_df)
reviews_df = clean_reviews_df(reviews_df)

# COMMAND ----------

cosmetic_df.show(5)
mapping_df.show(5)
reviews_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Table Paths

# COMMAND ----------

cosmetic_delta_path = "/mnt/delta/cosmetic_store_data"
reviews_delta_path = "/mnt/delta/product_reviews"
mapping_delta_path = "/mnt/delta/product_mapping"

# COMMAND ----------

# Save cleaned DataFrames to Delta format
cosmetic_df.write.format("delta").mode("overwrite").save(cosmetic_delta_path)
reviews_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(reviews_delta_path)
mapping_df.write.format("delta").mode("overwrite").save(mapping_delta_path)

# COMMAND ----------

# Load Delta tables
cosmetic_df = spark.read.format("delta").load(cosmetic_delta_path)
reviews_df = spark.read.format("delta").load(reviews_delta_path)
mapping_df = spark.read.format("delta").load(mapping_delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Process unstructured data (reviews)

# COMMAND ----------

reviews_df = process_reviews_df(reviews_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add outcome variable (Y)

# COMMAND ----------

# Sometimes the scripts dont get updated here, so this should do it
import importlib
from scripts import feature_engineering

# Reload the module
importlib.reload(feature_engineering)
from scripts.feature_engineering import process_reviews_df, add_customer_engagement, add_predictor_features

# COMMAND ----------

cosmetic_df = add_customer_engagement(cosmetic_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add predictor variables (X)

# COMMAND ----------

cosmetic_df = add_predictor_features(cosmetic_df)

# COMMAND ----------

reviews_df.show(5)
cosmetic_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Transformation

# COMMAND ----------

# Sometimes the scripts dont get updated here, so this should do it
import importlib
from scripts import data_transformation

# Reload the module
importlib.reload(data_transformation)
from scripts.data_transformation import transform_cosmetic_data, transform_reviews_data, transform_mapping_data

# COMMAND ----------

cosmetic_df = transform_cosmetic_data(cosmetic_df)
reviews_df = transform_reviews_data(reviews_df)
mapping_df = transform_mapping_data(mapping_df)

# COMMAND ----------

cosmetic_df.show(5)
reviews_df.show(5)
mapping_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA (optional)

# COMMAND ----------

# Combine datasets for EDA
cosmetic_mapped_df = cosmetic_df.join(mapping_df, cosmetic_df["cosmeticProductId"] == mapping_df["cosmeticProductId"], "inner")
combined_df = cosmetic_mapped_df.join(reviews_df, cosmetic_mapped_df["reviewProductId"] == reviews_df["reviewProductId"], "inner")

# Drop unnecessary columns
combined_df = combined_df.drop(
    "cosmeticProductId", "reviewProductId", "event_type", "brand_name", "stemmed_title", "lemmatized_title",
    "stemmed_text", "lemmatized_text", "review_text_clean", "review_title_clean"
)

# Save combined data to Delta table
combined_data_path = "/mnt/delta/combined_cleaned_data"
combined_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(combined_data_path)

# Reload combined Delta table for EDA
combined_df = spark.read.format("delta").load(combined_data_path)

# COMMAND ----------

# Need to fix function, not fully working. Check EDA.ipynb for EDA
# perform_eda(combined_df, reviews_df)

