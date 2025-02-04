import mlflow
import mlflow.spark
from pyspark.ml.recommendation import ALS
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import explode, col, collect_set
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------
# 1) Train ALS with MLflow
# ---------------------------------------

def train_als_model(df):
    """
    Train ALS model on user-product interactions using MLflow.
    """
    with mlflow.start_run(run_name="ALS Training"):
        try:
            als = ALS(
                maxIter=10,
                regParam=0.1,
                userCol="user_id",
                itemCol="cosmetic_product_id",
                ratingCol="sentiment_score",
                coldStartStrategy="drop"
            )

            als_model = als.fit(df)

            # Log model to MLflow
            mlflow.spark.log_model(als_model, "als_model")
            mlflow.log_param("maxIter", 10)
            mlflow.log_param("regParam", 0.1)

            print("ALS model trained successfully and logged to MLflow.")
            return als_model

        except Exception as e:
            print(f"Error training ALS model: {e}")
            raise e

# ---------------------------------------
# 2) Train FP-Growth with MLflow
# ---------------------------------------

def train_fp_growth(df):
    """
    Train FP-Growth for association rule mining and log in MLflow.
    """
    with mlflow.start_run(run_name="FP-Growth Training"):
        try:
            df = df.withColumn("topic", explode(col("topics")))

            transactions = df.groupBy("cosmetic_product_id").agg(
                collect_set("final_embedding").alias("topics")
            )

            fp_growth = FPGrowth(itemsCol="topics", minSupport=0.01, minConfidence=0.2)
            fp_model = fp_growth.fit(transactions)

            # Log model to MLflow
            mlflow.spark.log_model(fp_model, "fp_growth_model")
            mlflow.log_param("minSupport", 0.01)
            mlflow.log_param("minConfidence", 0.2)

            print("FP-Growth model trained successfully and logged to MLflow.")
            return fp_model

        except Exception as e:
            print(f"Error training FP-Growth model: {e}")
            raise e

# ---------------------------------------
# 3) Train Cosine Similarity with MLflow
# ---------------------------------------

def train_cosine_similarity(product_embeddings_df):
    """
    Train a Cosine Similarity model using product embeddings and log recommendations to MLflow.
    """
    with mlflow.start_run(run_name="Cosine Similarity Training"):
        try:
            # Extract product IDs and embeddings
            product_ids = product_embeddings_df["review_product_id"]
            embeddings = np.array(product_embeddings_df["aggregated_embedding"].to_list())

            # Compute cosine similarity
            similarities = cosine_similarity(embeddings)
            df_sim = pd.DataFrame(similarities, index=product_ids, columns=product_ids)

            print("Cosine similarity model trained successfully.")
            return df_sim  # This will be evaluated separately

        except Exception as e:
            print(f"Error training Cosine Similarity model: {e}")
            raise e
