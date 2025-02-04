import mlflow
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np

# ---------------------------------------
# 1) Evaluate ALS Model with RMSE
# ---------------------------------------

def evaluate_als_model(model, df):
    """
    Evaluate ALS model using RMSE and log to MLflow.
    """
    with mlflow.start_run(run_name="ALS Evaluation"):
        try:
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="sentiment_score", predictionCol="prediction")
            predictions = model.transform(df)
            rmse = evaluator.evaluate(predictions)

            # Log metrics
            mlflow.log_metric("ALS_RMSE", rmse)

            print(f"ALS RMSE: {rmse}")
            return rmse

        except Exception as e:
            print(f"Error evaluating ALS model: {e}")
            raise e

# ---------------------------------------
# 2) Evaluate FP-Growth Model
# ---------------------------------------

def evaluate_fp_growth(model):
    """
    Show top frequent itemsets and log to MLflow.
    """
    with mlflow.start_run(run_name="FP-Growth Evaluation"):
        try:
            freq_itemsets = model.freqItemsets.limit(5).toPandas()
            
            # Log to MLflow
            mlflow.log_metric("FP-Growth Frequent Itemsets Count", freq_itemsets.shape[0])
            
            print("Top Frequent Itemsets:\n", freq_itemsets)
            return freq_itemsets

        except Exception as e:
            print(f"Error evaluating FP-Growth model: {e}")
            raise e

# ---------------------------------------
# 3) Evaluate Cosine Similarity Model
# ---------------------------------------

def evaluate_cosine_similarity(df_sim, product_embeddings_df, filename="cosine_sim_recommendations.csv", top=10):
    """
    Evaluate cosine similarity by generating top recommendations and logging results.
    """
    with mlflow.start_run(run_name="Cosine Similarity Evaluation"):
        try:
            product_ids = product_embeddings_df["review_product_id"]

            recommendations = []
            for product_id in product_ids:
                top_recs = (
                    df_sim[product_id]
                    .sort_values(ascending=False)
                    .iloc[1:top + 1]  # Exclude self-similarity
                )
                recommendations.append({
                    "product_id": product_id,
                    "recommendations": list(top_recs.index),
                    "scores": list(top_recs.values)
                })

            # Save recommendations to a file
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df.to_csv(filename, index=False)
            mlflow.log_artifact(filename)

            print("Cosine similarity recommendations saved and logged to MLflow.")
            return recommendations_df

        except Exception as e:
            print(f"Error evaluating Cosine Similarity model: {e}")
            raise e
