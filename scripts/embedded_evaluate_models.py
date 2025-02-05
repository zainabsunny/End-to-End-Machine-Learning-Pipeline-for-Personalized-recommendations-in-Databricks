import mlflow
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, explode, collect_set

# ---------------------------------------
# 1) Evaluate ALS Model
# ---------------------------------------

def evaluate_als_model(model, df):
    """
    Evaluate ALS model using RMSE, Precision, and Recall, and log metrics to MLflow.
    """
    with mlflow.start_run(run_name="ALS Evaluation"):
        try:
            # ✅ Ensure `review_rating` is a float
            df = df.withColumn("review_rating", col("review_rating").cast("float"))

            # Predict ratings
            predictions = model.transform(df)

            # ✅ Use `review_rating` now as a numeric column
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="review_rating", predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)

            # Convert predictions to binary classification (positive = review_rating >= 3)
            predictions = predictions.withColumn("binary_prediction", (col("prediction") >= 3).cast("int"))
            predictions = predictions.withColumn("binary_actual", (col("review_rating") >= 3).cast("int"))

            # Precision & Recall Calculation
            precision = predictions.selectExpr(
                "sum(case when binary_prediction = 1 and binary_actual = 1 then 1 else 0 end) / sum(binary_prediction) as precision"
            ).collect()[0][0]
            recall = predictions.selectExpr(
                "sum(case when binary_prediction = 1 and binary_actual = 1 then 1 else 0 end) / sum(binary_actual) as recall"
            ).collect()[0][0]

            # Handle NaN cases
            precision = precision if precision is not None else 0.0
            recall = recall if recall is not None else 0.0

            # Log metrics
            mlflow.log_metric("ALS_RMSE", rmse)
            mlflow.log_metric("ALS_Precision", precision)
            mlflow.log_metric("ALS_Recall", recall)

            print(f"✅ ALS Model Evaluation\n  RMSE: {rmse}\n  Precision: {precision}\n  Recall: {recall}")
            return {"rmse": rmse, "precision": precision, "recall": recall}

        except Exception as e:
            print(f"❌ Error evaluating ALS model: {e}")
            raise e

# ---------------------------------------
# 2) Evaluate FP-Growth Model 
# ---------------------------------------

def evaluate_fp_growth(model, df):
    """
    Evaluate FP-Growth model by displaying frequent itemsets and calculating precision & recall.
    """
    with mlflow.start_run(run_name="FP-Growth Evaluation"):
        try:
            # ✅ Identify correct column for topics
            topic_col = "topics" if "topics" in df.columns else "category_code"

            # ✅ Ensure topics are properly exploded into itemsets
            df = df.withColumn("topic", explode(col(topic_col)))
            transactions = df.groupBy("cosmetic_product_id").agg(
                collect_set("topic").alias("topics")
            )

            # ✅ Apply FP-Growth model
            transformed_df = model.transform(transactions)

            # ✅ Get frequent itemsets & association rules
            freq_itemsets = model.freqItemsets.limit(5).toPandas()
            association_rules = model.associationRules.limit(5).toPandas()

            # ✅ Compute Precision and Recall
            matched_recommendations = transformed_df.select("topics", "prediction").toPandas()

            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for index, row in matched_recommendations.iterrows():
                actual_items = set(row["topics"]) if row["topics"] else set()
                predicted_items = set(row["prediction"]) if row["prediction"] else set()

                tp = len(actual_items & predicted_items)  # Intersection
                fp = len(predicted_items - actual_items)  # Predicted but not actual
                fn = len(actual_items - predicted_items)  # Actual but not predicted

                true_positives += tp
                false_positives += fp
                false_negatives += fn

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

            # ✅ Log metrics to MLflow
            mlflow.log_metric("FP-Growth Precision", precision)
            mlflow.log_metric("FP-Growth Recall", recall)
            mlflow.log_metric("FP-Growth Frequent Itemsets Count", freq_itemsets.shape[0])
            mlflow.log_metric("FP-Growth Association Rules Count", association_rules.shape[0])

            print(f"✅ FP-Growth Model Evaluation\nPrecision: {precision:.4f}, Recall: {recall:.4f}")
            print("\nTop Frequent Itemsets:\n", freq_itemsets)
            print("\nTop Association Rules:\n", association_rules)

            return {
                "precision": precision,
                "recall": recall,
                "freq_itemsets": freq_itemsets,
                "association_rules": association_rules
            }

        except Exception as e:
            print(f"❌ Error evaluating FP-Growth model: {e}")
            raise e

# ---------------------------------------
# 3) Evaluate Cosine Similarity Model
# ---------------------------------------

def evaluate_cosine_similarity(df_sim, product_embeddings_df, filename="cosine_sim_recommendations.csv", top=10):
    """
    Evaluate cosine similarity by generating top recommendations and logging Precision & Recall.
    """
    with mlflow.start_run(run_name="Cosine Similarity Evaluation"):
        try:
            # Get product IDs
            product_ids = product_embeddings_df.select("review_id").toPandas()["review_id"]

            recommendations = []
            for product_id in product_ids:
                top_recs = (
                    df_sim[product_id]
                    .sort_values(ascending=False)
                    .iloc[1:top + 1]  # Exclude self-similarity
                )
                recommendations.append({
                    "review_id": product_id,
                    "recommendations": list(top_recs.index),
                    "scores": list(top_recs.values)
                })

            # Convert recommendations to DataFrame
            recommendations_df = pd.DataFrame(recommendations)

            # Compute Precision & Recall
            actual_interactions = product_embeddings_df.select("review_id", "product_id").toPandas()
            actual_dict = actual_interactions.groupby("review_id")["product_id"].apply(set).to_dict()

            precisions, recalls = [], []
            for rec in recommendations:
                actual_set = actual_dict.get(rec["review_id"], set())
                predicted_set = set(rec["recommendations"])

                if len(predicted_set) > 0:
                    precision = len(predicted_set & actual_set) / len(predicted_set)
                else:
                    precision = 0
                
                if len(actual_set) > 0:
                    recall = len(predicted_set & actual_set) / len(actual_set)
                else:
                    recall = 0

                precisions.append(precision)
                recalls.append(recall)

            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)

            # Save recommendations
            recommendations_df.to_csv(filename, index=False)
            mlflow.log_artifact(filename)

            # Log Precision & Recall
            mlflow.log_metric("Cosine_Similarity_Precision", avg_precision)
            mlflow.log_metric("Cosine_Similarity_Recall", avg_recall)

            print(f"✅ Cosine Similarity Evaluation\n  Precision: {avg_precision}\n  Recall: {avg_recall}")
            return {"precision": avg_precision, "recall": avg_recall}

        except Exception as e:
            print(f"❌ Error evaluating Cosine Similarity model: {e}")
            raise e
