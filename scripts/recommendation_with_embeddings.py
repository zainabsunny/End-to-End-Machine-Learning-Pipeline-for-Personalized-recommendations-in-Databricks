import numpy as np
import pandas as pd
import scipy.sparse as sparse
import time
import mlflow
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, FloatType


# -----------------------------
# Cosine Similarity Recommendations with Embeddings
# -----------------------------

def generate_cosine_sim_recs_with_embeddings(product_embeddings_df, filename, top=10):
    """
    Generate recommendations using cosine similarity with precomputed product embeddings.
    """
    import mlflow
    from sklearn.metrics.pairwise import cosine_similarity

    with mlflow.start_run(run_name="Cosine Similarity with Embeddings"):
        try:
            # Extract product IDs and embeddings
            product_ids = product_embeddings_df["review_product_id"]
            embeddings = np.array(product_embeddings_df["aggregated_embedding"].to_list())

            # Compute cosine similarity between embeddings
            similarities = cosine_similarity(embeddings)
            df_sim = pd.DataFrame(similarities, index=product_ids, columns=product_ids)

            # Generate recommendations for each product
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

            print("Cosine similarity recommendations with embeddings generated.")
            return recommendations_df

        except Exception as e:
            print(f"Error in cosine similarity with embeddings: {e}")
            raise e


# -----------------------------
# FP-Growth with Embedding Reranking
# -----------------------------

def rerank_fp_growth_rules(association_rules_df, product_embeddings_dict, top=10):
    """
    Rerank FP-Growth association rules based on cosine similarity of product embeddings.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        # Extract antecedents and consequents
        association_rules_df["antecedent_vector"] = association_rules_df["antecedent"].apply(
            lambda x: np.mean([product_embeddings_dict.get(i, np.zeros(384)) for i in x], axis=0)
        )
        association_rules_df["consequent_vector"] = association_rules_df["consequent"].apply(
            lambda x: np.mean([product_embeddings_dict.get(i, np.zeros(384)) for i in x], axis=0)
        )

        # Compute cosine similarity
        association_rules_df["embedding_similarity"] = association_rules_df.apply(
            lambda row: cosine_similarity(
                [row["antecedent_vector"]], [row["consequent_vector"]]
            )[0][0],
            axis=1
        )

        # Rerank rules based on similarity
        reranked_rules = association_rules_df.sort_values("embedding_similarity", ascending=False).head(top)

        print("FP-Growth rules reranked with embeddings.")
        return reranked_rules

    except Exception as e:
        print(f"Error in reranking FP-Growth rules: {e}")
        raise e

# -----------------------------
# ALS with Embeddings
# -----------------------------

def run_als_recommender_with_embeddings(
    spark_df, 
    embeddings_df, 
    user_col='user_session', 
    item_col='cosmetic_product_id', 
    rating_col='product_quantity',
    embedding_col='aggregated_embedding',
    rank=10, 
    maxIter=10, 
    regParam=0.1
):
    """
    Trains an ALS recommender with product embeddings and logs outputs via MLflow.
    """
    with mlflow.start_run(run_name="ALS-Recommender with Embeddings"):
        try:
            # Join product embeddings with the main dataset
            joined_df = spark_df.join(embeddings_df, item_col, "inner")

            # Check for valid rows after the join
            if joined_df.count() == 0:
                raise ValueError("No valid rows after joining with embeddings. Check your data.")

            # Prepare data: Convert string user_session to numeric index
            user_indexer = StringIndexer(inputCol=user_col, outputCol="user_session_index", handleInvalid="skip")
            item_indexer = StringIndexer(inputCol=item_col, outputCol="review_product_id_index", handleInvalid="skip")
            pipeline = Pipeline(stages=[user_indexer, item_indexer])
            indexed_df = pipeline.fit(joined_df).transform(joined_df)

            # Train ALS model
            als = ALS(
                userCol="user_session_index",
                itemCol="review_product_id_index",
                ratingCol=rating_col,
                rank=rank,
                maxIter=maxIter,
                regParam=regParam,
                implicitPrefs=True,
                coldStartStrategy="drop"
            )
            model = als.fit(indexed_df)

            # Generate recommendations
            user_recs = model.recommendForAllUsers(10).toPandas()
            item_recs = model.recommendForAllItems(10).toPandas()

            # Log ALS parameters
            mlflow.log_param("rank", rank)
            mlflow.log_param("maxIter", maxIter)
            mlflow.log_param("regParam", regParam)

            # Save recommendations
            mlflow.log_metric("user_recommendations_count", len(user_recs))
            mlflow.log_metric("item_recommendations_count", len(item_recs))

            print("ALS model training with embeddings completed successfully.")
            return model, user_recs, item_recs

        except Exception as e:
            print(f"Error in ALS model training: {e}")
            raise e
