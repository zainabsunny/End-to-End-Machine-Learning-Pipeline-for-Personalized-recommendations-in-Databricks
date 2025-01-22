import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
import mlflow
import mlflow.spark
from itertools import combinations 

# -----------------------------
# 1) Precision and Recall Calculation
# -----------------------------

def calculate_precision_recall_at_k(recs, ground_truth, k=10):
    """
    Calculate Precision@k and Recall@k for recommendations.

    :param recs: DataFrame containing recommendations.
    :param ground_truth: DataFrame containing ground truth purchases.
    :param k: Number of recommendations to evaluate.
    :return: Precision@k, Recall@k
    """
    precision_list = []
    recall_list = []

    for user in ground_truth["user_session"].unique():
        true_items = ground_truth[ground_truth["user_session"] == user]["cosmetic_product_id"].tolist()
        recommended_items = recs[recs["user_session"] == user]["cosmetic_product_id"].tolist()[:k]

        if not true_items:
            continue  # Skip users without ground truth

        true_positive = len(set(recommended_items) & set(true_items))
        precision = true_positive / len(recommended_items) if recommended_items else 0
        recall = true_positive / len(true_items) if true_items else 0

        precision_list.append(precision)
        recall_list.append(recall)

    precision_at_k = np.mean(precision_list) if precision_list else 0
    recall_at_k = np.mean(recall_list) if recall_list else 0

    return precision_at_k, recall_at_k

# -----------------------------
# 2) MLflow Logging
# -----------------------------

def log_baseline_metrics(model_name, precision, recall, additional_params=None):
    """
    Log precision and recall metrics for a baseline model to MLflow.

    :param model_name: Name of the baseline model (e.g., "Cosine Similarity").
    :param precision: Precision@k value.
    :param recall: Recall@k value.
    :param additional_params: Additional parameters to log (optional).
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        if additional_params:
            for param, value in additional_params.items():
                mlflow.log_param(param, value)
        mlflow.log_metric("precision_at_10", precision)
        mlflow.log_metric("recall_at_10", recall)
        print(f"{model_name} metrics logged to MLflow.")

# -----------------------------
# 3) Evaluation Cosine Models
# -----------------------------

def evaluate_cosine_model(recs, ground_truth, model_name, k=10, additional_params=None):
    """
    Evaluate a baseline model and log metrics to MLflow.

    :param recs: Recommendations DataFrame.
    :param ground_truth: Ground truth DataFrame.
    :param model_name: Name of the model being evaluated.
    :param k: Number of recommendations to evaluate.
    :param additional_params: Additional parameters to log.
    """
    precision, recall = calculate_precision_recall_at_k(recs, ground_truth, k=k)
    log_baseline_metrics(model_name, precision, recall, additional_params)
    return precision, recall

# -----------------------------
# 3) Evaluation Frequent Itemset Model
# -----------------------------

def calculate_precision_recall_frequent_itemsets(frequent_itemsets, ground_truth, k=10, model_name="FP-Growth"):
    """
    Calculate Precision@k and Recall@k for FP-Growth recommendations, considering user sessions.

    :param frequent_itemsets: DataFrame containing frequent itemsets and their frequencies.
    :param ground_truth: DataFrame containing ground truth user sessions and purchased products.
    :param k: Number of recommendations to evaluate.
    """
    precision_list = []
    recall_list = []

    # Prepare ground truth: group products by user_session
    ground_truth_sessions = (
        ground_truth.groupby("user_session")["cosmetic_product_id"]
        .apply(list)
        .reset_index()
    )

    for _, row in ground_truth_sessions.iterrows():
        user_session = row["user_session"]
        true_items = set(row["cosmetic_product_id"])

        # Filter frequent itemsets for items in the user's session
        recommended_items = set()
        for _, itemset_row in frequent_itemsets.iterrows():
            itemset = set(itemset_row["items"])
            if len(itemset & true_items) > 0:  # Overlap between true items and frequent itemsets
                recommended_items.update(itemset)

        # Select top-k recommended items (arbitrary ranking by frequency)
        recommended_items = list(recommended_items)[:k]

        # Calculate Precision and Recall
        if recommended_items:
            true_positive = len(set(recommended_items) & true_items)
            precision = true_positive / len(recommended_items)
            recall = true_positive / len(true_items)
        else:
            precision = 0
            recall = 0

        precision_list.append(precision)
        recall_list.append(recall)

    # Aggregate results
    precision_at_k = np.mean(precision_list) if precision_list else 0
    recall_at_k = np.mean(recall_list) if recall_list else 0

    # Log metrics to MLflow
    log_baseline_metrics(
        model_name=model_name,
        precision=precision_at_k,
        recall=recall_at_k,
        additional_params={"k": k}
    )

    return precision_at_k, recall_at_k

# -----------------------------
# 4) Evaluation ALS
# -----------------------------

def evaluate_als_with_mapping(recs, ground_truth, user_mapping, k=10, model_name="ALS"):
    """
    Evaluate ALS recommendations using user_session and cosmetic_product_id.

    :param recs: ALS recommendations DataFrame with user_session_index and recommendations.
    :param ground_truth: Ground truth DataFrame with user_session and cosmetic_product_id.
    :param user_mapping: DataFrame mapping user_session_index to user_session.
    :param k: Number of recommendations to evaluate.
    """
    # Map user_session_index to user_session
    recs = recs.merge(user_mapping, on="user_session_index", how="left")

    precision_list = []
    recall_list = []

    for _, row in recs.iterrows():
        user = row["user_session"]
        recommended_items = [rec["cosmetic_product_id"] for rec in row["recommendations"][:k]]

        # Get ground truth items for the user
        true_items = ground_truth[ground_truth["user_session"] == user]["cosmetic_product_id"].tolist()

        if not true_items:
            continue

        # Calculate Precision and Recall
        true_positive = len(set(recommended_items) & set(true_items))
        precision = true_positive / len(recommended_items) if recommended_items else 0
        recall = true_positive / len(true_items) if true_items else 0

        precision_list.append(precision)
        recall_list.append(recall)

    # Aggregate results
    precision_at_k = np.mean(precision_list) if precision_list else 0
    recall_at_k = np.mean(recall_list) if recall_list else 0

    # Log metrics to MLflow
    log_baseline_metrics(
        model_name=model_name,
        precision=precision_at_k,
        recall=recall_at_k,
        additional_params={"k": k}
    )

    return precision_at_k, recall_at_k

