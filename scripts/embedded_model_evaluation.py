import torch
import mlflow
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error

# -------------------------------
# 1) Evaluate NCF Model
# -------------------------------

def evaluate_ncf_embedded(model, interaction_data, user_encoder, product_encoder):
    """
    Evaluate NCF model using RMSE, Precision, and Recall.
    """
    with mlflow.start_run(run_name="NCF Evaluation"):
        try:
            # Convert test data to PyTorch tensors
            users = torch.tensor(interaction_data["user_id"].values, dtype=torch.long)
            items = torch.tensor(interaction_data["cosmetic_product_id"].values, dtype=torch.long)
            labels = interaction_data["interaction_score"].values

            # Model Predictions
            with torch.no_grad():
                predictions = model(users, items).squeeze().numpy()

            # Compute RMSE
            rmse = np.sqrt(mean_squared_error(labels, predictions))
            mlflow.log_metric("NCF_RMSE", rmse)

            # Compute Precision & Recall (Threshold = 1.5 for relevant recommendations)
            binary_labels = (labels >= 1.5).astype(int)  # Treat 'purchase' & 'add-to-cart' as positive
            binary_preds = (predictions >= 1.5).astype(int)

            precision = precision_score(binary_labels, binary_preds, zero_division=0)
            recall = recall_score(binary_labels, binary_preds, zero_division=0)

            mlflow.log_metric("NCF_Precision", precision)
            mlflow.log_metric("NCF_Recall", recall)

            print(f"✅ RMSE: {rmse}, Precision: {precision}, Recall: {recall}")
            return {"RMSE": rmse, "Precision": precision, "Recall": recall}

        except Exception as e:
            print(f"❌ Error evaluating NCF model: {e}")
            raise e
