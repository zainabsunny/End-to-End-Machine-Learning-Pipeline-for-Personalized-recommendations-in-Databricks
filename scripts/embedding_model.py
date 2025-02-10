import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from pyspark.sql.functions import when, col, rand
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, mean_squared_error, f1_score

# ---------------------------------------
# 1) Assign Numerical Values to Event Types (Corrected)
# ---------------------------------------
def preprocess_data(final_embedding_df):
    """
    Prepares interaction data for Neural Collaborative Filtering (NCF).
    Assigns numerical scores to event types and encodes user/product IDs.
    Handles class imbalance.
    """

    # ✅ Corrected Interaction Score Mapping
    interaction_df = final_embedding_df.withColumn(
        "interaction_score",
        when(col("event_type") == "purchase", 3)
        .when(col("event_type") == "cart", 2)
        .when(col("event_type") == "view", 1)
    ).select("user_id", "cosmetic_product_id", "interaction_score")

    # ✅ Convert to Pandas DataFrame
    interaction_data = interaction_df.toPandas()

    # ✅ Handle Class Imbalance (Reduce Views)
    purchase_df = interaction_data[interaction_data["interaction_score"] == 3]
    cart_df = interaction_data[interaction_data["interaction_score"] == 2]
    view_df = interaction_data[interaction_data["interaction_score"] == 1].sample(frac=0.1, random_state=42)  # Reduce views

    balanced_data = pd.concat([purchase_df, cart_df, view_df])

    # ✅ Encode User & Product IDs (Fix Label Encoding)
    user_encoder = LabelEncoder().fit(balanced_data["user_id"])
    product_encoder = LabelEncoder().fit(balanced_data["cosmetic_product_id"])

    balanced_data["user_id"] = balanced_data["user_id"].apply(lambda x: user_encoder.transform([x])[0] if x in user_encoder.classes_ else -1)
    balanced_data["cosmetic_product_id"] = balanced_data["cosmetic_product_id"].apply(lambda x: product_encoder.transform([x])[0] if x in product_encoder.classes_ else -1)

    return balanced_data, user_encoder, product_encoder

# ---------------------------------------
# 2) Define Neural Collaborative Filtering Model
# ---------------------------------------
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output interaction score
        )

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        x = torch.cat([user_embeds, item_embeds], dim=-1)
        return self.fc_layers(x)

# ---------------------------------------
# 3) Train Neural Collaborative Filtering Model
# ---------------------------------------
def train_ncf_embedded(final_embedding_df, num_epochs=10, batch_size=64, learning_rate=0.001):
    """
    Train the Neural Collaborative Filtering (NCF) model and log results to MLflow.
    """
    # ✅ Preprocess data
    interaction_data, user_encoder, product_encoder = preprocess_data(final_embedding_df)

    # ✅ Split into Train/Test (Stratified)
    train_data, test_data = train_test_split(
        interaction_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=interaction_data["interaction_score"]
    )

    # ✅ Convert to PyTorch tensors
    users = torch.tensor(train_data["user_id"].values, dtype=torch.long)
    items = torch.tensor(train_data["cosmetic_product_id"].values, dtype=torch.long)
    labels = torch.tensor(train_data["interaction_score"].values, dtype=torch.float32)

    # ✅ Create DataLoader
    dataset = TensorDataset(users, items, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ✅ Initialize Model
    num_users = len(user_encoder.classes_)
    num_items = len(product_encoder.classes_)
    model = NCF(num_users, num_items)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ✅ Train Model
    with mlflow.start_run(run_name="NCF Training"):
        for epoch in range(num_epochs):  
            total_loss = 0
            for user_batch, item_batch, label_batch in train_loader:
                optimizer.zero_grad()
                preds = model(user_batch, item_batch).squeeze()
                loss = criterion(preds, label_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("Train_Loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        mlflow.pytorch.log_model(model, "ncf_model")

    return model, user_encoder, product_encoder, test_data

# ---------------------------------------
# 4) Evaluate Model Performance
# ---------------------------------------

def evaluate_ncf_embedded(model, test_data):
    """
    Evaluate NCF model using RMSE, Precision, Recall, and F1-score.
    """
    with mlflow.start_run(run_name="NCF Evaluation"):
        try:
            users = torch.tensor(test_data["user_id"].values, dtype=torch.long)
            items = torch.tensor(test_data["cosmetic_product_id"].values, dtype=torch.long)
            labels = test_data["interaction_score"].values

            with torch.no_grad():
                predictions = model(users, items).squeeze().numpy()

            # Compute RMSE
            rmse = np.sqrt(mean_squared_error(labels, predictions))
            mlflow.log_metric("NCF_RMSE", rmse)

            # ✅ Compute Weighted Precision, Recall, and F1-score
            binary_labels = (labels >= 2).astype(int)  # Purchase & Cart = 1, View = 0
            binary_preds = (predictions >= 2).astype(int)

            precision = precision_score(binary_labels, binary_preds, zero_division=0, average="weighted")
            recall = recall_score(binary_labels, binary_preds, zero_division=0, average="weighted")
            f1 = f1_score(binary_labels, binary_preds, zero_division=0, average="weighted")

            mlflow.log_metric("NCF_Precision", precision)
            mlflow.log_metric("NCF_Recall", recall)
            mlflow.log_metric("NCF_F1_Score", f1)

            print(f"✅ RMSE: {rmse}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
            return {"RMSE": rmse, "Precision": precision, "Recall": recall, "F1-Score": f1}

        except Exception as e:
            print(f"❌ Error evaluating NCF model: {e}")
            raise e

# ---------------------------------------
# 5) Generate Recommendations
# ---------------------------------------
def generate_ncf_embedded_recommendations(model, interaction_data, user_encoder, product_encoder, filename="ncf_embedded_recs.csv", k=10):
    """
    Generate NCF recommendations for each user using the trained embedded model.
    Saves recommendations to both a CSV file and Unity Catalog.
    """
    with mlflow.start_run(run_name="NCF Embedded Recommendations"):
        try:
            start_time = time.time()

            # Get unique user and product IDs
            user_ids = interaction_data["user_id"].unique()
            product_ids = interaction_data["cosmetic_product_id"].unique()

            recommendations = []

            for user in user_ids:
                # Convert user IDs safely (unseen users get a default value)
                user_idx = torch.tensor(
                    [user_encoder.transform([user])[0] if user in user_encoder.classes_ else 0] * len(product_ids),
                    dtype=torch.long
                )

                # Convert product IDs safely (unseen products get a default value)
                item_idx = torch.tensor([
                    product_encoder.transform([item])[0] if item in product_encoder.classes_ else 0 for item in product_ids
                ], dtype=torch.long)

                # Predict scores
                with torch.no_grad():
                    scores = model(user_idx, item_idx).squeeze().numpy()

                # Get top K recommended products
                top_k_items = product_ids[np.argsort(scores)[-k:][::-1]]

                # Store recommendations in a DataFrame format
                for rank, item in enumerate(top_k_items, 1):
                    recommendations.append([user, item, rank])

            # Convert to Pandas DataFrame
            rec_df = pd.DataFrame(recommendations, columns=["user_id", "cosmetic_product_id", "rank"])

            # ✅ Save to CSV
            rec_df.to_csv(filename, index=False)
            mlflow.log_artifact(filename)

            mlflow.log_metric("recommendation_generation_time", round(time.time() - start_time, 2))
            print(f"✅ NCF Embedded recommendations saved to {filename}")

            return rec_df

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise e
