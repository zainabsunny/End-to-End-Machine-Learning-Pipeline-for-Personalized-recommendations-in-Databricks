import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from pyspark.sql.functions import when, col, lit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error

# ---------------------------------------
# 1) Prepare Data for NCF Training
# ---------------------------------------
def preprocess_ncf_data(purchase_df):
    """
    Prepares purchase data for Neural Collaborative Filtering (NCF).
    Assigns numerical scores to event types and encodes user/product IDs.
    """
    purchase_df = purchase_df.withColumn("interaction_score", lit(2.0))  # Purchase = 2.0
    
    interaction_df = purchase_df.select("user_id", "cosmetic_product_id", "interaction_score")
    interaction_data = interaction_df.toPandas()

    # Encode user & product IDs
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    interaction_data["user_id"] = user_encoder.fit_transform(interaction_data["user_id"])
    interaction_data["cosmetic_product_id"] = product_encoder.fit_transform(interaction_data["cosmetic_product_id"])

    return interaction_data, user_encoder, product_encoder

# ---------------------------------------
# 2) Define NCF Model
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
            nn.Linear(64, 1)  # Output a single score
        )

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        x = torch.cat([user_embeds, item_embeds], dim=-1)
        return self.fc_layers(x)

# ---------------------------------------
# 3) Train NCF Model
# ---------------------------------------
def train_ncf(purchase_df, num_epochs=10, batch_size=64, learning_rate=0.001):
    """
    Train Neural Collaborative Filtering (NCF) model and log results to MLflow.
    """
    interaction_data, user_encoder, product_encoder = preprocess_ncf_data(purchase_df)

    # Convert to PyTorch tensors
    users = torch.tensor(interaction_data["user_id"].values, dtype=torch.long)
    items = torch.tensor(interaction_data["cosmetic_product_id"].values, dtype=torch.long)
    labels = torch.tensor(interaction_data["interaction_score"].values, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(users, items, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    num_users = len(user_encoder.classes_)
    num_items = len(product_encoder.classes_)
    model = NCF(num_users, num_items)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Model
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

        # Log model
        mlflow.pytorch.log_model(model, "ncf_model")

    return model, user_encoder, product_encoder, interaction_data

def generate_ncf_recommendations(model, interaction_data, user_encoder, product_encoder, k=10):
    """
    Generate NCF recommendations for each user.
    """
    user_ids = interaction_data["user_id"].unique()
    product_ids = interaction_data["cosmetic_product_id"].unique()

    recommendations = {}

    for user in user_ids:
        user_tensor = torch.tensor([user] * len(product_ids), dtype=torch.long)
        item_tensor = torch.tensor(product_ids, dtype=torch.long)

        with torch.no_grad():
            scores = model(user_tensor, item_tensor).squeeze().numpy()

        top_k_items = product_ids[np.argsort(scores)[-k:][::-1]]
        recommendations[user] = top_k_items

    return recommendations
