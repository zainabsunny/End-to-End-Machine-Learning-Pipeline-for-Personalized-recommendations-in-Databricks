import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from pyspark.sql.functions import when, col
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------
# 1) Assign Numerical Values to Event Types
# ---------------------------------------
def preprocess_data(final_embedding_df):
    """
    Prepares interaction data for Neural Collaborative Filtering (NCF).
    Assigns numerical scores to event types and encodes user/product IDs.
    """
    interaction_df = final_embedding_df.withColumn(
        "interaction_score",
        when(col("event_type") == "purchase", 2.0)
        .when(col("event_type") == "add-to-cart", 1.5)
        .when(col("event_type") == "view", 1.0)
        .otherwise(0.5)  # Default low weight for unknown interactions
    ).select("user_id", "cosmetic_product_id", "interaction_score")

    interaction_data = interaction_df.toPandas()

    # Encode user & product IDs to sequential format for PyTorch
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    interaction_data["user_id"] = user_encoder.fit_transform(interaction_data["user_id"])
    interaction_data["cosmetic_product_id"] = product_encoder.fit_transform(interaction_data["cosmetic_product_id"])

    return interaction_data, user_encoder, product_encoder

# ---------------------------------------
# 2) Define Neural Collaborative Filtering Model
# ---------------------------------------
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single interaction score
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
    # Preprocess data
    interaction_data, user_encoder, product_encoder = preprocess_data(final_embedding_df)

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
