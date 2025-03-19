# Make sure you have these installed:
# pip install torch pandas faiss-gpu (or faiss-cpu if no GPU)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ProblemKernelDataset(Dataset):
    def __init__(self, csv_file):
        """
        :param csv_file: Path to the CSV file with columns like:
                         M, N, K, log_flops, EFF, and other kernel feature columns.
        """
        # Read the CSV
        data = pd.read_csv(csv_file)

        # Identify the columns for problem features
        problem_cols = ["M", "N", "K", "log_flops"]

        # Identify the label column
        label_col = "EFF"

        # Identify kernel feature columns as "everything else" (except problem_cols + label_col)
        kernel_cols = [c for c in data.columns if c not in problem_cols + [label_col]]

        # Store the data in tensors
        self.problem_features = data[problem_cols].values.astype("float32")
        self.kernel_features = data[kernel_cols].values.astype("float32")
        self.labels = data[label_col].values.astype("float32")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.problem_features[idx], dtype=torch.float32),
            torch.tensor(self.kernel_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class TowerNetwork(nn.Module):
    """A simple MLP tower to transform input features into an embedding."""

    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(TowerNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)  # (batch_size, embed_dim)


class TwoTowerModelDense(nn.Module):
    """
    Two-tower model:
      - One tower for problem features
      - One tower for kernel features
      - A dense layer to combine the two embeddings and predict EFF
    """

    def __init__(self, problem_input_dim, kernel_input_dim, hidden_dim, embed_dim):
        super(TwoTowerModelDense, self).__init__()

        # Towers
        self.problem_net = TowerNetwork(problem_input_dim, hidden_dim, embed_dim)
        self.kernel_net = TowerNetwork(kernel_input_dim, hidden_dim, embed_dim)

        # Combination layer: here we concatenate embeddings and pass through more layers
        self.combination = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),  # final output is a single scalar (EFF)
        )

    def forward(self, problem_x, kernel_x):
        # Get the embeddings from each tower
        p_emb = self.problem_net(problem_x)  # (batch_size, embed_dim)
        k_emb = self.kernel_net(kernel_x)  # (batch_size, embed_dim)

        # Concatenate them
        combined = torch.cat([p_emb, k_emb], dim=1)  # (batch_size, 2*embed_dim)

        # Predict EFF
        eff_pred = self.combination(combined).squeeze(1)  # (batch_size,)
        return eff_pred


def train_two_tower_model(csv_file, epochs=10, batch_size=32, lr=1e-3):
    # 1. Create the dataset and data loader
    dataset = ProblemKernelDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Infer input dimensions
    #    We assume the dataset stores problem_features and kernel_features as numpy arrays
    sample_problem_feat, sample_kernel_feat, _ = dataset[0]
    problem_input_dim = sample_problem_feat.shape[0]
    kernel_input_dim = sample_kernel_feat.shape[0]

    # 3. Create the model
    hidden_dim = 64
    embed_dim = 32
    model = TwoTowerModelDense(
        problem_input_dim, kernel_input_dim, hidden_dim, embed_dim
    )

    # 4. Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for problem_x, kernel_x, eff_true in dataloader:
            optimizer.zero_grad()
            eff_pred = model(problem_x, kernel_x)
            loss = criterion(eff_pred, eff_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * problem_x.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss = {epoch_loss:.6f}")

    return model


# --- Example usage ---
model = train_two_tower_model("nngrid_dataset.csv", epochs=10, batch_size=32, lr=1e-3)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

# # --- Define the Attention-Based Embedding Network ---

# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(SelfAttention, self).__init__()
#         # Using batch_first=True so inputs are (batch, seq_len, embed_dim)
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

#     def forward(self, x):
#         # x: (batch, seq_len, embed_dim)
#         attn_output, _ = self.multihead_attn(x, x, x)
#         return attn_output

# class AttentionEmbeddingNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, embed_dim, num_heads):
#         super(AttentionEmbeddingNetwork, self).__init__()
#         # Initial MLP layers to transform raw input
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, embed_dim)
#         # Self-attention layer: if you have more than one token per instance, attention can aggregate
#         self.attention = SelfAttention(embed_dim, num_heads)
#         # Final transformation to produce the embedding
#         self.final_fc = nn.Linear(embed_dim, embed_dim)

#     def forward(self, x):
#         # x: (batch, input_dim)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)  # Now x is (batch, embed_dim)
#         # Add a sequence dimension (here we use seq_len = 1)
#         x = x.unsqueeze(1)  # (batch, 1, embed_dim)
#         x = self.attention(x)  # (batch, 1, embed_dim)
#         x = x.squeeze(1)       # (batch, embed_dim)
#         x = self.final_fc(x)   # Final embedding
#         return x

# # --- Define the Two-Tower Model ---
# # One tower for problem features and one for kernel features.
# class TwoTowerModel(nn.Module):
#     def __init__(self, problem_input_dim, kernel_input_dim, hidden_dim, embed_dim, num_heads):
#         super(TwoTowerModel, self).__init__()
#         self.problem_net = AttentionEmbeddingNetwork(problem_input_dim, hidden_dim, embed_dim, num_heads)
#         self.kernel_net  = AttentionEmbeddingNetwork(kernel_input_dim, hidden_dim, embed_dim, num_heads)

#     def forward(self, problem_features, kernel_features):
#         # Obtain embeddings for problems and kernels
#         problem_emb = self.problem_net(problem_features)   # (batch, embed_dim)
#         kernel_emb  = self.kernel_net(kernel_features)       # (batch, embed_dim)
#         # Compute the dot product for each pair as the predicted performance metric
#         predictions = (problem_emb * kernel_emb).sum(dim=1)  # (batch,)
#         return predictions

# # --- Create a Synthetic Dataset ---
# # Each sample is a tuple: (problem_features, kernel_features, performance)
# class PairDataset(Dataset):
#     def __init__(self, num_samples, input_dim):
#         self.problem_features = torch.randn(num_samples, input_dim)
#         self.kernel_features  = torch.randn(num_samples, input_dim)
#         # For simulation, let's define performance as the dot product of the two feature vectors plus some noise.
#         self.performance = (self.problem_features * self.kernel_features).sum(dim=1) + 0.1 * torch.randn(num_samples)

#     def __len__(self):
#         return self.performance.size(0)

#     def __getitem__(self, idx):
#         return self.problem_features[idx], self.kernel_features[idx], self.performance[idx]

# # --- Training Setup ---

# # Parameters
# num_samples   = 10000
# input_dim     = 10     # Dimension of raw feature vector for problems and kernels
# hidden_dim    = 32
# embed_dim     = 64
# num_heads     = 4
# batch_size    = 32
# num_epochs    = 10

# # Create dataset and DataLoader
# dataset = PairDataset(num_samples, input_dim)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Initialize model, loss function, and optimizer
# model = TwoTowerModel(input_dim, input_dim, hidden_dim, embed_dim, num_heads)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # --- Training Loop ---
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     for problem_features, kernel_features, performance in dataloader:
#         optimizer.zero_grad()
#         # Forward pass: get predicted performance from dot product of embeddings
#         predictions = model(problem_features, kernel_features)
#         loss = criterion(predictions, performance)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * problem_features.size(0)
#     avg_loss = total_loss / len(dataset)
#     print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

# # --- Extracting Embeddings ---
# # After training, you can extract embeddings for a new sample.
# sample_problem = torch.randn(1, input_dim)
# sample_kernel  = torch.randn(1, input_dim)
# problem_embedding = model.problem_net(sample_problem)
# kernel_embedding  = model.kernel_net(sample_kernel)
# print("Sample problem embedding:", problem_embedding)
# print("Sample kernel embedding:", kernel_embedding)


def extract_embeddings(model, dataset, batch_size=128, num_workers=4):
    """
    Extracts embeddings for problem and kernel features from the given model.
    Returns:
      - problem_embeddings: tensor of shape (num_samples, embed_dim)
      - kernel_embeddings: tensor of shape (num_samples, embed_dim)
      - labels: tensor of shape (num_samples,)
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    problem_emb_list = []
    kernel_emb_list = []
    label_list = []

    model.eval()  # set model to evaluation mode

    with torch.no_grad():
        for problem_x, kernel_x, labels in dataloader:
            # If using GPU, make sure to send data to the device.
            problem_x = problem_x.to(next(model.parameters()).device)
            kernel_x = kernel_x.to(next(model.parameters()).device)
            # Get embeddings from each tower
            p_emb = model.problem_net(problem_x)
            k_emb = model.kernel_net(kernel_x)
            problem_emb_list.append(p_emb.cpu())
            kernel_emb_list.append(k_emb.cpu())
            label_list.append(labels.cpu())

    problem_embeddings = torch.cat(problem_emb_list, dim=0)
    kernel_embeddings = torch.cat(kernel_emb_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return problem_embeddings, kernel_embeddings, labels


# Example usage:
# Assuming you have a test dataset (or use the training dataset for quick sanity checks)
test_dataset = ProblemKernelDataset("nngrid_dataset.csv")
problem_embeddings, kernel_embeddings, labels = extract_embeddings(model, test_dataset)
print("Extracted problem embeddings shape:", problem_embeddings.shape)
print("Extracted kernel embeddings shape:", kernel_embeddings.shape)


def evaluate_model(model, dataset, batch_size=128, num_workers=4):
    """
    Evaluates the model on the provided dataset and prints the average MSE loss.
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    criterion = nn.MSELoss()
    total_loss = 0.0
    model.eval()

    device = next(model.parameters()).device
    with torch.no_grad():
        for problem_x, kernel_x, eff_true in dataloader:
            problem_x = problem_x.to(device)
            kernel_x = kernel_x.to(device)
            eff_true = eff_true.to(device)
            eff_pred = model(problem_x, kernel_x)
            loss = criterion(eff_pred, eff_true)
            total_loss += loss.item() * problem_x.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Average Test Loss: {avg_loss:.6f}")
    return avg_loss


# Example usage:
evaluate_model(model, test_dataset)


import faiss
import numpy as np

# Convert kernel embeddings to numpy (and ensure type is float32)
kernel_emb_np = kernel_embeddings.numpy().astype("float32")
embed_dim = kernel_emb_np.shape[1]

# Create a FAISS index (here using L2 distance)
index = faiss.IndexFlatL2(embed_dim)
index.add(kernel_emb_np)
print("FAISS index contains", index.ntotal, "kernel embeddings.")

# For a new problem (or sample from the test set), extract its embedding:
sample_problem, _, _ = test_dataset[0]
sample_problem = sample_problem.unsqueeze(0).to(next(model.parameters()).device)
with torch.no_grad():
    sample_problem_emb = (
        model.problem_net(sample_problem).cpu().numpy().astype("float32")
    )

# Retrieve the top-5 nearest kernel embeddings:
k = 5
distances, indices = index.search(sample_problem_emb, k)
print("Top-5 nearest kernel indices:", indices)
print("Corresponding distances:", distances)


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


def visualize_embeddings(problem_embeddings, kernel_embeddings, sample_size=1000):
    """
    Takes problem_embeddings and kernel_embeddings (tensors or numpy arrays),
    samples them, and plots them in 2D using PCA.
    """
    # Convert to numpy if needed
    if isinstance(problem_embeddings, torch.Tensor):
        problem_embeddings = problem_embeddings.cpu().numpy()
    if isinstance(kernel_embeddings, torch.Tensor):
        kernel_embeddings = kernel_embeddings.cpu().numpy()

    # Determine how many points we actually have
    num_problem = problem_embeddings.shape[0]
    num_kernel = kernel_embeddings.shape[0]

    # Sample indices if the embeddings are large
    sample_problem_idx = np.random.choice(
        num_problem, size=min(sample_size, num_problem), replace=False
    )
    sample_kernel_idx = np.random.choice(
        num_kernel, size=min(sample_size, num_kernel), replace=False
    )

    # Extract the sampled embeddings
    problem_sample = problem_embeddings[sample_problem_idx]
    kernel_sample = kernel_embeddings[sample_kernel_idx]

    # Combine into one array for PCA
    combined = np.concatenate([problem_sample, kernel_sample], axis=0)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    # Separate back out the problem/kernel embeddings in 2D
    prob_2d = combined_2d[: problem_sample.shape[0]]
    kern_2d = combined_2d[problem_sample.shape[0] :]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        prob_2d[:, 0], prob_2d[:, 1], c="blue", alpha=0.6, label="Problems", s=20
    )
    plt.scatter(kern_2d[:, 0], kern_2d[:, 1], c="red", alpha=0.6, label="Kernels", s=20)
    plt.title("PCA Visualization of Problem & Kernel Embeddings")
    plt.legend()
    plt.show()
