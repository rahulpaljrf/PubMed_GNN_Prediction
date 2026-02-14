import torch
from torch_geometric.nn import knn_graph
import torch.nn.functional as F
import os

emb_path = "/root/PubMed_GNN_Prediction/scripts/data/embeddings"
if os.path.isdir(emb_path):
    parts = sorted([os.path.join(emb_path, f) for f in os.listdir(emb_path) if f.endswith('.pt')])
    if not parts:
        raise FileNotFoundError(f"No .pt files found in {emb_path}")
    X_list = [torch.load(p) for p in parts]
    X = torch.cat(X_list, dim=0)
else:
    X = torch.load(emb_path)

X = F.normalize(X, dim=1)

k = 10
theta = 0.8

edge_index = knn_graph(
    X, k=k, cosine=True, loop=False
)

row, col = edge_index
sim = (X[row] * X[col]).sum(dim=1)

mask = sim >= theta
edge_index = edge_index[:, mask]
edge_weight = sim[mask]

os.makedirs("/root/PubMed_GNN_Prediction/scripts/data/graph", exist_ok=True)
torch.save(edge_index, "/root/PubMed_GNN_Prediction/scripts/data/graph/edge_index.pt")
torch.save(edge_weight, "/root/PubMed_GNN_Prediction/scripts/data/graph/edge_weight.pt")