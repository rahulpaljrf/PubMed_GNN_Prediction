import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import torch
import os

dataset_Name='/root/PubMed_GNN_Prediction/dataset/PubMed Multi Label Text Classification Dataset Processed.csv'
df = pd.read_csv(dataset_Name)


cols = list(df.columns)
mesh_Heading_categories = cols[6:]

df_train, df_test = train_test_split(
    df, random_state=32, test_size=0.20, shuffle=True
)

df_train['one_hot_labels'] = list(df_train[mesh_Heading_categories].values)

labels = torch.tensor(
    list(df_train.one_hot_labels.values),
    dtype=torch.float
)

Article_train = list(df_train.abstractText.values)

os.makedirs("/root/PubMed_GNN_Prediction/scripts/data/raw", exist_ok=True)
os.makedirs("/root/PubMed_GNN_Prediction/scripts/data/labels", exist_ok=True)

with open("/root/PubMed_GNN_Prediction/scripts/data/raw/Article_train.pkl", "wb") as f:
    pickle.dump(Article_train, f)

torch.save(labels, "/root/PubMed_GNN_Prediction/scripts/data/labels/labels.pt")