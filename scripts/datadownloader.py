import kagglehub

# Download latest version
path = kagglehub.dataset_download("owaiskhan9654/pubmed-multilabel-text-classification")

print("Path to dataset files:", path)