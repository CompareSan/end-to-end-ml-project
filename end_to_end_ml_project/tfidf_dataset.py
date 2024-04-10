import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class TfidfDataset(Dataset):
    def __init__(self, tfidf_vectors, labels, transform=None, target_transform=None):
        self.labels = torch.tensor(labels.values)
        self.feature_vectors = torch.tensor(csr_matrix.todense(tfidf_vectors)).float()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_vector = self.feature_vectors[idx]

        label = self.labels[idx]
        if self.transform:
            feature_vector = self.transform(feature_vector)
        if self.target_transform:
            label = self.target_transform(label)
        return feature_vector, label
