import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, embeddings_file):
        self.data = torch.load(embeddings_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, embedding = self.data[idx]
        return torch.tensor(embedding), label

# Example usage
if __name__ == "__main__":
    embeddings_file = "datasets/bert/dataset_mini.pt"  # Path to the embeddings file
    batch_size = 2

    dataset = MyDataset(embeddings_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Example usage of the DataLoader
    for batch in dataloader:
        embeddings, labels = batch
        print("Embeddings shape:", embeddings.shape)
        print("Labels:", labels)
        break  # Just print the first batch for demonstration