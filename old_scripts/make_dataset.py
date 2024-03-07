import torch
from torch.utils.data import Dataset, DataLoader
import os

class BalancedDataset(Dataset):
    def __init__(self, label_names, dataset_size, max_tensor_length, save_path):
        self.label_names = label_names
        self.dataset_size = dataset_size
        self.max_tensor_length = max_tensor_length
        self.save_path = save_path
        self.data = []
        self.labels = []
        self.load_and_balance_data()
        self.save_dataset()

    def load_and_balance_data(self):
        samples_per_label = self.dataset_size // len(self.label_names)

        # Sample datapoints from each label!
        for label in self.label_names:
            tensor_file = f"datasets/mini/classes/{label}.pt"
            if os.path.exists(tensor_file):
                # Load the pytorch file
                tensors = torch.load(tensor_file)

                # Loop through sampled datapoints
                for idx in torch.randperm(len(tensors))[:samples_per_label]:
                    tensor = tensors[idx]

                    # Pad or trim the tensor
                    if tensor.size(0) < self.max_tensor_length:
                        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, self.max_tensor_length - tensor.size(0)))
                    else:
                        tensor = tensor[:self.max_tensor_length]

                    # Add the datapoint and label
                    self.data.append(tensor)
                    self.labels.append(label)

    def save_dataset(self):
        # Combine data and labels into a dictionary
        dataset_dict = {'data': self.data, 'labels': self.labels}
        # Save the dataset as a .pt file
        torch.save(dataset_dict, self.save_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Create 10 Label Dataset
label_names = ["ohiostate", "georgia", "oklahoma", "texas", "floridastate", "oregon", "alabama", "notredame", "iowa"]
dataset_size = 10
max_tensor_length = 64
save_path = "datasets/mini/mini-10.pt"  # Path where the balanced dataset will be saved

dataset = BalancedDataset(label_names, dataset_size, max_tensor_length, save_path)

# Now the dataset is saved and can be loaded directly for training
loaded_dataset = torch.load(save_path)