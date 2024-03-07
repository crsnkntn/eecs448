import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Check for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
CLEAN_LEVEL = "max"
MODEL_DIM = 768
SAMPLES_PER_CLASS = 5000
dataset_path = f'datasets/mini/{CLEAN_LEVEL}-{MODEL_DIM}-{SAMPLES_PER_CLASS}/dataset.pt'
dataset = torch.load(dataset_path)

labels = ["michigan", "ohiostate", "georgia", "oklahoma", "texas", "floridastate", "oregon", "alabama", "notredame", "iowa"]

X = torch.stack([torch.Tensor(item[0]) for item in dataset]).to(device)
y = torch.tensor([labels.index(item[1]) for item in dataset], dtype=torch.long).to(device)

# Training and Validation sets
X_train, X_val, y_train, y_val = train_test_split(X.cpu(), y.cpu(), test_size=0.3, random_state=420)

# Create DataLoaders
train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
val_dataset = TensorDataset(X_val.to(device), y_val.to(device))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Super Simple 2 layer NN
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=3072):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights using kaiming initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        
        # initialize biases to zero
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc3.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc4.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Eval function
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Train function
def train_model(model, train_loader, val_loader, criterion, epochs=15):
    lr = 0.001
    for epoch in range(epochs):
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")
        
        # Evaluate
        evaluate_model(model, val_loader, criterion)


# Script
model = SimpleDNN(input_dim=MODEL_DIM, output_dim=len(labels)).to(device)

# Loss Function, CE for now since thats my initials
criterion = nn.CrossEntropyLoss()

# Train and evaluate 
train_model(model, train_loader, val_loader, criterion, epochs=10)
