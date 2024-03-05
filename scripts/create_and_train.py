# Per class accuracy
# Examples of correct and incorrect predictions
# 

import torch
import csv
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

BERT_MODEL = "prajjwal1/bert-tiny"
CSV_TYPE = "max"
EMBEDDING_DIM = 128
SAMPLES_PER_CLASS = 7500
MAX_CHARS = 256
BATCH_SIZE = 16
labels = ["michigan", "ohiostate", "georgia", "oklahoma", "texas", "floridastate", "oregon", "alabama", "notredame", "iowa"]
csv_dir = f'datasets/all/{CSV_TYPE}/csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = AutoModel.from_pretrained(BERT_MODEL).to(device)

# Used in 'read_and_sample_csv'
def count_rows_in_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        row_count = sum(1 for row in reader)
    return row_count


def convert_to_bert_embeddings(texts):
    model.eval()
    embeddings = []
    dataloader = DataLoader(texts, batch_size=BATCH_SIZE, 
        collate_fn=lambda x: tokenizer(x, padding=True, truncation=True, return_tensors="pt", max_length=512))

    for batch in tqdm(dataloader, desc="Converting texts to BERT embeddings"):
        inputs = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings += [np.array(e) for e in cls_embeddings]

    return np.array(embeddings)

'''
Iterate through the csv file by chunks

Sample from each chunk

|return_list| < sample_size + samples_per_chunk
'''
def read_and_sample_csv(file_path):

    sampled_texts = []

    # Enforces even sampling
    n_rows = count_rows_in_csv(file_path)
    n_chunks = max(n_rows // 8000, 1)
    samples_per_chunk = SAMPLES_PER_CLASS // n_chunks

    def truncate_string(s, max_len): 
        return s if len(s) <= max_len else s[:max_len]

    # Iterate chunk by chunk
    for chunk in pd.read_csv(file_path, chunksize=8000, encoding='utf-8'):
        chunk_sample = chunk.sample(n=min(samples_per_chunk, len(chunk)))
        for _, row in chunk_sample.iterrows():
            text = str(truncate_string(str(row['post_title']), int(MAX_CHARS//6))) + ' ' + str(truncate_string(str(row['post_selftext']), int(MAX_CHARS//3))) + ' ' + str(row['comment_body'])
            sampled_texts.append(text)

        if len(sampled_texts) >= SAMPLES_PER_CLASS:
            break

    # List of sampled texts
    return sampled_texts


def sample_and_preprocess_data():
    dataset = []
    samples_per_label = {}

    for label in labels:
        file_path = f"{csv_dir}/{label}.csv"
        texts = read_and_sample_csv(file_path)
        embeddings = convert_to_bert_embeddings(texts)
        samples_per_label[label] = len(embeddings)
        # Pair embeddings with the corresponding label
        for embedding in embeddings:
            dataset.append((embedding, label))

    return dataset, samples_per_label


dataset, samples_per_label = sample_and_preprocess_data()

session_info =f"""BERT_MODEL={BERT_MODEL}
    CSV_TYPE={CSV_TYPE}
    EMBEDDING_DIM={EMBEDDING_DIM}
    SAMPLES_PER_CLASS={SAMPLES_PER_CLASS}
    MAX_CHARS={MAX_CHARS}
    BATCH_SIZE={BATCH_SIZE}

    Labels: {', '.join(labels)}
    Samples / label: {str(samples_per_label)}
    """

print(session_info)

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
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
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


def evaluate_model(model, val_loader, criterion, class_names=labels):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    n_correct = {l: 0 for l in class_names}
    n_samples = {l: 0 for l in class_names}

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            for label, prediction in zip(y, predicted):
                if label == prediction:
                    n_correct[class_names[label]] += 1
                n_samples[class_names[label]] += 1

    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    for class_name in class_names:
        if n_samples[class_name] > 0:
            print(f'Accuracy of {class_name}: {n_correct[class_name] / n_samples[class_name]:.4f}')
        else:
            print(f'Accuracy of {class_name}: N/A (No samples)')

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
model = SimpleDNN(input_dim=EMBEDDING_DIM, output_dim=len(labels)).to(device)

# Loss Function, CE for now since thats my initials
criterion = nn.CrossEntropyLoss()

# Train and evaluate 
train_model(model, train_loader, val_loader, criterion, epochs=10)
