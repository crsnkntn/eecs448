import torch
import csv
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Assuming BERT_MODEL is defined
BERT_MODEL = "bert-base-uncased"
CSV_TYPE = "max"
EMBEDDING_DIM = 768
SAMPLES_PER_CLASS = 5000
MAX_CHARS = 256
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = AutoModel.from_pretrained(BERT_MODEL).to(device)

labels = ["michigan", "ohiostate", "georgia", "oklahoma", "texas", "floridastate", "oregon", "alabama", "notredame", "iowa"]
csv_dir = f'datasets/all/{CSV_TYPE}/csv'
dataset_dir = f'datasets/mini/{CSV_TYPE}-{EMBEDDING_DIM}-{SAMPLES_PER_CLASS}/'

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

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
        print(outputs.last_hidden_state[:, 0, :].shape)
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


# SCRIPT

dataset, samples_per_label = sample_and_preprocess_data()

print(session_content)

