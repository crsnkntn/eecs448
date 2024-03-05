import os
import csv
from transformers import BertTokenizer, BertModel
import torch

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
input_directory = "datasets/all/max/csv"  # Update this path to your input directory
output_directory = "datasets/mini/classes"  # Update this path to your output directory
model_name = 'bert-base-uncased'  # Model to use for embeddings
max_length = 64  # Maximum sequence length for BERT
max_chars = 256  # Max characters to keep from each text field
batch_size = 16  # Number of texts to process in a single batch

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def truncate_string(s, max_length):
    return s[:max_length] if len(s) > max_length else s

def convert_to_bert_embeddings(texts, tokenizer, model, batch_size, device):
    embeddings = []
    model.to(device)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', max_length=max_length, padding=True, truncation=True, return_attention_mask=True).to(device)
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.detach().cpu()  # Move embeddings back to CPU
        embeddings.extend(batch_embeddings)
    return embeddings

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

labels = {"ohiostate", "georgia", "oklahoma", "texas", "floridastate", "oregon", "alabama", "notredame", "iowa"}

# Process each CSV file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv") and filename[:-4] in labels:
        output_file_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.pt')
        
        # Check if the .pt file already exists
        if os.path.exists(output_file_path):
            print(f"{output_file_path} already exists. Skipping...")
            continue  # Skip the rest of the loop and process the next file
        
        file_path = os.path.join(input_directory, filename)
        texts = []

        with open(file_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                text = truncate_string(row['post_title'], int(max_chars/6)) + ' ' + \
                       truncate_string(row['post_selftext'], int(max_chars/3)) + ' ' + \
                       row['comment_body']
                texts.append(text)

        # Convert texts to BERT embeddings in batches
        embeddings = convert_to_bert_embeddings(texts, tokenizer, model, batch_size, device)

        # Save embeddings to a .pt file in the output directory
        torch.save(embeddings, output_file_path)
        print(f"Saved embeddings to {output_file_path}")
