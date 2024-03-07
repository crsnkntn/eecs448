import json
import csv
import os

# Replace these paths with the actual paths
source_directory_path = 'top_ten/jsonl'  # Directory containing JSONL files
destination_directory_path = 'top_ten/csv'  # Directory where CSV files will be stored

# Ensure the destination directory exists
os.makedirs(destination_directory_path, exist_ok=True)

for filename in os.listdir(source_directory_path):
    if filename.endswith(".jsonl"):
        jsonl_file_path = os.path.join(source_directory_path, filename)
        # Modify the path for the CSV file to be in the new directory
        csv_file_name = filename.replace('.jsonl', '.csv')
        csv_file_path = os.path.join(destination_directory_path, csv_file_name)
        
        with open(jsonl_file_path, 'r') as jsonl_file, open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = None
            for line in jsonl_file:
                data = json.loads(line)  # Convert JSON string to dictionary
                if csv_writer is None:
                    # Initialize CSV writer and write headers
                    headers = data.keys()
                    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
                    csv_writer.writeheader()
                csv_writer.writerow(data)
        print(f"Converted {jsonl_file_path} to {csv_file_path}")
