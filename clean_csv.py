import csv
import os

# Define the source and target directories
source_directory_path = 'top_ten/csv'
target_directory_path = 'top_ten/csv_nolinks'

# Ensure the target directory exists
os.makedirs(target_directory_path, exist_ok=True)

def clean_row(row):
    # Placeholder for your row cleaning logic
    return row

for filename in os.listdir(source_directory_path):
    if filename.endswith(".csv"):
        source_file_path = os.path.join(source_directory_path, filename)
        target_file_path = os.path.join(target_directory_path, filename)
        
        with open(source_file_path, 'r', newline='') as source_file, open(target_file_path, 'w', newline='') as target_file:
            csv_reader = csv.DictReader(source_file)
            headers = csv_reader.fieldnames
            csv_writer = csv.DictWriter(target_file, fieldnames=headers)
            csv_writer.writeheader()
            
            for row in csv_reader:
                cleaned_row = clean_row(row)  # Clean the row
                csv_writer.writerow(cleaned_row)
        
        print(f"Processed and cleaned {source_file_path} to {target_file_path}")
