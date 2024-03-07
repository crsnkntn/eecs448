import json
import os

# Setup directories
batch_files_dir = 'arctic-shift-dataset/partitions/'
posts_dir = 'arctic-shift-dataset/posts/'

# Ensure the posts directory exists
if not os.path.exists(posts_dir):
    os.makedirs(posts_dir)


def get_batch_files(batch_files_dir):
    """Retrieve batch JSONL files, sorted and starting from a given cutoff."""
    batch_files = [os.path.join(batch_files_dir, f) for f in os.listdir(batch_files_dir)]
    return sorted(batch_files)


def process_comment_files(batch_files, posts_dir):
    """Append each comment to its corresponding post file."""
    for batch_file in batch_files:
        print("Processing:", batch_file)
        with open(batch_file, 'r', encoding='utf-8') as bf:
            for line in bf:
                try:
                    comment = json.loads(line)
                    post_id = comment.get('link_id')
                    if not post_id:
                        continue
                    post_file_path = os.path.join(posts_dir, f'{post_id[3:]}.jsonl')
                    with open(post_file_path, 'a', encoding='utf-8') as pf:
                        pf.write(json.dumps(comment) + '\n')
                except Exception as e:
                    print(f"Error processing comment in {batch_file}: {e}")

# Main execution
batch_files = get_batch_files(batch_files_dir)
process_comment_files(batch_files, posts_dir)
