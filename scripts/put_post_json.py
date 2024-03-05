import json
import os

# Path to the posts file
posts_file = 'arctic-shift-dataset/reduced/r_CFB_posts.jsonl'

# Directory containing the comments files
comments_dir = 'arctic-shift-dataset/depth1_th10_len10'

# Ensure the comments directory exists
if not os.path.exists(comments_dir):
    print(f"Directory {comments_dir} does not exist.")
else:
    # Open and iterate through each line (each post) in the posts file
    with open(posts_file, 'r') as pf:
        for line in pf:
            post = json.loads(line)  # Parse the JSON data
            post_id = post['id']  # Extract the post ID

            # Path to the corresponding comments file for this post
            comments_file_path = os.path.join(comments_dir, f"{post_id}.jsonl")
            
            # Check if the comments file exists for this post
            if os.path.exists(comments_file_path):
                # Read the existing comments from the file
                with open(comments_file_path, 'r') as cf:
                    comments = cf.readlines()

                # Write the post JSON object to the top of its comments and save back to the file
                with open(comments_file_path, 'w') as cf:
                    # Write the post JSON as the first line
                    cf.write(json.dumps(post) + '\n')
                    # Then write back the original comments
                    cf.writelines(comments)

print("Completed appending posts to their corresponding comments files.")
