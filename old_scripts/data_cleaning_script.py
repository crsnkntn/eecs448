import json


###### CLEAN THE COMMENTS ######

desired_fields = ["author", "author_fullname", "author_flair_richtext", "author_flair_text", "author_is_blocked", "body", "controversiality", "id", "link_id", "name", "parent_id", "permalink", "score", "ups"]

# The path to your original JSONL file
input_file_path = 'arctic-shift-dataset/original/r_CFB_comments.jsonl'

# The path where you want to save the new JSONL file
output_file_path = 'arctic-shift-dataset/reduced/r_CFB_comments.jsonl'

with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        # Load the JSON object from the current line
        json_obj = json.loads(line)
        
        # Create a new dictionary with only the desired fields
        filtered_json_obj = {field: json_obj[field] for field in desired_fields if field in json_obj}
        
        # Write the filtered JSON object to the new file
        output_file.write(json.dumps(filtered_json_obj) + '\n')


###### CLEAN THE POSTS ######

desired_fields = ["author", "author_fullname", "author_flair_richtext", "author_flair_text", "author_is_blocked", "id", "name", "permalink", "score", "selftext", "title", "ups", "upvote_ratio", "url"]

# The path to your original JSONL file
input_file_path = 'arctic-shift-dataset/original/r_CFB_posts.jsonl'

# The path where you want to save the new JSONL file
output_file_path = 'arctic-shift-dataset/reduced/r_CFB_posts.jsonl'

with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        # Load the JSON object from the current line
        json_obj = json.loads(line)
        
        # Create a new dictionary with only the desired fields
        filtered_json_obj = {field: json_obj[field] for field in desired_fields if field in json_obj}
        
        # Write the filtered JSON object to the new file
        output_file.write(json.dumps(filtered_json_obj) + '\n')
