import os
import pandas as pd
import re
import nltk
import ast
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# NOTE: filter out team names
# NOTE: 

# ROUND 1: banned_words = {"et", "am", "pm", "football", "vs", "game", "thread", "state", "field", "stadium"}
# ROUND 2:
banned_words = {"week", "time", "team", "per", "tv", "cbs", "abc"}
stop_words = stop_words | banned_words

# NOTE: This will change for future datasets
# NOTE: add all teams to this
team_names = {"michigan", "ohiostate", "georgia", "oklahoma", "texas", "floridastate", "oregon", "alabama", "notredame", "iowa"}
state_names = state_names = {
    "alabama", "alaska", "arizona", "arkansas", "california",
    "colorado", "connecticut", "delaware", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
    "kentucky", "louisiana", "maine", "maryland", "massachusetts",
    "michigan", "minnesota", "mississippi", "missouri", "montana",
    "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming"
}

# Multi-word statenames with spaces removed: new york -> newyork
squished_state_names = {
    "newhampshire", "newjersey", "newmexico", "newyork", "northcarolina", 
    "northdakota", "rhodeisland", "southcarolina", "southdakota", "westvirginia"
}

# Not exactly state names, but words that appear often in team names
other_state_tokens = {
   "state", "carolina", "dakota", "north", "south",  "northern", "eastern", "southern", 
   "western", "central", "coastal"
}

acc_teams = {
    'Boston College',
    'Clemson',
    'Duke',
    'Florida State',
    'Georgia Tech',
    'Louisville',
    'Miami',
    'North Carolina',
    'NC State', 
    'Pittsburgh',
    'Syracuse',
    'Virginia',
    'Virginia Tech',
    'Wake Forest'
}

big_ten_teams = {
    'Illinois',
    'Indiana',
    'Iowa',
    'Maryland',
    'Michigan',
    'Michigan State',
    'Minnesota',
    'Nebraska',
    'Northwestern',
    'Ohio State',
    'Penn State',
    'Purdue',
    'Rutgers',
    'Wisconsin'
}

sec_teams = {
    'Alabama',
    'Arkansas',
    'Auburn',
    'Florida',
    'Georgia',
    'Kentucky',
    'LSU',
    'Mississippi State',
    'Missouri',
    'Ole Miss',
    'South Carolina',
    'Tennessee',
    'Texas A&M',
    'Vanderbilt'
}

big_12_teams = {
    'Baylor',
    'Iowa State',
    'Kansas',
    'Kansas State',
    'Oklahoma',
    'Oklahoma State',
    'TCU',
    'Texas',
    'Texas Tech',
    'West Virginia',
    'UCF',
    'Cincinnati',
    'Houston',
    'BYU'
}

pac_12_teams = {
    'Arizona',
    'Arizona State',
    'California',
    'Colorado',
    'Oregon',
    'Oregon State',
    'Stanford',
    'UCLA',
    'USC',
    'Utah',
    'Washington',
    'Washington State'
}

independant_teams = {
    'Notre Dame',
    'Army',
    'UMass',
    'Uconn',
}

aac_teams = {
    'SMU',
    'Tulane',
    'UTSA',
    'Memphis',
    'South Fla.',
    'Rice',
    'Navy',
    'North Texas',
    'UAB',
    'Fla. Atlantic',
    'Tulsa',
    'Charlotte',
    'Temple',
    'East Carolina'
}

cusa_teams = {
    'Liberty',
    'New Mexico St.',
    'Jacksonville St.',
    'Western Ky.',
    'Middle Tenn.',
    'Louisiana Tech',
    'Sam Houston',
    'UTEP',
    'FIU'
}

mac_teams = {
    'Miami (OH)',
    'Ohio',
    'Bowling Green',
    'Buffalo',
    'Akron',
    'Kent St.',
    'Toledo',
    'NIU',
    'Eastern Mich.',
    'Central Mich.',
    'Western Mich.',
    'Ball St.'
}

mw_teams = {
    'Boise St.',
    'San Jose St.',
    'UNLV',
    'Air Force',
    'Wyoming',
    'Fresno St.',
    'Utah St.',
    'Colorado St.',
    'Hawaii',
    'New Mexico',
    'San Diego St.',
    'Nevada'
}

sun_belt_teams = {
    'James Madison',
    'App State',
    'Coastal Carolina',
    'Old Dominion',
    'Georgia St.',
    'Ga. Southern',
    'Marshall',
    'Troy',
    'Texas St.',
    'South Alabama',
    'Arkansas St.',
    'Louisiana',
    'Southern Miss.',
    'ULM'
}

def transform_team_names(teams):
    transformed_teams = set()
    for team in teams:
        # Make all letters lowercase
        team = team.lower()
        # Remove spaces
        team = team.replace(" ", "")
        # Spell out abbreviations like "St." to "state"
        team = team.replace("st.", "state")
        transformed_teams.add(team)
    return transformed_teams

acc_teams = transform_team_names(acc_teams)
big_ten_teams = transform_team_names(big_ten_teams)
sec_teams = transform_team_names(sec_teams)
big_12_teams = transform_team_names(big_12_teams)
pac_12_teams = transform_team_names(pac_12_teams)
independant_teams = transform_team_names(independant_teams)
aac_teams = transform_team_names(aac_teams)
cusa_teams = transform_team_names(cusa_teams)
mac_teams = transform_team_names(mac_teams)
mw_teams = transform_team_names(mw_teams)
sun_belt_teams = transform_team_names(sun_belt_teams)

# Input: set of strings with spaces
# Output: set of strings with spaces removed
def remove_spaces(strings_set):
    modified_strings_set = set()
    for string in strings_set:
        modified_string = string.replace(" ", "")
        modified_strings_set.add(modified_string)
    return modified_strings_set

# Combined set of all teams
all_fbs_teams = set()
all_fbs_teams.update(acc_teams)
all_fbs_teams.update(big_ten_teams)
all_fbs_teams.update(sec_teams)
all_fbs_teams.update(big_12_teams)
all_fbs_teams.update(pac_12_teams)
all_fbs_teams.update(independant_teams)
all_fbs_teams.update(aac_teams)
all_fbs_teams.update(cusa_teams)
all_fbs_teams.update(mac_teams)
all_fbs_teams.update(mw_teams)
all_fbs_teams.update(sun_belt_teams)

stop_words.update(banned_words)
stop_words.update(team_names)
# stop_words.update(state_names)
# stop_words.update(squished_state_names)
# stop_words.update(other_state_tokens)
# stop_words.update(all_fbs_teams)

min_stop_words = stop_words
mid_stop_words = stop_words | team_names
max_stop_words = stop_words | all_fbs_teams | state_names | squished_state_names | other_state_tokens

MAX_WORD_LEN = 16


def clean_csv_files(csv_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each CSV file in the directory
    for file_name in os.listdir(csv_dir):
        if file_name.endswith(".csv"):
            # Construct the full path to the CSV file
            file_path = os.path.join(csv_dir, file_name)

            # Read CSV file into DataFrame
            #dfmin = pd.read_csv(file_path)
            #dfmid = pd.read_csv(file_path)
            dfmax = pd.read_csv(file_path)

            # For each comment, we want to get the comment author's flairs so we can remove those from the comment body text
            # Instead of using lambda function on comment body, we can apply it to the whole df row and then only actually change the 
            #   comment body field.
            # self_flairs = ast.literal_eval(dfmin["comment_author_flairs"])

            # TODO: maybe use stopwords as a global instead of passing in as a param for each of these?
            # Construct dfmin csv file
            #dfmin['comment_body'] = dfmin.apply(lambda row: clean_text_min(str(row['comment_body']), set(ast.literal_eval(row['comment_author_flairs'])) | min_stop_words), axis=1)
            #dfmin['post_title'] = dfmin['post_title'].apply(lambda x: clean_text_min(str(x), min_stop_words))
            #dfmin['post_selftext'] = dfmin['post_selftext'].apply(lambda x: clean_text_min(str(x), min_stop_words))
            
            # Construct dfmid csv file
            #dfmid['comment_body'] = dfmid['comment_body'].apply(lambda x: clean_text_mid(str(x)))
            #dfmid['post_title'] = dfmid['post_title'].apply(lambda x: clean_text_mid(str(x)))
            #dfmid['post_selftext'] = dfmid['post_selftext'].apply(lambda x: clean_text_mid(str(x)))

            # Construct dfmax csv file
            dfmax['comment_body'] = dfmax['comment_body'].apply(lambda x: clean_text_max(str(x)))
            dfmax['post_title'] = dfmax['post_title'].apply(lambda x: clean_text_max(str(x)))
            dfmax['post_selftext'] = dfmax['post_selftext'].apply(lambda x: clean_text_max(str(x)))

            # Save cleaned DataFrame to a new CSV file
            #output_file_path_min = os.path.join(output_dir, "min", file_name)
            #output_file_path_mid = os.path.join(output_dir, "mid", file_name)
            output_file_path_max = os.path.join(output_dir, file_name)

            #dfmin.to_csv(output_file_path_min, index=False)
            #dfmid.to_csv(output_file_path_mid, index=False)
            dfmax.to_csv(output_file_path_max, index=False)

            #print(f"CSV file cleaned and saved to: {output_file_path_min}")
            #print(f"CSV file cleaned and saved to: {output_file_path_mid}")
            print(f"CSV file cleaned and saved to: {output_file_path_max}")


# Remove stopwords, [TODO] self team name, long words
def clean_text_min(text, sw):
    # Remove stopwords
    words = text.split()

    filtered_words = []
    for word in words:
        if len(word) < MAX_WORD_LEN and word not in sw:
            filtered_words.append(word)

    # Join filtered words back into a single string
    text = ' '.join(filtered_words)

    return text

# TODO: call clean_text_min instead of code dupe
# Min + "top 10" team names
def clean_text_mid(text):
    # Remove stopwords
    words = text.split()

    # filtered_words = [word for word in words if word not in stop_words and len(word) < MAX_WORD_LEN]
    filtered_words = []
    for word in words:
        if len(word) < MAX_WORD_LEN and word not in mid_stop_words and not word.isnumeric():
            filtered_words.append(word)

    # Join filtered words back into a single string
    text = ' '.join(filtered_words)

    return text

# TODO: call clean_text_mid instead of code dupe
# Mid + all team names, [IN PROGRESS] all state names, numbers, [TODO] flair residuals
def clean_text_max(text):
    # Remove stopwords
    words = text.split()

    # filtered_words = [word for word in words if word not in stop_words and len(word) < MAX_WORD_LEN and not word.isnumeric()]
    filtered_words = []
    for word in words:
        if len(word) > MAX_WORD_LEN or word.isnumeric():
            continue

        if word in max_stop_words:
            continue

        # somethingfsomething
        if bool(re.match(r'\b(\w+)f\1\b', word)):
            continue
        
        # somethingfsomethingsomething
        if bool(re.match(r'\b(\w+)f\1\1\b', word)):
            continue

        filtered_words.append(word)

    # Join filtered words back into a single string
    text = ' '.join(filtered_words)

    return text

# Example usage:
csv_dir = "cleaned_combined_classdata_csv"
output_dir = "datasets/all/max/csv"

clean_csv_files(csv_dir, output_dir)
