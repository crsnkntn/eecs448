import ast
import matplotlib.pyplot as plt

# Step 1: Read the data from the file
file_path = 'frequencies.txt'  # Update this to your actual file path
with open(file_path, 'r') as file:
    data_str = file.read()
    data = ast.literal_eval(data_str)  # Parses the string as a dictionary

# Step 2: Calculate the distribution of frequency intervals
frequency_values = [freq for freq in data.values() if freq > 1]
max_frequency = max(frequency_values)
interval_range = 10  # Define the range of each frequency interval; adjust as needed
intervals = [i for i in range(0, max_frequency + interval_range, interval_range)]
frequency_distribution = {i: 0 for i in range(len(intervals)-1)}

for value in frequency_values:
    for i in range(len(intervals)-1):
        if intervals[i] <= value < intervals[i+1]:
            frequency_distribution[i] += 1
            break

# Step 3: Plot the distribution
plt.bar(frequency_distribution.keys(), frequency_distribution.values(), tick_label=[f"{intervals[i]}-{intervals[i+1]-1}" for i in range(len(intervals)-1)])
plt.xlabel('Frequency Intervals')
plt.ylabel('Number of Occurrences')
plt.title('Distribution of Frequency Intervals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
