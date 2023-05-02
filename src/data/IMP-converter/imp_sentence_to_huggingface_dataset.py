# %%
import os
from datasets import DatasetDict, Dataset
import pandas as pd
# %%
cwd = os.getcwd()
cwd_project = cwd.split('nlp-course-mbj')[0] + 'nlp-course-mbj'
directory_path = cwd_project + '/data/IMP-corpus/IMP-corpus-csv-sentence/'

# Read all .csv files in the directory into a list of pandas DataFrames
dfs = []
for file in os.listdir(directory_path):
    if file.endswith(".csv"):
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Concatenate all DataFrames into one combined DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Convert the combined DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(combined_df)

# Split the dataset into train, validate, and test sets
train_dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.7)))
valid_dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.7), int(len(dataset) * 0.85)))
test_dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.85), len(dataset)))

# Create a DatasetDict object with train, validate, and test sets
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validate': valid_dataset,
    'test': test_dataset
})

# Access the individual datasets using dictionary-like syntax
train_dataset = dataset_dict['train']
valid_dataset = dataset_dict['validate']
test_dataset = dataset_dict['test']
# %%
# count unique tokens in the dataset
token_list = set()
for index, row in combined_df.iterrows():
    for token in row['orig'].split():
        token_list.add(token.lower())
    # token_list.append(row['orig'].split())

# %%
