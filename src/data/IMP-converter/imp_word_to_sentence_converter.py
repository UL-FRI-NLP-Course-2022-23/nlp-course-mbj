# %%
import csv
import os
from tqdm import tqdm

# get working directory and append to file path
cwd = os.getcwd()
cwd_project = cwd.split('nlp-course-mbj')[0] + 'nlp-course-mbj'
input_dir = cwd_project + '/data/IMP-corpus/IMP-corpus-csv-word/'
output_dir = cwd_project + '/data/IMP-corpus/IMP-corpus-csv-sentence/'

def join_list_to_string_separator(list):
    # join the list into string, with separators not having space before them
    string = ""
    for i in range(len(list)):
        if list[i] in ["„", "“", "’", "‘", "«", "»", "-", "”", "(", ")", "[", "]", "{", "}", "<", ">", "–", "—", "…", "—"]:
            continue
        elif list[i] in [".", "?", "!", ",", ";", ":"]:
            string += list[i]
        else:
            string += " " + list[i]
    return string.strip()

def percent_of_matched_words(list1, list2):
    # get the percent of matched words between two lists
    matched_words = 0
    for i in range(len(list1)):
        if list1[i].lower() == list2[i].lower():
            matched_words += 1
    return matched_words / len(list1)

# %%
num_files = len(os.listdir(input_dir))
for file_name in tqdm(os.listdir(input_dir), total=num_files, desc="Processing files"):
    # only process .csv files
    if not file_name.endswith('.csv'):
        continue

    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the CSV file for writing
    with open(os.path.join(output_dir, f'{file_name}'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # Write the header row
        writer.writerow(['orig', 'reg', 'lemma'])

        # read csv file line by line
        with open(os.path.join(input_dir, file_name), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')

            # skip header row
            next(reader)
            
            # for each line, read the orig, reg, and lemma columns, then append each to a list, if the element is a "." write list to csv
            orig_list = []
            reg_list = []
            lemma_list = []
            for row in reader:
                orig_list.append(row[0])
                reg_list.append(row[1])
                lemma_list.append(row[2])
                last_char = row[0][-1]
                if last_char in [".", "?", "!"]:
                    # join the lists into strings with last and before last element without space and write to csv
                    orig_string = join_list_to_string_separator(orig_list)
                    reg_string = join_list_to_string_separator(reg_list)
                    if percent_of_matched_words(orig_list, reg_list) == 1:
                        orig_list = []
                        reg_list = []
                        lemma_list = []
                        continue
                    lemma_string = join_list_to_string_separator(lemma_list)
                    writer.writerow([orig_string, reg_string, lemma_string])
                    orig_list = []
                    reg_list = []
                    lemma_list = []
