# %%
import csv
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

# get working directory and append to file path
cwd = os.getcwd()
cwd_project = cwd.split('nlp-course-mbj')[0] + 'nlp-course-mbj'
input_dir = cwd_project + '/data/IMP-corpus/IMP-corpus-tei/'
output_dir = cwd_project + '/data/IMP-corpus/IMP-corpus-csv-word/'

# %%
num_files = len(os.listdir(input_dir))
for file_name in tqdm(os.listdir(input_dir), total=num_files, desc="Processing files"):
    # Only process XML files
    if not file_name.endswith('.xml'):
        continue

    # Load the XML file into Beautiful Soup
    with open(os.path.join(input_dir, file_name), 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')

    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the CSV file for writing
    with open(os.path.join(output_dir, f'{file_name.split(".xml")[0]}.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # Write the header row
        writer.writerow(['orig', 'reg', 'lemma'])

        # Process the <choice> tags in the <body> tag
        in_choice = False
        body = soup.find('body')
        if body is not None:
            reg_text = ''
            for elem in body.descendants:
                # If we're inside a <choice> tag, extract the data from the <w> tags
                if elem.name == 'choice':
                    in_choice = True
                    orig_w = elem.select_one('orig w')
                    orig_text = orig_w.get_text().strip() if orig_w is not None else ''

                    reg_w = elem.select_one('reg w')
                    reg_text = reg_w.get_text().strip() if reg_w is not None else ''
                    reg_lemma = reg_w.get('lemma', '').strip() if reg_w is not None and reg_w.has_attr('lemma') else ''
                    writer.writerow([orig_text, reg_text, reg_lemma])
                # If we're not inside a <choice> tag, skip over <w> tags
                elif elem.name == 'w' and not in_choice and elem.has_attr('lemma'):
                    w_text = elem.get_text().strip()
                    w_lemma = elem.get('lemma', '').strip()
                    if w_text.lower() != reg_text.lower():
                        writer.writerow([w_text, w_text, w_lemma])
                # If we encounter a non-<choice> tag, we're no longer inside a <choice> tag
                elif elem.name != 'choice':
                    in_choice = False
                if elem.name == 'pc':
                    pc_text = elem.get_text().strip()
                    writer.writerow([pc_text, pc_text, ''])

