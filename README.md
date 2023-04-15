# Natural language processing course 2022/23: `Put name of your project here`

Team members:
 * `Martin Jurkovič`, `63180015`, `mj5835@student.uni-lj.si`
 * `Blaž Pridgar`, `63220482`, `bp59607@student.uni-lj.si`
 * `Jure Savnik`, `63170258`, `js8049@student.uni-lj.si`
 
Group public acronym/name: `MBJ`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

## Project description

`Put your project description here`

## Project structure

`Put your project structure here`

## How to download and prepare the IMP dataset
Go to `data/IMP-corpus` and follow the instructions in the `README.md` file.

Then run the following python scripts to convert the dataset from TEI to csv of words:
```bash
python src/data/IMP-converter/imp_converter.py
```

Then convert from word csv to sentence csv:
```bash
python src/data/IMP-converter/imp_word_to_sentence_converter.py
```

Then you can create a huggingface DatasetDict from the sentence csv in `src/data/IMP-converter/imp_sentence_to_huggingface_dataset.py`.