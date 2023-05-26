# Natural language processing course 2022/23: `Reviving Historical Slovene: Exploring Large Language Models for Translating Modern Slovene Texts to the Past`

Team members:
 * `Martin Jurkovič`, `63180015`, `mj5835@student.uni-lj.si`
 * `Blaž Pridgar`, `63220482`, `bp59607@student.uni-lj.si`
 * `Jure Savnik`, `63170258`, `js8049@student.uni-lj.si`
 
Group public acronym/name: `MBJ`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

## Project description

This project presents a study on paraphrasing sentences from modern Slovene to old Slovene using generative large language models. The focus of the study is on fine-tuning multiple Slovene text models on a corpus of old Slovene texts from the IMP corpus. The IMP corpus, which includes slovene texts from the 16th to the 20th century, provides a valuable resource for training and evaluating the models. The study compares the performance of different models, including Slovene T5 small, Slovene T5 large, and multilingual T5, using the ROUGE metric for evaluation. The results show that the fine-tuned models achieved high accuracy, with scores above 90\% on average. The findings suggest that generative text models can effectively paraphrase modern Slovene sentences into old Slovene, opening up possibilities for language translation and historical text analysis.

## Project structure

- **data**: Contains datasets used in the project.
- **interim reports**: Stores two interim reports.
- **src**:
  - **data**:
    - **web_crawler**: Scripts for downloading books from dlib.com, which were in the end not used in the project.
    - **IMP-converter**: Scripts for processing raw XML TEI files of the IMP-corpus.
  - **model_finetuning**:
    - **notebooks**: Jupyter Notebook files for model fine-tuning.
    - **HPC_scripts**: Python scripts for model fine-tuning for HPC system.
  - **model_evaluation**: Script for testing model performance.

- **requirements.txt**: Lists project dependencies.
- **Reviving Historical Slovene: Exploring Large Language Models for Translating Modern Slovene Texts to the Past.pdf**: Project report in PDF format.
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

## How to fine-tune the models
Since T5 models are too large to finetune on personal computers, we have prepared the notebooks for google colab and corresponding python scripts for HPC system. In both cases you must copy the preprocessed dataset to either google colab or HPC, then set the location of the files in the notebook or python script.

## List of the finetuned model repos
- [Slovene T5 small](https://huggingface.co/martinjurkovic/t5-sl-small-finetuned-old-slovene)
- [Slovene T5 large](https://huggingface.co/martinjurkovic/t5-sl-large-finetuned-old-slovene-3)
- [Multilingual mT5](https://huggingface.co/martinjurkovic/mt5-base-finetuned-old-slovene)
