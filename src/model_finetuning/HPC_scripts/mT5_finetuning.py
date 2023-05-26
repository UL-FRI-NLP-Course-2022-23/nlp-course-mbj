
"""
<a href="https://colab.research.google.com/github/UL-FRI-NLP-Course-2022-23/nlp-course-mbj/blob/main/mT5_finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# !!! mT5 needs some more libraruies: sentencepiece and accelerate !!!


# !pip show torch
# !pip install --no-cache-dir transformers sentencepiece accelerate
# ! pip install rouge-score nltk datasets

import nltk
nltk.download('punkt')

# from huggingface_hub import notebook_login
# notebook_login()

import transformers
print(transformers.__version__)

# model_checkpoint = "cjvt/t5-sl-small"
model_checkpoint = "google/mt5-base"
model_name = model_checkpoint.split("/")[-1]
hf_repo_name = f"{model_name}-finetuned-old-slovene-3"

import os
from datasets import DatasetDict, Dataset
import pandas as pd


# path for when you use uploaded files:
# directory_path = os.getcwd()

# path for files from drive:
# directory_path = "/content/drive/MyDrive/MAG-1/NLP/IMP-corpus-csv-sentence"
directory_path = "data/IMP-corpus/IMP-corpus-csv-sentence"

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



from datasets import load_metric
metric = load_metric("rouge")


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

prefix = "translate: "


max_input_length = 128
max_target_length = 128

input_param = "reg"
target_param = "orig"

def preprocess_function(examples):
    inputs = [prefix + original_text for original_text in examples[input_param]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[target_param], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)


from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


batch_size = 8
args = Seq2SeqTrainingArguments(
    hf_repo_name,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)


import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validate"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# trainer.train()


train_result = trainer.train()

trainer.save_model('/models/')

# training_args.logging_dir = 'logs' # or any dir you want to save logs

# training
train_result = trainer.train() 

# compute train results
metrics = train_result.metrics

# save train results
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# compute evaluation results
metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

# save evaluation results
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


