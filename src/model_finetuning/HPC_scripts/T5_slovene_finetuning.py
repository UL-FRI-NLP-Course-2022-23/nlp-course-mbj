#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/UL-FRI-NLP-Course-2022-23/nlp-course-mbj/blob/main/T5_slovene_finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# If you're opening this Notebook on colab, you will probably need to install 🤗 Transformers, ROUGE score and other dependencies.

# In[ ]:


# get_ipython().system(' pip install transformers rouge-score nltk datasets')


# In[ ]:


import nltk
nltk.download('punkt')


# First you have to store your authentication token from the Hugging Face website (sign up here if you haven't already!) then execute the following cell and input your username and password:

# In[ ]:


from huggingface_hub import notebook_login, login

# login(token="")


# Then you need to install Git-LFS.

# In[ ]:


# get_ipython().system('apt install git-lfs')


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# Make sure your version of Transformers is at least 4.11.0 since the functionality was introduced in that version:

# In[ ]:


import transformers

print(transformers.__version__)


# # Fine-tuning a model on a translation/paraphrasing task

# Choose which model checkpoint to use

# In[ ]:


model_checkpoint = "cjvt/t5-sl-small"
# model_checkpoint = "cjvt/t5-sl-large"
model_name = model_checkpoint.split("/")[-1]
repo_name = f"{model_name}-finetuned-old-slovene"


# ## Loading the dataset

# In[ ]:


import os
from datasets import DatasetDict, Dataset
import pandas as pd
# %%
# path for when you use uploaded files:
# directory_path = os.getcwd()

# path for files from drive:
directory_path = "/data/IMP-corpus/IMP-corpus-csv-sentence"


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


# In[ ]:


dataset_dict


# In[ ]:


print(dataset_dict['train'][0])


# The metric is an instance of [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric):

# In[ ]:


from datasets import load_metric
metric = load_metric("rouge")
metric


# ## Preprocessing data

# Before we can feed those texts to our model, we need to preprocess them. This is done by a 🤗 Transformers Tokenizer which will (as the name indicates) tokenize the inputs (including converting the tokens to their corresponding IDs in the pretrained vocabulary) and put it in a format the model expects, as well as generate the other inputs that the model requires.
# 
# To do all of this, we instantiate our tokenizer with the AutoTokenizer.from_pretrained method, which will ensure:
# 
# we get a tokenizer that corresponds to the model architecture we want to use,
# we download the vocabulary used when pretraining this specific checkpoint.
# That vocabulary will be cached, so it's not downloaded again the next time we run the cell.

# In[ ]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# You can directly call this tokenizer on one sentence or a list of sentences:

# In[ ]:


tokenizer("Pozdrav, to je ena poved!")


# In[ ]:


tokenizer(["Pozdrav, to je prva poved.","To pa je druga."])


# To prepare the targets for our model, we need to tokenize them inside the as_target_tokenizer context manager. This will make sure the tokenizer uses the special tokens corresponding to the targets:

# In[ ]:


with tokenizer.as_target_tokenizer():
    print(tokenizer(["Pozdrav, to je prva poved.","To pa je druga."]))


# If you are using one of the five T5 checkpoints we have to prefix the inputs with "summarize:" (the model can also translate and it needs the prefix to know which task it has to perform).

# In[ ]:


if model_checkpoint in ["cjvt/t5-sl-small", "cjvt/t5-sl-large", "t5-small", "t5-large"]:
    prefix = "translate: "
else:
    prefix = ""


# We can then write the function that will preprocess our samples. We just feed them to the `tokenizer` with the argument `truncation=True`. This will ensure that an input longer that what the model selected can handle will be truncated to the maximum length accepted by the model. The padding will be dealt with later on (in a data collator) so we pad examples to the longest length in the batch and not the whole dataset.

# In[ ]:


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


# This function works with one or several examples. In the case of several examples, the tokenizer will return a list of lists for each key:

# In[ ]:


preprocess_function(dataset_dict['train'][:2])


# To apply this function on all the pairs of sentences in our dataset, we just use the map method of our dataset object we created earlier. This will apply the function on all the elements of all the splits in dataset, so our training, validation and testing data will be preprocessed in one single command.

# In[ ]:


tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)


# Even better, the results are automatically cached by the 🤗 Datasets library to avoid spending time on this step the next time you run your notebook. The 🤗 Datasets library is normally smart enough to detect when the function you pass to map has changed (and thus requires to not use the cache data). For instance, it will properly detect if you change the task in the first cell and rerun the notebook. 🤗 Datasets warns you when it uses cached files, you can pass `load_from_cache_file=False` in the call to `map` to not use the cached files and force the preprocessing to be applied again.
# 
# Note that we passed `batched=True` to encode the texts by batches together. This is to leverage the full benefit of the fast tokenizer we loaded earlier, which will use multi-threading to treat the texts in a batch concurrently.

# ## Fine Tuning the model

# Now that our data is ready, we can download the pretrained model and fine-tune it. Since our task is of the sequence-to-sequence kind, we use the AutoModelForSeq2SeqLM class. Like with the tokenizer, the from_pretrained method will download and cache the model for us.

# In[ ]:


from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# model = AutoModelForSeq2SeqLM.from_pretrained("cjvt/t5-sl-small")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# To instantiate a `Seq2SeqTrainer`, we will need to define three more things. The most important is the [`Seq2SeqTrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Seq2SeqTrainingArguments), which is a class that contains all the attributes to customize the training. It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional:

# In[ ]:


batch_size = 16
args = Seq2SeqTrainingArguments(
    repo_name,
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


# Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the batch_size defined at the top of the cell and customize the weight decay. Since the Seq2SeqTrainer will save the model regularly and our dataset is quite large, we tell it to make three saves maximum. Lastly, we use the predict_with_generate option (to properly generate translations) and activate mixed precision training (to go a bit faster).
# 
# The last argument to setup everything so we can push the model to the Hub regularly during training. Remove it if you didn't follow the installation steps at the top of the notebook. If you want to save your model locally in a name that is different than the name of the repository it will be pushed, or if you want to push your model under an organization and not your name space, use the hub_model_id argument to set the repo name (it needs to be the full name, including your namespace: for instance "sgugger/t5-finetuned-xsum" or "huggingface/t5-finetuned-xsum").
# 
# Then, we need a special kind of data collator, which will not only pad the inputs to the maximum length in the batch, but also the labels:

# In[ ]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# The last thing to define for our Seq2SeqTrainer is how to compute the metrics from the predictions. We need to define a function for this, which will just use the metric we loaded earlier, and we have to do a bit of pre-processing to decode the predictions into texts:

# In[ ]:


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


# Then we just need to pass all of this along with our datasets to the `Seq2SeqTrainer`:

# In[ ]:


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validate"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# We can now finetune our model by just calling the `train` method:

# In[ ]:


train_result = trainer.train()

# trainer.save_pretrained('/d/hpc/projects/FRI/mj5835/models/')
trainer.save_model('models/')

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

# ## TESTING THE MODEL

# ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a metric most commonly used for the evaluation of automatically generated text summaries. It measures the quality of a summary by the number of overlapping units (n-grams, sequences of texts, etc.) between summaries created by humans and summaries created by summarization systems. ROUGE is not a single metric but a family of metrics. The most commonly used are ROUGE-N and ROUGE-L. The first measures the overlapping of n-grams (typically unigrams and bigrams), while the second measures the longest common subsequence found in both summaries.

# In[ ]:


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained(f"martinjurkovic/{model_checkpoint}-finetuned-old-slovene")

# model = AutoModelForSeq2SeqLM.from_pretrained(f"martinjurkovic/{model_checkpoint}-finetuned-old-slovene")

