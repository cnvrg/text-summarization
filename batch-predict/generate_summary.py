# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import torch
import numpy as np
import datasets
import transformers
from transformers import Trainer, TrainingArguments
import nltk
from pyarrow import csv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import os

parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument(
    "--input_path",
    action="store",
    dest="input_path",
    default="/input/wikipedia_connector/cnvrg/wiki_output.csv",
    required=False,
    help="""name of the file containing the wikipedia output""",
)
parser.add_argument(
    "--modelpath",
    action="store",
    dest="modelpath",
    default="",
    required=False,
    help="""model through which the summaries will be generated""",
)
parser.add_argument(
    "--tokenizer",
    action="store",
    dest="tokenizer",
    default="/data/summarization-dataset/Tokenizer/",
    required=True,
    help="""default tokenizer to use""",
)
parser.add_argument(
    "--min_percent",
    action="store",
    dest="min_percent",
    default="0.07",
    required=False,
    help="""ratio of minimum length of the summary""",
)
parser.add_argument(
    "--encoder_max_length",
    action="store",
    dest="encoder_max_length",
    default="256",
    required=True,
    help="""hyperparamter while training""",
)
# Extracting values from Arguments
args = parser.parse_args()
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
language = "english"
address_model_cnvrg = args.modelpath
tokenizer_path = args.tokenizer
rows_cnt = pd.read_csv(args.input_path).shape[0]
sub1 = "train[:" + str(rows_cnt) + "]"
model_cnvrg = AutoModelForSeq2SeqLM.from_pretrained(address_model_cnvrg)
output_file_name = 'summaries.csv'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
min_percent = float(args.min_percent)
encoder_max_length = int(args.encoder_max_length)
# This function takes a single string as input and breaks it up into multiple strings each of which has a length less than the limit set. The strings are broken down at full stops closest to the the limit set.
limit = 512
def breakup(input_text):
    # add full stop at the end of the text if not already present to mark end
    if input_text[-1] != ".":
        input_text += "."
    encoded_input = tokenizer(input_text)# encode text to get total token size
    process = []
    to_loop = (len(encoded_input["input_ids"]) // limit + 1)#optimum token ratio
    for i in range(to_loop):
        breakup = tokenizer.decode(
            encoded_input["input_ids"][:limit]
        )  # convert first 512 tokens to raw text.
        end_sentence = breakup.rfind(".")# last full stop(end of last sentence)
        if end_sentence != -1:
            process.append(
                breakup[0 : end_sentence + 1]
            )  # break at the last complete sentence and add it to the list
            input_text = input_text[end_sentence + 1 :]  # remaining raw text
            encoded_input = tokenizer(input_text)  # convert into tokens again
        else:
            process.append(
                breakup
            )  # if full stop not found add the entire text to the list
            input_text = input_text[len(breakup) :]  # remaining raw text
            encoded_input = tokenizer(input_text)  # convert into tokens again
    return process

# Function which generates summaries from text and the modelpath
def generate_summary(test_samples, model):
    outputs_1 = []
    outputs_str_1 = []
    for i in range(len(test_samples)):
        inputs = tokenizer(
            test_samples["text"][i],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        # defining minimum length of summaries
        min_length_1 = min_percent * len(test_samples["text"][i])

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=500,
            min_length=round(min_length_1),
        )
        # decoding output
        outputs_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs_1.append(outputs)
        outputs_str_1.append(outputs_str)

    return outputs_1, outputs_str_1


print("defined_generate function")


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["text"], batch["summary"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )
    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

# Splitting the input into multiple paragraphs
cnt=0
split_frame = pd.DataFrame(columns=['broken_text','title'])
input_file = pd.read_csv(args.input_path)
for i in range(input_file.shape[0]):
    text = input_file['text'][i]
    broken_text = breakup(text)
    for j in len(broken_text):
        j = j+cnt
        split_frame.add[j,'broken_text'] = broken_text[j]
        split_frame.add[j,'title'] = df['title'][i]
    cnt=j
split_csv_path = cnvrg_workdir+'split_input.csv'
split_frame.to_csv(split_csv_path)

# running the function
#input_doc = datasets.load_dataset(
#    "csv", data_files=args.input_path, split=(str(sub1)))
input_doc = datasets.load_dataset(
    "csv", data_files=split_csv_path, split=(str(sub1)))

#smaller_text = breakup(input_doc)
summaries_case_0 = generate_summary(input_doc, model_cnvrg)[1]

summaries_generated = pd.DataFrame(
    summaries_case_0, columns=["Generated_Summary"])
# Concatenatnig summaries
summaries_generated['title'] = list(split_frame['title'])
summaries_generated['Summary'] = summaries_generated.groupby(['title'])['Generated_Summary'].transform(lambda x: ','.join(x))
shra[['title','Summary']].drop_duplicates()

summaries_generated.to_csv(
    cnvrg_workdir+"/{}".format(output_file_name), index=False)
