import wikipedia
import re
import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import argparse
from urllib.request import urlopen
import lxml
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import pathlib
import requests
import shutil

max_word_length = 20

files_model = ['config.json', 'pytorch_model.bin']

files_tokenizer = ['merges.txt', 'special_tokens_map.json',
                   'tokenizer.json', 'tokenizer_config.json', 'vocab.json']

base_folder_url_model = "https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/summarization/bart_large_cnn/"
base_folder_url_tokenizer = "https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/summarization/tokenizer_files/"


def download_model_files():
    """
    Downloads the model files if they are not already present or pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    for f in files_model:
        if not os.path.exists(current_dir + f'/{f}') and not os.path.exists('/input/train/' + f):
            print(f'Downloading file: {f}')
            response = requests.get(base_folder_url_model + f)
            f1 = os.path.join(current_dir, f)
            with open(f1, "wb") as fb:
                fb.write(response.content)
    for f2 in files_tokenizer:
        if not os.path.exists(current_dir + f'/{f2}') and not os.path.exists('/input/train/' + f2):
            print(f'Downloading file: {f2}')
            response = requests.get(base_folder_url_tokenizer + f2)
            f11 = os.path.join(current_dir, f2)
            with open(f11, "wb") as fb1:
                fb1.write(response.content)

download_model_files()

def moving_files():
    script_dir = pathlib.Path(__file__).parent.resolve()
    model_path = os.path.join(script_dir,"model/")
    os.makedirs(model_path,exist_ok=True)
    tokenizer_path = os.path.join(script_dir,"tokenizer/")
    os.makedirs(tokenizer_path,exist_ok=True)
    FILE_Model = ['config.json', 'pytorch_model.bin']

    FILE_Tokenizer = ['merges.txt', 'special_tokens_map.json',
                   'tokenizer.json', 'tokenizer_config.json', 'vocab.json']

    for loaded_file in FILE_Model:
        shutil.move(os.path.join(script_dir,loaded_file),os.path.join(model_path,loaded_file))
    for loaded_tok in FILE_Tokenizer:
        shutil.move(os.path.join(script_dir,loaded_tok),os.path.join(tokenizer_path,loaded_tok))

moving_files()

def predict_summary(text, model_cnvrg, tokenizer):

    encoder_max_length = 256
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model_cnvrg.device)
    attention_mask = inputs.attention_mask.to(model_cnvrg.device)
    min_length_1 = 0.07*len(text)
    outputs = model_cnvrg.generate(
        input_ids, attention_mask=attention_mask, max_length=500, min_length=round(min_length_1))
    outputs_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return outputs_str
# Class which contains the code to extract text from wikipedia


class WikiPage:
    def __init__(self, page):
        if page is not None:
            self.page = page
        else:
            raise ValueError(
                "Please specify a page or check if the page is valid")

    def get_wiki_page(self, page):
        ''' Uses the wikipedia library to get the content of the page specified as variable. Calls the get_clean_text() function and returns the cleaned text '''
        output = []
        flag = 0
        if ".org" in page:
            soup = BeautifulSoup(urlopen(page).read(), 'lxml')
            text = ''
            for paragraph in soup.find_all('p'):
                text += paragraph.text
            cleaned_text = self.get_clean_text(text)
            if(len(cleaned_text) < 100):
                flag = 1
        else:
            cnt = 0
            done = 0
            while done < 1:
                wik_url = wikipedia.search(page)[cnt].replace(' ', '_')
                wiki = 'https://en.wikipedia.org/wiki/'+str(wik_url)
                soup = BeautifulSoup(urlopen(wiki).read(), 'lxml')
                text = ''
                for paragraph in soup.find_all('p'):
                    text += paragraph.text
                cleaned_text = self.get_clean_text(text)
                if(len(cleaned_text) > 100):
                    done = 1
                else:
                    flag = 1
                    cnt = cnt+1

            output.append(cleaned_text)
            output.append(str(flag))
            return output

    def get_clean_text(self, text):
        text = re.sub(r'\[.*?\]+', '', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text
# function to run wikipedia class and functions on the user-input


def wikipedia_extraction(text):
    input_list = []
    list2 = []
    disambiguation = 0
    wiki_text_content = WikiPage(str(text))
    wiki_text_content1 = wiki_text_content.get_wiki_page(str(text))[0]
    input_list.append(wiki_text_content1)
    if(wiki_text_content.get_wiki_page(str(text))[1] == '1'):
        disambiguation = disambiguation + 1
    list2.append(str(text))
    return input_list[0]
# function to integrate the summarization and wikipedia functions together and output a json response

if os.path.exists("/input/train/My_Custom_Model/"):
    script_dir = pathlib.Path(__file__).parent.resolve()
    model_dir = "/input/train/My_Custom_Model/"
    tokenizer_dir = os.path.join(script_dir,"tokenizer")
else:
    print('Running Stand Alone Endpoint')
    script_dir = pathlib.Path(__file__).parent.resolve()
    model_dir = os.path.join(script_dir,'model/')
    tokenizer_dir = os.path.join(script_dir,"tokenizer/")

model_cnvrg = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

def predict(data):
    predicted_response = {}
    cnt_iter = 'prediction'
    predicted_response[cnt_iter] = []
    script_dir = pathlib.Path(__file__).parent.resolve()
    data = data['txt']
    predicted_response = {}
    response = {}
    if len(data) > max_word_length and (("wikipedia.org/wiki/" not in data) and ("https://" not in data)):
        summary_output = predict_summary(data, model_cnvrg, tokenizer)
        response["summary"] = str(summary_output[0])

    else:
        text0 = wikipedia_extraction(data)
        summary_output = predict_summary(text0, model_cnvrg, tokenizer)
        response["summary"] = str(summary_output[0])
    predicted_response[cnt_iter] = response
    return predicted_response
