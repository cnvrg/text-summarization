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
# Function to generate summaries


def predict_1(text, model_cnvrg, tokenizer):
    
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
            if(len(cleaned_text) < 50):
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
                if(len(cleaned_text) > 50):
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


def wikipedia1(text):
    input_list = []
    list2 = []
    disambiguation = 0
    abc1 = WikiPage(str(text))
    abc11 = abc1.get_wiki_page(str(text))[0]
    input_list.append(abc11)
    if(abc1.get_wiki_page(str(text))[1] == '1'):
        disambiguation = disambiguation + 1
    list2.append(str(text))
    return input_list[0]
# function to integrate the summarization and wikipedia functions together and output a json response


def predict(data):
    script_dir = pathlib.Path(__file__).parent.resolve()

    data = data['txt']

    address_model_cnvrg = os.path.join(script_dir, 'model/')
    model_cnvrg = AutoModelForSeq2SeqLM.from_pretrained(address_model_cnvrg)
    tokenizer_address = os.path.join(script_dir, 'tokenizer/')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_address)
    response = {}
    if len(data) > 5:
        summary_output = predict_1(data, model_cnvrg, tokenizer)
        response["summary"] = str(summary_output[0])

    else:
        text0 = wikipedia1(data)
        summary_output = predict_1(text0, model_cnvrg, tokenizer)
        response["summary"] = str(summary_output[0])
    return response

