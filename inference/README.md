# Generating the summary of a single articles via a fine-tuned huggingface model
## Summarization Inference

This library serves as a tool for getting abstractive summarization of a single English article at a time, without training the model further or the summary of any article from Wikipedia, via its keywords. It uses a specific model trained by CNVRG on custom data (wiki_lingua dataset) and gives summaries of around 7% of the total article size. While running this library, the user needs to give the following parameter: -
## Arguments
- `--data` refers to the paragraph, which needs to be summarized. Be careful to not introduce any apostrophes or other punctuation marks in the paragraph. It can also refer to a topic that you want the wikipedia summary of.

### Output
- Summary Text

### Model Used
is a fine-tuned version of [bart_large_cnn](https://huggingface.co/facebook/bart-large-cnn) from [AutoModelSeq2Seq](https://huggingface.co/transformers/model_doc/encoderdecoder.html) class of [transformers](https://huggingface.co/transformers/) library. The function used is model.generate() and the summary length is restricted to 500 words as well and is always higher than 7% of the article length.
BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.
BART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs.

### Fine-tuning Code (using the Trainer API)
https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html
https://huggingface.co/course/chapter3/3?fw=pt

### Techniques/Libraries Used
[Beautiful Soup](https://beautiful-soup-4.readthedocs.io/en/latest/)
Beautiful Soup is a Python library for pulling data out of HTML and XML files. It works with your favorite parser to provide idiomatic ways of navigating, searching, and modifying the parse tree. It commonly saves programmers hours or days of work.
[Regular Expressions](https://www.w3schools.com/python/python_regex.asp)
A RegEx, or Regular Expression, is a sequence of characters that forms a search pattern.
RegEx can be used to check if a string contains the specified search pattern.
[Wikipedia](https://pypi.org/project/wikipedia/)
Wikipedia is a Python library that makes it easy to access and parse data from Wikipedia.
Search Wikipedia, get article summaries, get data like links and images from a page, and more. Wikipedia wraps the MediaWiki API so you can focus on using Wikipedia data, not getting it.
