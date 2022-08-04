# Wikipedia Connector (Library)
Wikipedia connector is a library that creates a connection to wikipedia from your system and extracts the text from any article without any special characters and tags, in the form of a csv file.
## Input Arguments
- `topics`: It is a list of topics that the user wants to extract from wikipedia. It can be in either a comma separated text, or present as rows in a tabluar format in a csv file whose name can be given in this argument
    **Default Value -** <"Topza,Gun,Republic,Donald"
    **Default Value -** <"/data/dataset_name/input_file.csv"
## Features
- input format is flexible, as it can be provided as a text or in a tabluar format or even as a list of columns in a table
- some topics are ambigious in the sense that a search for that particular keyword doesn't yield any specific page. In that case, the first 5 of the disambiguation page values are circled through one by one
- HTML tags, non-english words and other unncessary punctuation is removed leaving just English words
- the text is outputted in a csv format

## Model Artifacts
- `--wiki_output.csv` output file which contains the text in the following format
text 	title 

| text           | title  |
|--------------------|-------------------------|
| Predation is a biological interaction where one organism, the predator, kills and eats another organism, its prey. It is one of a family of common feeding behaviours that includ | predator     |
| Halo is a military science fiction video game and media franchise created by Bungie. The franchise is currently managed and developed by 343 Industries, and owned and published by Xbox Game Studios.| halo    |

## How to run
```
python3 wiki/training/wiki.py --topics "predator,zaineb,knuckles,dust" 
```
### Techniques/Libraries Used
[Beautiful Soup](https://beautiful-soup-4.readthedocs.io/en/latest/)
Beautiful Soup is a Python library for pulling data out of HTML and XML files. It works with your favorite parser to provide idiomatic ways of navigating, searching, and modifying the parse tree. It commonly saves programmers hours or days of work.
[Regular Expressions](https://www.w3schools.com/python/python_regex.asp)
A RegEx, or Regular Expression, is a sequence of characters that forms a search pattern.
RegEx can be used to check if a string contains the specified search pattern.
[Wikipedia](https://pypi.org/project/wikipedia/)
Wikipedia is a Python library that makes it easy to access and parse data from Wikipedia.
Search Wikipedia, get article summaries, get data like links and images from a page, and more. Wikipedia wraps the MediaWiki API so you can focus on using Wikipedia data, not getting it.