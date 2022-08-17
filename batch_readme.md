You can use this blueprint to extract summaries out of wikipedia articles and custom text using a custom-trained model.
In order to train this model with your data, you would need to provide one folder located in s3:
- summarization: the super folder where the training file, file containing text to be summarized (optional) and the model/tokenizer files are kept.It has 2 sub-directories namely:
    - default_model: A folder with the base model you want to fine tune, to get your custom model
    - tokenizer: A folder with the tokenizer files that are used to assist in text summarization

1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:

   In the `S3 Connector` task:
    * Under the `bucketname` parameter provide the bucket name of the data
    * Under the `prefix` parameter provide the main path to where the training file and model/tokenizer folders are located

   In the `Batch Predict` task:
    *  Under the `input_path` parameter provide the path to the training file including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/input_file.csv`
    *  Under the `modelpath` parameter provide the path to the base model including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/bart_large_cnn`
    *  Under the `tokenizer` parameter provide the path to the tokenizer files including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/tokenizer_files`

**NOTE**: You can use prebuilt data examples paths that are already provided

4. Click on the 'Run Flow' button
5. In a few minutes you will deploy a new text summarization batch model.
6. Go to the 'Serving' tab in the project and look for your endpoint
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have deployed a custom model that summarizes custom text and wikipedia articles.

[See here how we created this blueprint](https://github.com/cnvrg/text-summarization)

This blueprint the following libraries
1. **Wikipedia Connector**
2. **Text Summarization Prediction**

## Wikipedia Connector
This library serves as a tool for getting the parsed text from wikipedia articles in a csv format.
## Text Summarization Prediction
This library serves as a tool for getting abstractive summarization of English articles without training the model further. It uses a specific model trained by CNVRG on custom data (wiki_lingua dataset) and gives summaries of around 7% of the total article size. While running this library, the user needs to give the following parameters: -
## What can you expect?
- abstractive summaryof any article

## What you need to provide?
- "input file" containing text or wikipedia output
    | document| 
    | - |
    |The first part of the game is set in the small Sword Coast village of West Harbor, which was the site of a battle between an evil host led by an entity known as the "King of Shadows"|
- hyper paramters
    'encoder_max_length' : '256'

