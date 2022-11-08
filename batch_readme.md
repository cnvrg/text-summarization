Use this blueprint with a tailor-trained model to extract text summaries from custom text and Wikipedia articles. To use this blueprint, provide one ` summarization` folder in the S3 Connector with the training file containing the text for summarization and the model/tokenizer files.

This main summarization folder has two subdirectories, namely:
- `default_model` − A folder with the base model used to fine tune to customize the model
- `tokenizer` − A folder with the tokenizer files used to assist in text summarization

NOTE: This documentation uses the Wikipedia connection and subsequent summarization as an example. The users of this blueprint can select any source of text and input it to the Batch Predict task.

Complete the following steps to run the text-summarizer model in batch mode:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. Click the **S3 Connector** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: bucketname − Value: provide the data bucket name
     - Key: prefix – Value: provide the main path to the model/tokenizer folders
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Click the **Wikipedia Connector** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `topics` − Value: provide the topics to extract from Wikipedia
     - Format − use one of the following the flexible formats: comma-separated text (shown), tabular CSV, or URL link
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Batch Predict** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key:  `input_path` − Value: provide the path to the training file including the Wikipedia prefix. Ensure the path has the following format: ` /input/wikipedia_connector/wiki_output.csv`
     - Key:  `modelpath` − Value:  provide the path to the model including the S3 prefix. Ensure the path has the following format: ` /input/s3_connector/model_files/summarization/bart_large_cnn/`
     - Key:  `tokenizer` − Value:  provide the path to the tokenizer files including the S3 prefix. Ensure the path has the following format: ` /input/s3_connector/model_files/summarization/tokenizer_files/`
     NOTE: You can use prebuilt example data paths provided.
   - Click the **Advanced** tab to change resources to run the blueprint, as required. 
5. Click the **Run** button.
6. Track the blueprint’s real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.

   The cnvrg software deploys a text summarization model that summarizes custom text and Wikipedia articles and downloads CSV files containing the summaries.

7. Select **Batch Predict > Experiments > Artifacts** and locate the output CSV files.
8. Select the desired CSV File Name, click the right menu icon, and click **Open File** to view the output CSV file.

A tailored model that summarizes custom text and Wikipedia articles has now been deployed in batch mode.

Click [here](link) for this blueprint's detailed run instructions. To learn how this blueprint was created, click [here](https://github.com/cnvrg/text-summarization).
