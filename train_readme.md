Use this blueprint to train a custom model that can summarize Wikipedia articles and custom textual paragraphs to short sentences using the Bert model. This blueprint also establishes an endpoint that can be used to summarize paragraphs based on the newly trained model.

To train this model with your data, create a summarization folder in the S3 Connector that comprises the training file in CSV format containing text to be summarized (optional). Also, include two subdirectories that contain the model and tokenizer files, namely:
- `default_model` − A folder with the base model to fine-tune to obtain the custom model
- `tokenizer` − A folder with the tokenizer files to use to assist in text summarization

NOTE: This documentation uses the Wikipedia connection and subsequent summarization as an example. The users of this blueprint can select any source of text and input it to the Batch Predict task.

Complete the following steps to train the text-summarizer model:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **S3 Connector** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `bucketname` − Value: enter the data bucket name
     * Key: `prefix` − Value: provide the main path to the images folder
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Return to the flow and click the **Train** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `training_file` − Value: provide the path to the CSV file including the S3 prefix in the following format: `/input/s3_connector/<prefix>/wiki_lingua_file.csv`
     * Key: `default_model` − Value: provide the path to the base model including the S3 prefix in the following format: `/input/s3_connector/<prefix>/bart_large_cnn`
     * Key: `tokenizer` − Value: provide the path to the tokenizer files including the S3 prefix in the following format: `/input/s3_connector/<prefix>/tokenizer_files`
     NOTE: You can use prebuilt data examples paths already provided.
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
     ![Train Advanced](../images/blueprints/text-summarization-train-train-advanced.png)
4. Click the **Wikipedia** Connector task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `topics` − Value: provide the topics to extract from Wikipedia
     * Format − use one of the following three flexible formats: comma-separated text, tabular CSV, or URL link
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
5. Click the **Batch Predict** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `input_path` − Value: provide the path to the Wikipedia Connector’s output CSV file in the following format: `/input/wikipedia_connector/wiki_ output.csv`
     * Key: `modelpath` − Value^: provide the path to the Train task’s custom model in the following format: `/input/train/my_custom_model`
     * Key: `tokenizer` − Value: provide the path to the S3 Connector’s tokenizer files including in the following format: `/input/s3_connector/model_files`
     ^NOTE: The Value for the `modelpath` Key can also point to a user-defined model, not just the one trained in the Train task.
     NOTE: You can use prebuilt data examples paths provided.
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
6. Click the **Run** button. The cnvrg software launches the training blueprint as set of experiments, generating a trained text-summarizer model and deploying it as a new API endpoint.
7. Track the blueprint's real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
8. Click the **Serving** tab in the project, locate your endpoint, and complete one or both of the following options:
   * Use the Try it Live section with any text to check the model’s ability to summarize.
   * Use the bottom integration panel to integrate your API with your code by copying in your code snippet.

A custom model and an API endpoint, which can summarize text, have now been trained and deployed. If using the Batch Predict task, a custom text-summarizer model has now been deployed in batch mode.

Click [here](link) for this blueprint's detailed run instructions. To learn how this blueprint was created, click [here](https://github.com/cnvrg/text-summarization).
