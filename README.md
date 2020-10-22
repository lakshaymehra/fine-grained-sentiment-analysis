# Predicting Fine-Grained Sentiments For Scraped Amazon Reviews Using SVM and FastText Models Trained On Stanford NLP Treebank :


## Usage Guide:

### File Descriptions:

* ##### 'amazon_review.py' :
    contains the code to scrape Amazon Reviews.
* ##### 'stanford_sentiment_treebank_exploratory_data_analysis.py' :
    contains the code to generate train ('sst_train.txt'), dev ('sst_dev.txt'), and test ('sst_test.txt') files and perform EDA.
* ##### 'svm_train_and_predict.py' :
    contains the code to train and predict using SVM model and store as a CSV ('svm_predicted_sentiments.csv').
* ##### 'train_fasttext_sentiment_analysis.py' :
    contains the code to train FastText model and store non-quantized('sst.bin') as well as quantized('sst_quantized.ftz') models.
* ##### 'fasttext_predict_sentiment.py' :
    contains the code to predict sentiments using FastText for Amazon Reviews and store as a CSV ('fastText_predicted_sentiments.csv').
* ##### 'visualize_results.py' :
    contains the code to Visualize Results.
* ##### 'customer_reviews.csv' : 
    contains the Scraped Reviews.
* ##### 'svm_predicted_sentiments.csv' :
    contains the sentiments predicted using SVM.
* ##### 'fastText_predicted_sentiments.csv' :
    contains the sentiments predicted using FastText.

#### Run 'amazon_review.py' at Command Line using Scrapy Runspider:

Run this code from the command line to run 'amazon_review.py' and store results as 'customer_reviews.csv'
```bash
scrapy runspider amazon_review.py -o customer_reviews.csv
```

#### Training and Testing SVM Model:
Use the following code to train and predict using SVM Model.

```bash
python svm_train_and_predict.py
```

After running this file, there is a 'svm_predicted_sentiments.csv' file generated containing the predicted sentiments.

#### Training FastText:
Use the following code to train FastText Model. It takes around 3-5 minutes on CPU to complete training.

```bash
python train_fasttext_sentiment_analysis.py
```

After training, there will be a model saved as 'sst.bin' and a quantized model saved as 'sst_quantized.ftz'.

#### Testing Trained Model:

Use the following code to test the quantized FastText Model.

```bash
python fasttext_predict_sentiment.py
```
This code will output a 'fastText_predicted_sentiments.csv' file containing the predicted sentiments.


## Results:

### WordCloud For Amazon Reviews:
![Alt text](Figure_7.png?raw=true "Figure_7")

### Fasttext Model Results:
![Alt text](Figure_1.png?raw=true "Figure_1")
![Alt text](Figure_2.png?raw=true "Figure_2")
![Alt text](Figure_3.png?raw=true "Figure_3")

##### Accuracy: 34.25531914893617%
##### Macro F1-score: 0.28648371805100803

### Support Vector Machine(SVM) Model Results:
![Alt text](Figure_4.png?raw=true "Figure_4")
![Alt text](Figure_5.png?raw=true "Figure_5")
![Alt text](Figure_6.png?raw=true "Figure_6")

##### Accuracy: 40.0% 
##### Macro F1-score: 0.3325577295787083

## Thus, it is clearly evident that SVM outperforms FastText for the test dataset containing reviews scraped from Amazon!


### Contact :
For any query/feedback, please contact:
```
Lakshay Mehra: mehralakshay2@gmail.com
```

