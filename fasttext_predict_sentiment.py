import fasttext
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


class Base:
    """Base class that houses common utilities for reading in test data
    and calculating model accuracy and F1 scores.
    """
    def __init__(self) -> None:
        pass

    def read_data(self, fname: str, lower_case: bool=False,
                  colnames=['truth', 'text']) -> pd.DataFrame:
        "Read in test data into a Pandas DataFrame"
        df = pd.read_csv(fname, sep='\t', header=None, names=colnames,encoding = "ISO-8859-1")
        df['truth'] = df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        df['truth'] = df['truth'].astype(int).astype('category')
        # Optional lowercase for test data (if model was trained on lowercased text)
        if lower_case:
            df['text'] = df['text'].str.lower()
        return df

    def read_amazon_data(self, fname: str, ) -> pd.DataFrame:
        "Read in test data into a Pandas DataFrame"

        df = pd.read_csv(fname)
        return df

    def accuracy(self, df: pd.DataFrame) -> None:
        "Prediction accuracy (percentage) and F1 score"
        acc = accuracy_score(df['truth'], df['pred'])*100
        f1 = f1_score(df['truth'], df['pred'], average='macro')
        print("Accuracy: {}\nMacro F1-score: {}".format(acc, f1))

    def amazon_reviews_accuracy(self, df: pd.DataFrame) -> None:
        "Prediction accuracy (percentage) and F1 score"
        rating = [int(i[0]) for i in pd.Series(df['Star Rating'])]
        acc = accuracy_score(rating, df['Sentiment Analysis Score'])*100
        f1 = f1_score(rating, df['Sentiment Analysis Score'], average='macro')
        print("Accuracy: {}\nMacro F1-score: {}".format(acc, f1))


class FastTextSentiment(Base):
    """Predict fine-grained sentiment scores using FastText"""
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        self.model = fasttext.load_model(model_file)

    def score(self, text: str) -> int:
        # Predict just the top label (hence 1 index below)
        labels, probabilities = self.model.predict(text, 1)
        pred = int(labels[0][-1])
        return pred

    def predict(self, test_file: str, lower_case: bool) -> pd.DataFrame:
        df = self.read_data(test_file, lower_case)
        df['pred'] = df['text'].apply(self.score)
        return df

    def predict_amazon_reviews(self,test_file: str) -> pd.DataFrame:
        df = self.read_amazon_data(test_file)
        df['Sentiment Analysis Score'] = df['Customer Review'].apply(self.score)
        return df

if __name__ == "__main__":

    # Load the quantized model
    model = 'sst_quantized.ftz'

    # Initialize an object of the 'FastTextSentiment' class
    sentiment = FastTextSentiment(model)

    # OPTIONALLY:  predict the sentiments of the test test from Stanford NLP Treebank alongwith the accuracy.
    # sentiment_df = sentiment.predict('sst_test.txt',False)
    # print(sentiment_df)
    # sentiment.accuracy(sentiment_df)

    # Predict the sentiments of the Customer Reviews scraped from Amazon
    amazon_reviews_df = sentiment.predict_amazon_reviews('customer_reviews.csv')

    # Create another column to show the category name
    categories = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    amazon_reviews_df['Sentiment Category'] = [categories[i-1] for i in amazon_reviews_df['Sentiment Analysis Score']]

    # print(amazon_reviews_df)
    # Store the results in a CSV
    amazon_reviews_df.to_csv('fastText_predicted_sentiments.csv', index=False)



