import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import argparse



def generate_wordcloud(df):

    ''' Create a wordcloud using Customer Reviews to see most commonly used words'''

    comment_words = ''
    stopwords = set(STOPWORDS)

    # iterate through the csv file
    for val in df['Customer Review']:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens) + " "

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()

def star_ratings_bar_chart(df,rating):
    # Generate Horizontal Bar Chart for the Number of Reviews vs Star Rating Given By The Customer

    ax = rating.value_counts(sort=False).plot(kind='barh',color = 'black')
    ax.set_xlabel('Number of Amazon Reviews')
    ax.set_ylabel('Star Rating Given By The Customer')
    plt.show()

def predicted_sentiments_bar_chart(df):
    # Generate Horizontal Bar Chart for the Number of Reviews vs Predicted Sentiment Analysis Score
    ax = df['Sentiment Analysis Score'].value_counts(sort=False).plot(kind='barh', color = 'red')
    ax.set_xlabel('Number of Amazon Reviews')
    ax.set_ylabel('Predicted Sentiment Analysis Score')
    plt.show()

def print_accuracy_and_f1_score(df,rating):
    # Print Accuracy and F1 Score
    acc = accuracy_score(rating, df['Sentiment Analysis Score']) * 100
    f1 = f1_score(rating, df['Sentiment Analysis Score'], average='macro')
    print("Accuracy: {}\nMacro F1-score: {}".format(acc, f1))

def plot_cm(df,rating):
    # Plot Confusion Matrix Heatmap
    cm = confusion_matrix(rating, df['Sentiment Analysis Score'])
    fig, ax = plt.subplots(figsize=(8,6))
    sn.heatmap(cm, annot=True, fmt='d',xticklabels=categories, yticklabels=categories)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sentiments_file_path',
        type=str,
        default='svm_predicted_sentiments.csv')
    args = parser.parse_args()
    sentiments_file_path = args.sentiments_file_path

    # Read the sentiments File
    df = pd.read_csv(sentiments_file_path)

    # Print the top 5 results to inspect the data frame
    print(df.head())

    categories = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    rating = pd.Series([int(i[0]) for i in pd.Series(df['Star Rating'])])

    generate_wordcloud(df)

    star_ratings_bar_chart(df, rating)

    predicted_sentiments_bar_chart(df)

    print_accuracy_and_f1_score(df, rating)

    plot_cm(df, rating)
