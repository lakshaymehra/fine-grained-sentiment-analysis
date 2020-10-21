# Load data
import pytreebank
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    out_path = os.path.join(sys.path[0], 'sst_{}.txt')
    dataset = pytreebank.load_sst()

    # I ran the following commented code to get train, dev and test sets

    # Store train, dev and test in separate files
    for category in ['train', 'test', 'dev']:
        with open(out_path.format(category), 'w') as outfile:
            for item in dataset[category]:
                outfile.write("__label__{}\t{}\n".format(
                    item.to_labeled_lines()[0][0] + 1,
                    item.to_labeled_lines()[0][1]
                ))
    # Print the length of the training set
    print(len(dataset['train']))

    # Read train data
    df = pd.read_csv('sst_train.txt', sep='\t', header=None, names=['truth', 'text'], encoding = "ISO-8859-1")
    df['truth'] = df['truth'].str.replace('__label__', '')
    df['truth'] = df['truth'].astype(int).astype('category')
    print(df.head())

    # Plot number of samples in the train data per category
    ax = df['truth'].value_counts(sort=False).plot(kind='barh')
    ax.set_xlabel('Number of Samples in training Set')
    ax.set_ylabel('Label')
    plt.show()

    # Sort sentences in ascending order according to the length to
    # analyze that short sentences have been labeled as 'neutral'
    df['len'] = df['text'].str.len()
    df = df.sort_values(['len'], ascending=True)
    print(df.head(20))