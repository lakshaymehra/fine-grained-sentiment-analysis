import fasttext
import os

if __name__ == "__main__":

    hyper_params = {
        "lr": 0.35,         # Learning rate
        "epoch": 100,       # Number of training epochs to train for
        "wordNgrams": 3,    # Number of word n-grams to consider during training
        "dim": 155,         # Size of word vectors
        "ws": 5,            # Size of the context window for CBOW or skip-gram
        "minn": 2,          # Min length of char ngram
        "maxn": 5,          # Max length of char ngram
        "bucket": 2014846,  # Number of buckets
    }

    training_data_path = 'sst_train.txt'

    # Train the FastText model
    model = fasttext.train_supervised(input=training_data_path, **hyper_params)
    print("FastText model trained with the hyperparameters: \n {}".format(hyper_params))

    model.save_model(os.path.join('C:/Users/mehra/OneDrive/Documents/GitHub/73StringsAssignment', "sst.bin"))

    # Quantize model to reduce space usage
    model.quantize(input=training_data_path, qnorm=True, retrain=True, cutoff=110539)
    model.save_model(os.path.join('C:/Users/mehra/OneDrive/Documents/GitHub/73StringsAssignment', "sst_quantized.ftz"))