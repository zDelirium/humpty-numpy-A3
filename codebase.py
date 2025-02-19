import numpy as np
import csv
import math


# Return feature data in total, yes and no labels as dictionaries
def load_training_data(filename='covid_training.tsv'):
    feature_complete = {}
    feature_yes = {}
    feature_no = {}
    labels = []

    with open('A3_Dataset/' + filename, encoding="utf8") as train_data_file:
        file_rows = csv.reader(train_data_file, delimiter='\t')
        next(file_rows)

        for row in file_rows:

            words = row[1].lower().split()
            labels.append(row[2])

            for word in words:

                # If the word is not in the vocabulary, add it. Else add to the count
                if (word not in feature_complete.keys()):
                    feature_complete[word] = 1
                else:
                    feature_complete[word] += 1

                # Do the same thing, but with the per-label features
                if (row[2] == 'yes'):
                    if (word not in feature_yes.keys()):
                        feature_yes[word] = 1
                    else:
                        feature_yes[word] += 1
                else:
                    if (word not in feature_no.keys()):
                        feature_no[word] = 1
                    else:
                        feature_no[word] += 1

    return feature_complete, feature_yes, feature_no, labels


# Removes from the passed dictionaries the key-value pairs for which the key has a value of 1
def filter_words(feature_complete, feature_yes, feature_no):
    words_to_remove = []

    # Get the words that should be removed. Remove them from the yes and no list, but not the overall list
    for word in feature_complete:
        if (feature_complete[word] == 1):
            words_to_remove.append(word)
            if (word in feature_yes.keys()):
                del feature_yes[word]
            else:
                del feature_no[word]

    # Remove the words to remove from overall list
    for word in words_to_remove:
        del feature_complete[word]

    return feature_complete, feature_yes, feature_no


# Returns the size of the vocabulary
def get_vocabulary_size(feature_complete):
    return len(feature_complete)


# Computes prior probability of a Tweet being factual (yes/no)
def prior(label_str, labels):
    return len([word for word, label in enumerate(labels) if label == label_str]) / len(labels)


# Computes (smoothed) conditional probability from a passed word and passed yes/no feature list
def conditional(word, feature_yn, vocab_size, smoothing):
    if (word in feature_yn.keys()):
        return (feature_yn[word] + smoothing) / (sum(feature_yn.values()) + smoothing * vocab_size)
    else:
        return smoothing / (sum(feature_yn.values()) + smoothing * vocab_size)


# Load test data with Tweet ID, the tokenized lowercased string Tweet, and the actual labels of each Tweets (yes/no)
def load_test_data(filename='covid_test_public.tsv'):
    tweet_id = []
    labels = []
    tweets = []

    with open('A3_Dataset/' + filename, encoding="utf8") as test_data_file:
        file_rows = csv.reader(test_data_file, delimiter='\t')

        for row in file_rows:
            tweet_id.append(row[0])
            labels.append(row[2])
            tweets.append(row[1].lower().split())

    return tweet_id, tweets, labels


# Predicts the labels of a passed set of tokenized words. Return the predicted labels. Return the predicted labels of each Tweets in the gieven set and their best score
def predict(tweets, training_labels, feature_yes, feature_no, vocab_size, smoothing=0.01):
    predicted_labels = []
    best_scores = []

    priors = [prior('yes', training_labels), prior('no', training_labels)]

    feature_yn = [feature_yes, feature_no]

    for i in range(len(tweets)):
        scores = [0] * 2  # index 0 for yes, 1 for no
        for j in range(len(scores)):
            # Add prior probability
            scores[j] += math.log10(priors[j])
            for word in tweets[i]:
                # Add conditionals
                scores[j] += math.log10(conditional(word, feature_yn[j], vocab_size, smoothing))

        # Keep track of minimum scores and predicted labels
        max_score = max(scores)
        best_scores.append(max_score)
        if (max_score == scores[0]):
            predicted_labels.append('yes')
        else:
            predicted_labels.append('no')

    return predicted_labels, best_scores


# Creates an output file for the predictions of the NB classifier
def create_prediction_output_file(filename, tweet_id, predicted_labels, true_labels, best_scores):
    output = open('A3_Output/' + filename, 'w')

    for i in range(len(tweet_id)):

        # Write tweet id
        output.write(str(tweet_id[i]) + '  ')

        # Write prediction label
        output.write(predicted_labels[i] + '  ')

        # Write score of the predicted label
        output.write(format(best_scores[i], ".2E") + '  ')

        # Write true label
        output.write(true_labels[i] + '  ')

        # Write if the classifier predicted correctly or not
        if (predicted_labels[i] == true_labels[i]):
            output.write('correct\n')
        else:
            output.write('wrong\n')

    output.close()


# TODO metrics (accuracy, per-label recall, per-label precision, per-label f1-measure)

def create_metrics_file(filename, predicted_labels, true_labels):
    output = open('A3_Output/' + filename, 'w')

    # Initializing metric info. These are floats to account for decimals
    y_truePositive = 0.0
    y_falsePositive = 0.0
    y_falseNegative = 0.0

    n_truePositive = 0.0
    n_falsePositive = 0.0
    n_falseNegative = 0.0

    totalCount = len(predicted_labels)

    # Calculating true positives, false positives, and false negatives.
    for i in range(totalCount):
        if predicted_labels[i] == true_labels[i]:
            if predicted_labels[i] == "yes":
                y_truePositive += 1
            else:
                n_truePositive += 1
        elif true_labels[i] == "yes":
            # a false negative for Yes means a false positive for No, and vice versa
            y_falseNegative += 1
            n_falsePositive += 1
        else:
            # a false negative for No means a false positive for Yes, and vice versa
            n_falseNegative += 1
            y_falsePositive += 1

    # Calculating accuracy, precision, recall and F1 Measure.
    accuracy = (y_truePositive + n_truePositive) / totalCount

    y_precision = y_truePositive / (y_truePositive + y_falsePositive)
    y_recall = y_truePositive / (y_truePositive + y_falseNegative)
    y_f1Measure = (2 * y_precision * y_recall) / (y_precision + y_recall)

    n_precision = n_truePositive / (n_truePositive + n_falsePositive)
    n_recall = n_truePositive / (n_truePositive + n_falseNegative)
    n_f1Measure = (2 * n_precision * n_recall) / (n_precision + n_recall)

    # Printing output to file. Note we use 2 spaces between each word.

    spaceChar = ' '

    output.write(str(round(accuracy, 4)) + "\n")
    output.write(str(round(y_precision, 4)) + spaceChar * 2 + str(round(n_precision, 4)) + "\n")
    output.write(str(round(y_recall, 4)) + spaceChar * 2 + str(round(n_recall, 4)) + "\n")
    output.write(str(round(y_f1Measure, 4)) + spaceChar * 2 + str(round(n_f1Measure, 4)) + "\n")

# TODO output file for metrics
