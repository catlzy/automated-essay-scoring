import csv
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
import enchant
import textmining
import ml_metrics as metrics

import matplotlib
import matplotlib.pyplot as plt

from collections import Counter
from sklearn import linear_model
from sklearn.model_selection import KFold
import numpy as np
import pickle
import operator
import math
import string
import re
import pandas as pd


TOPIC_NUM = 0
TEXT = 1
SCORE = 2

def get_info():
    essay_list = [[] for i in range(8)]
    with open('asap-aes/training_set_rel3.tsv', encoding='ISO-8859-1') as tfile:
        csv_reader = csv.reader(tfile, delimiter='\t')
        prev_topic = '0'
        for row in csv_reader:
            if row[1] != 'essay_set':
                if row[1] != '2':
                    essay_list[int(row[1])-1].append([row[1], row[2].lower(), int(row[6])])
                else:
                    essay_list[int(row[1])-1].append([row[1], row[2].lower(), int(row[6])+int(row[9])])
    return essay_list


def sentence_num(topics):
    x_len = []
    for topic in topics:
        sent_num = len(sent_tokenize(topic[TEXT]))
        x_len.append(sent_num)

    return x_len


def sentence_len(topics):
    x_len = []
    for topic in topics:
        sent_num = len(sent_tokenize(topic[TEXT]))
        tokens = word_tokenize(topic[TEXT])
        words = [word for word in tokens if word.isalpha()]
        leng = min(len(words)/sent_num, 100)
        x_len.append(leng)
    return x_len


def word_count(topics):
    x_len = []
    for topic in topics:
        tokens = word_tokenize(topic[TEXT])
        words = [word for word in tokens if word.isalpha()]
        x_len.append(len(words))

    return x_len


def long_word_count(topics):
    x_len = []
    for topic in topics:
        tokens = word_tokenize(topic[TEXT])
        words = [word for word in tokens if word.isalpha() and len(word) > 4.5]
        x_len.append(len(words))

    return x_len


def punctuation_count(topics):
    p = string.punctuation.replace('@', '')
    x_len = []
    for topic in topics:
        tokens = word_tokenize(topic[TEXT])
        words = [word for word in tokens if word in p]
        x_len.append(len(words))

    return x_len

#number of unique words with correct spelling and no stopwords
def unique_valid_word_count(topics):
    x_len = []
    stop_words=set(stopwords.words("english"))
    d = enchant.Dict("en_US")
    for topic in topics:
        tokens = word_tokenize(topic[TEXT])
        words = [word for word in tokens if word.isalpha()]
        cleaned_words = set(word for word in words if word not in stop_words and d.check(word))
        x_len.append(len(cleaned_words))

    return x_len


def unique_valid_word_prop(topics):
    x_len = []
    stop_words=set(stopwords.words("english"))
    d = enchant.Dict("en_US")
    for topic in topics:
        tokens = word_tokenize(topic[TEXT])
        words = [word for word in tokens if word.isalpha()]
        cleaned_words = set(word for word in words if word not in stop_words and d.check(word))
        x_len.append(len(cleaned_words)/len(tokens))

    return x_len


def average_word_length(topics):
    #using all words
    # x_len = []
    # for topic in topics:
    #     tokens = word_tokenize(topic[TEXT])
    #     words = [word for word in tokens if word.isalpha()]
    #     total_len = len(''.join(words))
    #     x_len.append(total_len/len(words))

    #using unique valid words
    x_len = []
    stop_words=set(stopwords.words("english"))
    d = enchant.Dict("en_US")

    for topic in topics:
        tokens = word_tokenize(topic[TEXT])
        words = [word for word in tokens if word.isalpha()]
        cleaned_words = set(word for word in words if word not in stop_words and d.check(word))
        x_len.append(len(''.join(cleaned_words))/len(cleaned_words))

    return x_len


def part_of_speech_count(topics):
    tags = ['NN', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR', 'RBS']
    x_len = []
    for topic in topics:
        tag_dict = {}
        sents = sent_tokenize(topic[TEXT])
        for sent in sents:
            text = word_tokenize(sent.lower())
            tagged = nltk.pos_tag(text)
            for item in tagged:
                tag_dict[item[0]] = item[1]
        count = 0
        for word, tag in tag_dict.items():
            if tag in tags:
                count += 1
        x_len.append(count)

    return x_len


def n_grams(topics, n):
    x_len = []
    terms = []
    documents = []
    setnum = topics[0][TOPIC_NUM]
    for topic in topics:
        new = topic[TEXT].translate(str.maketrans('', '', string.punctuation))
        words = nltk.word_tokenize(new)
        grams = list(ngrams(words, n))
        documents.append(grams)
        terms = terms + grams
    fdist = nltk.FreqDist(terms)
    pickle.dump(fdist, open('fdist/set' + setnum + '_n' + str(n) + '.txt', 'wb'))
    for t in documents:
        x_len.append(sum(1/fdist[b] for b in t))
    return x_len


def feature_data(train):
    x_sennum = []
    x_senlen = []
    x_wlen = []
    x_lwlen = []
    x_pclen = []
    x_uclen = []
    x_awlen = []
    x_pslen = []
    x_uplen = []
    x_one = []
    x_two = []
    x_three = []
    x_four = []
    x_five = []
    x_six = []
    x_seven = []
    x_eight = []
    x_nine = []
    x_ten = []
    y_score = []
    for topics in train:
        y_score.append([topic[SCORE] for topic in topics])
        x_sennum.append(sentence_num(topics))
        x_senlen.append(sentence_len(topics))
        x_wlen.append(word_count(topics))
        x_lwlen.append(long_word_count(topics))
        x_pclen.append(punctuation_count(topics))
        x_uclen.append(unique_valid_word_count(topics))
        x_uplen.append(unique_valid_word_prop(topics))
        x_awlen.append(average_word_length(topics))
        x_pslen.append(part_of_speech_count(topics))
        x_one.append(n_grams(topics, 1))
        x_two.append(n_grams(topics, 2))
        x_three.append(n_grams(topics, 3))
        x_four.append(n_grams(topics, 4))
        x_five.append(n_grams(topics, 5))
    feature_dict = {'sentence_num': x_sennum, 'sentence_len': x_senlen, 'word_count': x_wlen, 'long_word_count': x_lwlen, 'punctuation_count': x_pclen, 'unique_valid_word_count': x_uclen, 'average_word_length': x_awlen,
                    'noun_adj_adv_count': x_pslen, 'unique_valid_word_prop': x_uplen, '1gram_frequency': x_one, '2gram_frequency': x_two, '3gram_frequency': x_three, '4gram_frequency': x_four, '5gram_frequency': x_five, 'score': y_score}
    with open('feature_data.txt', 'wb') as f:
        pickle.dump(feature_dict, f)


def plot_feature_data(feature_data):
    # all n-gram for each essayset
    scores = feature_data['score']
    for i in range(len(scores)):
        fig = plt.figure(figsize=(13,4))
        fig.suptitle('Essay Set ' + str(i+1))
        j = 1
        for feature_title, value in feature_data.items():
            if feature_title != 'score':
                if feature_title.endswith('gram_frequency'):
                    plt.subplot(1,5,j)
                    plt.scatter(value[i], scores[i], s=1.5)
                    plt.xlabel(feature_title)
                    if j == 1 or j == 6:
                        plt.ylabel("Score")
                    j += 1
        fig.savefig('images/Essay Set ' + str(i+1) + ' n_grams.jpg')

    # all features for each essayset
    for i in range(len(scores)):
        fig = plt.figure(figsize=(8,10))
        fig.suptitle('Essay Set ' + str(i+1))
        j = 1
        for feature_title, value in feature_data.items():
            if feature_title != 'score':
                if not feature_title.endswith('gram_frequency'):
                    plt.subplot(3,3,j)
                    plt.scatter(value[i], scores[i], s=1.5)
                    plt.xlabel(feature_title)
                    if j == 1 or j == 4 or j == 7:
                        plt.ylabel("Score")
                    j += 1
        fig.savefig('images/Essay Set ' + str(i+1) + ' Features.jpg')

    # all essayset for each feature
    for feature_title, value in feature_data.items():
        if feature_title != 'score':
            fig = plt.figure(figsize=(10,8))
            fig.suptitle(feature_title)
            j = 1
            for i in range(len(scores)):
                plt.subplot(2,4,j)
                plt.scatter(value[i], scores[i], s=1.5)
                plt.title('Essay Set ' + str(i+1))
                if j == 1 or j == 5:
                    plt.ylabel("Score")
                j += 1
            fig.savefig('images/' + feature_title + '.jpg')
    return


def feature_testing(feature_data):
    scores = feature_data['score']
    lm = linear_model.LinearRegression()
    kappa = {}
    #for each feature(j) in each essayset(i)
    for i in range(len(scores)):
        for key, value in feature_data.items():
            if key != 'score':
                model = lm.fit(np.array(value[i]).reshape(-1,1),scores[i])
                pred_score = lm.predict(np.array(value[i]).reshape(-1,1))
                if key not in kappa:
                    kappa[key] = []
                kappa[key].append(metrics.quadratic_weighted_kappa(scores[i], pred_score))
    for key, value in kappa.items():
        kappa[key] = sum(value)/len(value)
    sorted_kappa = dict(sorted(kappa.items(), key=operator.itemgetter(1),reverse=True))
    print(sorted_kappa)
    return sorted_kappa


def feature_selection(sorted_feature, feature_data):
    features = []
    scores = feature_data['score']
    lm = linear_model.LinearRegression()
    prev = 0
    for title in sorted_feature.keys():
        features.append(title)
        kappa = 0
        z = 0
        for i in range(len(scores)):
            X = []
            for t in features:
                X.append(feature_data[t][i])
            if len(X) == 1:
                X = np.array(X[0]).reshape(-1,1)
            else:
                X = [list(x) for x in zip(*X)]
            model = lm.fit(X,scores[i])
            pred_score = lm.predict(X)
            kappa = metrics.quadratic_weighted_kappa(scores[i], pred_score)
            z += 1/2*math.log((1+kappa)/(1-kappa))
        weighted_kappa_mean = (math.e**(2*z/(len(scores)))-1)/(math.e**(2*z/(len(scores)))+1)
        if weighted_kappa_mean < prev:
            features = features[:-1]
        else:
            prev = weighted_kappa_mean
        print(features)
        print(weighted_kappa_mean)
    pickle.dump(features, open('selected_features.txt', 'wb'))
    return features


def produce_model(selected, sorted_feature, feature_data):
    features = []
    temp = ['7gram_frequency', '8gram_frequency', '9gram_frequency', '10gram_frequency']
    scores = feature_data['score']
    lm = linear_model.LinearRegression()
    for title in sorted_feature.keys():
        if title in selected:
            features.append(title)
    print(selected)
    for i in range(len(scores)):
        X = []
        for t in features:
            X.append(feature_data[t][i])
        X = np.array([list(x) for x in zip(*X)])
        model = lm.fit(X,np.array(scores[i]))
        pickle.dump(model, open('linreg_models/model' + str(i+1) + '.sav', 'wb'))
    return


def linear_reg(selected, sorted_feature, feature_data):
    features = []
    scores = feature_data['score']
    lm = linear_model.LinearRegression()
    kf = KFold(n_splits=10)
    overall = []
    for title in sorted_feature.keys():
        if title in selected:
            features.append(title)
    for i in range(len(scores)):
        z = 0
        X = []
        for t in features:
            X.append(feature_data[t][i])
        X = np.array([list(x) for x in zip(*X)])
        count = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np.array(scores[i])[train_index], np.array(scores[i])[test_index]
            model = lm.fit(X_train,y_train)
            pred_score = lm.predict(X_test)
            kappa = metrics.quadratic_weighted_kappa(y_test, pred_score)
            z += 1/2*math.log((1+kappa)/(1-kappa))
            count += 1
        weighted_kappa_mean = (math.e**(2*z/count)-1)/(math.e**(2*z/count)+1)
        overall.append(weighted_kappa_mean)
        print(weighted_kappa_mean)
    print(sum(overall)/len(overall))
    return


if __name__=='__main__':
    train = get_info()

    # produce feature_data dictionary, only need for the first run
    # feature_data(train)

    # open saved feature_data dict
    with open('feature_data.txt', 'rb') as f:
        feature_data = pickle.load(f)

    # create plots
    # plot_feature_data(feature_data)

    # get sorted feature titles in most significant order
    sorted_feature = feature_testing(feature_data)

    # get final feature titles for model
    selected = feature_selection(sorted_feature, feature_data)

    # produce all 8 linear regression models
    produce_model(selected, sorted_feature, feature_data)

    # get accuracy for 10fold cross validation on training set
    linear_reg(selected, sorted_feature, feature_data)
