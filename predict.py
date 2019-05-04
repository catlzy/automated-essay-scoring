import utils
import pickle
import string
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams


def get_ngram(setnum, n, essay):
    with open('fdist/set' + str(setnum) + '_n' + str(n) + '.txt', 'rb') as f:
        fdist = pickle.load(f)
    new = essay.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(new)
    grams = list(ngrams(words, n))
    total = 0
    for g in grams:
        try:
            total += 1/fdist[g]
        except:
            pass
    return [total]


def vectorize(essay, setnum):
    topics = [[setnum, essay]]
    x_sennum = utils.sentence_num(topics)
    x_senlen = utils.sentence_len(topics)
    x_wlen = utils.word_count(topics)
    x_lwlen = utils.long_word_count(topics)
    x_pclen = utils.punctuation_count(topics)
    x_uclen = utils.unique_valid_word_count(topics)
    x_awlen = utils.average_word_length(topics)
    x_pslen = utils.part_of_speech_count(topics)
    x_uplen = utils.unique_valid_word_prop(topics)
    x_one = get_ngram(setnum, 1, essay)
    x_two = get_ngram(setnum, 2, essay)
    x_three = get_ngram(setnum, 3, essay)
    x_four = get_ngram(setnum, 4, essay)
    x_five = get_ngram(setnum, 5, essay)
    vector_dict = {'sentence_num': x_sennum, 'sentence_len': x_senlen, 'word_count': x_wlen, 'long_word_count': x_lwlen, 'punctuation_count': x_pclen, 'unique_valid_word_count': x_uclen, 'average_word_length': x_awlen,
                    'noun_adj_adv_count': x_pslen, 'unique_valid_word_prop': x_uplen, '1gram_frequency': x_one, '2gram_frequency': x_two, '3gram_frequency': x_three, '4gram_frequency': x_four, '5gram_frequency': x_five}
    return vector_dict


def predict(essay, setnum):
    vector_dict = vectorize(essay, setnum)
    with open('selected_features.txt', 'rb') as f:
        selected = pickle.load(f)
    X = []
    for s in selected:
        X = X + vector_dict[s]
    print(X)
    with open('linreg_models/model' + str(setnum) + '.sav', 'rb') as f:
        model = pickle.load(f)
    pred_score = model.predict(np.array(X).reshape(1,-1))
    return pred_score
