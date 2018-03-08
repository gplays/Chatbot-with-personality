# Adapted from Chatbot-from-Movie-Dialogue, itself an adaptation of 
# https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets
# /cornell_corpus/data.py.

import re
from collections import Counter
from random import sample

import numpy as np
import pandas as pd
from tensorflow.python.ops import lookup_ops

from utils.vocab_utils import build_vocab_file

DEBUG = True
UNK_ID = 0


def check_data(data_file):
    pass


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the
    format of words.'''

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text


def load_data():
    pass


def split_data(questions, answers, movies, ratio_test=0.2, ratio_dev=0.2):
    dataset_size = len(questions)
    m = Counter(movies)
    ids = m.keys()
    movie_set_size = len(ids)
    split_test = int(movie_set_size * ratio_test)
    split_dev = int(movie_set_size * (ratio_test + ratio_dev))

    s = sample(ids, movie_set_size)

    movie_test = {m: 1 for m in s[:split_test]}
    movie_dev = {m: 1 for m in s[split_test:split_dev]}
    movie_train = {m: 1 for m in s[split_dev:]}

    if DEBUG:
        test_size = sum([c for _, c in m.items() if _ in movie_test])
        dev_size = sum([c for _, c in m.items() if _ in movie_dev])

        real_ratio_test = test_size / dataset_size
        real_ratio_dev = dev_size / dataset_size
        real_ratio_training = 1 - real_ratio_test - real_ratio_dev

        ratio_train = 1 - ratio_dev - ratio_test

        print("Processing Dataset with {} utterances across {} "
              "movies".format(dataset_size, movie_set_size))
        print("{:.1f}% of the movies used for testing "
              "({:.1f}% of the entire dataset)".format(ratio_test * 100,
                                                       real_ratio_test * 100))
        print("{:.1f}% of the movies used for validation "
              "({:.1f}% of the entire dataset)".format(ratio_dev * 100,
                                                       real_ratio_dev * 100))
        print("{:.1f}% of the movies used for validation "
              "({:.1f}% of the entire dataset)".format(ratio_train * 100,
                                                       real_ratio_training *
                                                       100))

    train_q = []
    train_a = []
    dev_q = []
    dev_a = []
    test_q = []
    test_a = []

    for movie, question, answer in zip(movies, questions, answers):

        if movie in movie_train:
            train_a.append(answer)
            train_q.append(question)
        elif movie in movie_dev:
            dev_a.append(answer)
            dev_q.append(question)
        else:
            test_a.append(answer)
            test_q.append(question)

    return train_q, train_a, dev_q, dev_a, test_q, test_a


def build_data(input_dir, output_dir):
    lines, conv_lines = load_raw_data(input_dir)
    questions, answers, movies = build_data_utterances(lines, conv_lines)
    train_q, train_a, dev_q, dev_a, test_q, test_a = split_data(questions,
                                                                answers, movies)

    write_data(train_q, train_a, "train", output_dir)
    write_data(dev_q, dev_a, "dev", output_dir)
    write_data(test_q, test_a, "test", output_dir)

    build_vocab_file(questions, answers, 'vocab', output_dir, threshold=10)

    return


def load_raw_data(input_dir, lines_file=None, conv_file=None):
    if lines_file is None:
        lines_file = input_dir + "/" + 'movie_lines.txt'
    if conv_file is None:
        conv_file = input_dir + "/" + 'movie_conversations.txt'

    # Load the data
    lines = open(lines_file, encoding='utf-8',
                 errors='ignore').read().split('\n')
    conv_lines = open(conv_file, encoding='utf-8',
                      errors='ignore').read().split('\n')

    if DEBUG:
        # The sentences that we will be using to train our model.
        lines[:10]

        # The sentences' ids, which will be processed to become our input and
        # target data.
        conv_lines[:10]

    return lines, conv_lines


def build_data_utterances(lines, conv_lines, min_line_length=2,
                          max_line_length=20, keep_locutor=False):
    """Build a list of question and a list of answers from lines and 
    conversations"""
    # Create a dictionary to map each line's id with its text
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0].strip()] = clean_text(_line[4])

    # Create a list of all of the conversations' lines' ids.
    convs = []
    speakers = []
    movies = []

    for line in conv_lines[:-1]:
        split_line = line.split(' +++$+++ ')

        _utterances = split_line[-1][1:-1]
        _utterances = _utterances.replace("'", "")
        _utterances = _utterances.replace(" ", "")
        convs.append(_utterances.split(','))

        movies.append(split_line[2])

        if keep_locutor:
            speakers.appends([split_line[0], split_line[1]])

    if DEBUG:
        print("id2 line debug", [i for i in id2line][:5])
        print("conv debug", convs[:5])
        print("speakers debug", speakers[:5])
        print("movies debug", movies[:5])
        print()

    # Sort the sentences into questions (inputs) and answers (targets)
    questions = []
    answers = []
    movie_qa = []

    for j in range(len(convs)):

        conv = convs[j]
        movie = movies[j]

        if keep_locutor:
            speaker = speakers[j]

        for i in range(len(conv) - 1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i + 1]])

            if keep_locutor:
                questions.append([speaker[i % 2], id2line[conv[i]]])
                answers.append([speaker[(i + 1) % 2], id2line[conv[i + 1]]])

            movie_qa.append(movie)

    if DEBUG:
        # Check if we have loaded the data correctly
        limit = 0
        print("printing a few utterances")
        print()
        for i in range(limit, limit + 2):
            print(questions[i])
            print(answers[i])
            print()

        # Compare lengths of questions and answers
        print(len(questions), " questions")
        print(len(answers), " answers")
        print()

    # Clean the data
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))

    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    if DEBUG:
        # # Take a look at some of the data to ensure that it has been cleaned
        # # well.
        # limit = 0
        # for i in range(limit, limit + 5):
        #     print(clean_questions[i])
        #     print(clean_answers[i])
        #     print()

        # Find the length of sentences
        lengths = []
        for question in clean_questions:
            lengths.append(len(question.split()))
        for answer in clean_answers:
            lengths.append(len(answer.split()))

        # Create a dataframe so that the values can be inspected
        lengths = pd.DataFrame(lengths, columns=['counts'])

        lengths.describe()

        print("Analysing utterances length")
        print("80 percentile", np.percentile(lengths, 80))
        print("85 percentile", np.percentile(lengths, 85))
        print("90 percentile", np.percentile(lengths, 90))
        print("95 percentile", np.percentile(lengths, 95))
        print("99 percentile", np.percentile(lengths, 99))
        print()

    # Filter out the questions that are too short/long
    short_questions = []
    short_answers = []
    short_movies = []

    for i in range(len(clean_questions)):
        question = clean_questions[i]
        answer = clean_answers[i]
        movie = movie_qa[i]

        if keep_locutor:
            question = clean_questions[i][1]
            answer = clean_answers[i][1]

        if min_line_length <= len(question.split()) <= max_line_length:
            if min_line_length <= len(answer.split()) <= max_line_length:

                short_answers.append(clean_answers[i])
                short_questions.append(clean_questions[i])
                short_movies.append(movie)

    if DEBUG:
        # Compare the number of lines we will use with the total number of
        # lines.
        print("# of questions:", len(short_questions))
        print("# of answers:", len(short_answers))
        print("% of data used: {:.1f}%".format(
            len(short_questions) / len(questions) * 100))
        print()

    return short_answers, short_questions, short_movies


def create_speaker_tables(speaker_table_file):
    """Creates speaker tables for question file"""
    ## TODO account for speaker only present in answers

    speaker_table = lookup_ops.index_table_from_file(
        speaker_table_file, default_value=UNK_ID)
    return speaker_table


def write_data(questions, answers, fold, output_dir):
    # write the data


    with open(output_dir + "/" + fold + '.answers', "w") as f:
        f.write("\n".join(answers))
    with open(output_dir + "/" + fold + '.questions', "w") as f:
        f.write("\n".join(questions))
