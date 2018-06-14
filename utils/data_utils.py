# Adapted from Chatbot-from-Movie-Dialogue, itself an adaptation of 
# https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets
# /cornell_corpus/data.py.

import re
from collections import Counter
from random import sample

import numpy as np
import pandas as pd
from tensorflow.python.ops import lookup_ops

from .vocab_utils import build_vocab_file

DEBUG = True
UNK_ID = 0


def check_data(data_file):
    pass


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the
    format of words.'''

    text = text.lower()

    text = re.sub(r"<.>", "", text)
    text = re.sub(r"<..>", "", text)

    text = re.sub(r"n' ", "ng ", text)
    text = re.sub(r"'bout ", " about ", text)
    text = re.sub(r" til ", "until", text)
    text = re.sub(r"['-()\"#/@;:<>{}`+=~|.!?,]", " ", text)

    text = re.sub(r"  ", " ", text)
    text = re.sub(r"  ", " ", text)
    text = re.sub(r"  ", " ", text)

    text = re.sub(r"i m ", "i am ", text)
    text = re.sub(r"he s ", "he is ", text)
    text = re.sub(r"she s ", "she is ", text)
    text = re.sub(r"it s ", "it is ", text)
    text = re.sub(r"that s ", "that is ", text)
    text = re.sub(r"what s ", "that is ", text)
    text = re.sub(r"where s ", "where is ", text)
    text = re.sub(r"how s ", "how is ", text)
    text = re.sub(r" ll ", " will ", text)
    text = re.sub(r" ve ", " have ", text)
    text = re.sub(r" re ", " are ", text)
    text = re.sub(r" d ", " would ", text)
    text = re.sub(r" re ", " are ", text)
    text = re.sub(r"won t ", "will not ", text)
    text = re.sub(r"can t ", "cannot ", text)
    text = re.sub(r"n t ", " not ", text)

    return text


def load_data():
    pass


def split_data(movies, ratio_test=0.1, ratio_dev=0.1):
    dataset_size = len(movies)
    list_movie = list(set(movies))
    list_movie = sample(list_movie, len(list_movie))
    movie_set_size = len(list_movie)
    split_test = int(movie_set_size * ratio_test)
    split_dev = int(movie_set_size * (ratio_test + ratio_dev))

    test_ind= [i for i, m in enumerate(movies) if
                 m in list_movie[:split_test]]
    dev_ind = [i for i, m in enumerate(movies)
               if m in list_movie[split_test:split_dev]]
    train_ind = [i for i, m in enumerate(movies) if m in list_movie[split_dev:]]
    if DEBUG:
        test_size = len(test_ind)
        dev_size = len(dev_ind)

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
        print("{:.1f}% of the movies used for training "
              "({:.1f}% of the entire dataset)".format(ratio_train * 100,
                                                       real_ratio_training *
                                                       100))

    return train_ind, dev_ind, test_ind


def build_data(input_dir, output_dir):
    lines, conv_lines = load_raw_data(input_dir)
    movies, dataset = build_data_utterances(lines, conv_lines)
    train_ind, dev_ind, test_ind = split_data(movies)

    for fold, ind in [("train", train_ind), ("dev", dev_ind),
                      ("test", test_ind)]:
        write_data(dataset[ind, 0], "questions", fold, output_dir)
        write_data(dataset[ind, 1], "answers", fold, output_dir)
        write_data(dataset[ind, 2], "q_speaker", fold, output_dir)
        write_data(dataset[ind, 3], "a_speaker", fold, output_dir)

    build_vocab_file(dataset[:, 0], dataset[:, 1], 'vocab', output_dir,
                     threshold=10)
    build_vocab_file(dataset[:, 2], dataset[:, 3], 'speakers', output_dir,
                     threshold=4)

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
                          max_line_length=20):
    """Build a list of question and a list of answers from lines and 
    conversations"""
    # Create a dictionary to map each line's id with its text
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = (_line[1].strip(),
                                 clean_text(_line[4]))
    print(list(id2line)[:10])
    questions = []
    answers = []
    questions_speaker = []
    answers_speaker = []
    movie_qa = []

    for line in conv_lines[:-1]:
        split_line = line.split(' +++$+++ ')
        _utterances = split_line[-1][2:-2].split("', '")

        question = _utterances[0]
        for utt in _utterances[1:]:
            questions_speaker.append(id2line[question][0])
            answers_speaker.append(id2line[utt][0])
            questions.append(id2line[question][1])
            answers.append(id2line[utt][1])
            movie_qa.append(split_line[2])

            question = utt

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

        # Find the length of sentences
        lengths = []
        for question in questions:
            lengths.append(len(question.split()))
        for answer in answers:
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
    short_questions_speaker = []
    short_answers_speaker = []

    for i in range(len(questions)):
        question = questions[i]
        answer = answers[i]

        if min_line_length <= len(question.split()) <= max_line_length:
            if min_line_length <= len(answer.split()) <= max_line_length:

                short_answers.append(answers[i])
                short_questions.append(questions[i])
                short_movies.append(movie_qa[i])
                short_questions_speaker.append(questions_speaker[i])
                short_answers_speaker.append(answers_speaker[i])

    if DEBUG:
        # Compare the number of lines we will use with the total number of
        # lines.
        print("# of questions:", len(short_questions))
        print("# of answers:", len(short_answers))
        print("% of data used: {:.1f}%".format(
                len(short_questions) / len(questions) * 100))
        print()
    dataset = np.array(list(zip(short_questions, short_answers,
                                short_questions_speaker,
                                short_answers_speaker)))

    return np.array(short_movies), dataset


def create_speaker_tables(speaker_table_file):
    """Creates speaker tables for question file"""
    ## TODO account for speaker only present in answers

    speaker_table = lookup_ops.index_table_from_file(
            speaker_table_file, default_value=UNK_ID)
    return speaker_table


def write_data(data, input_type, fold, output_dir):
    # write the data
    with open(output_dir + "/" + fold + '.' + input_type, "w") as f:
        f.write("\n".join(data))
