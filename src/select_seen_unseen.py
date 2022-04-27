import os
import argparse
import string
import random
from random import sample
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval


def read_data(in_file):
    """
    Load data in a dataframe
    """
    data = []
    with open(in_file, 'r') as _:
        for line in tqdm(_.readlines()):
            row = line.replace('\n', '').split('\t')
            toks_probs = []
            for i in range(2, len(row)):
                result = literal_eval(row[i])
                if len(result) != 2:
                    toks = result.split(', ')
                    result = (toks[0][1:], float(toks[1][:-1]))
                toks_probs.append(result)

            avg_word_prob, avg_trans_prob = assess_memorization(toks_probs)
            data.append([row[0], float(row[1]), avg_word_prob, avg_trans_prob])
    df = pd.DataFrame(data, columns=['name', 'ppl', 'expword', 'exptrans'])
    return df


def assess_memorization(entity):
    """
    Assess memorization - see paper for more detail
    """
    # average word probabilities
    word_probs = []
    # average transition probabilities
    trans_probs = []
    for i in range(len(entity)):
        word, prob = entity[i]
        prev_word, prev_prob = entity[i - 1]
        if word[0] == 'Ġ' or word in string.punctuation:
            if word not in string.punctuation:
                trans_probs .append(prob)
            if i > 0:
                if prev_word[0] == 'Ġ':
                    # print(prev_word, 1)
                    word_probs .append(1)
                elif prev_word not in string.punctuation:
                    if i > 1:
                        # print(prev_word, prev_prob)
                        word_probs .append(prev_prob)
                    else:
                        # print(prev_word, 1)
                        word_probs .append(1)
        if i == len(entity) - 1 and word not in string.punctuation:
            if word[0] == 'Ġ' or i == 0:
                # print(word, 1)
                word_probs .append(1)
            else:
                # print(word, prob)
                word_probs .append(prob)
    # print(avg_word_probs, avg_trans_probs, '\n')
    if trans_probs:
        return np.prod(word_probs), np.min(trans_probs)
    return np.prod(word_probs), np.nan


def select_seen(df, seen_word_exp=0, seen_tran_exp=0, ignore_words=False):
    """
    Select seen entities
    """
    seen_ents = set()
    if not ignore_words:
        # entities with no transitions (single-word)
        seen_ents.update(df[(df.expword >= seen_word_exp)  & df.exptrans.isnull()].name.tolist())
    # entities with transitions (multi-word)
    seen_ents.update(df[(df.expword >= seen_word_exp) & (df.exptrans >= seen_tran_exp)].name.tolist())
    return seen_ents


def select_rare_unseen(df, rare_word_exp=0, rare_tran_exp=0):
    """
    Select rare or unseen entities
    """
    rare_entities = set()
    if rare_word_exp != 0 and rare_tran_exp != 0:
        rare_entities.update(df[(df.exptrans < rare_tran_exp) & (df.expword < rare_word_exp)].name.tolist())
    elif rare_word_exp != 0:
        rare_entities.update(df[df.expword < rare_word_exp].name.tolist())
    else:
        rare_entities.update(df[df.exptrans < rare_tran_exp].name.tolist())
    return rare_entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--input_file', dest='in_file', action='store', required=True, help='the path of the input file')
    parser.add_argument('--seen_word_exp', dest='seen_word_exp', type=float, action='store', required=False, default=0, help='seen word exposure threshold')
    parser.add_argument('--seen_tran_exp', dest='seen_tran_exp', type=float, action='store', required=False, default=0, help='seen transition exposure threshold')
    parser.add_argument('--rare_word_exp', dest='rare_word_exp', type=float, action='store', required=False, default=0, help='rare or unseen word exposure threshold')
    parser.add_argument('--rare_tran_exp', dest='rare_tran_exp', type=float, action='store', required=False, default=0, help='rare or unseen transition exposure threshold')
    args = parser.parse_args()
    args = vars(args)
    in_file = args['in_file']
    seen_word_exp = args['seen_word_exp']
    seen_tran_exp = args['seen_tran_exp']
    rare_word_exp = args['rare_word_exp']
    rare_tran_exp = args['rare_tran_exp']

    df = read_data(in_file)
    if 'dbp.' in in_file or 'mit' in in_file:
        seen_entities = select_seen(df, seen_word_exp, seen_tran_exp, ignore_words=True)
    else:
        seen_entities = select_seen(df, seen_word_exp, seen_tran_exp)
    rare_entities = select_rare_unseen(df, rare_word_exp, rare_tran_exp)

    random.seed(1)
    if len(rare_entities) > len(seen_entities):
        rare_entities = sample(rare_entities, len(seen_entities))
    print(len(seen_entities), len(rare_entities), df.shape[0])
    if len(rare_entities) > 10000:
        rare_entities = sample(rare_entities, 10000)
    if len(seen_entities) > 10000:
        seen_entities = sample(seen_entities, 10000)

    with open(os.path.splitext(in_file)[0] + '_seen_{0}_{1}.tsv'.format(seen_word_exp, seen_tran_exp), 'w') as _:
        for ent in seen_entities:
            _.write(ent + '\n')
    with open(os.path.splitext(in_file)[0] + '_rare_unseen_{0}_{1}.tsv'.format(rare_word_exp, rare_tran_exp), 'w') as _:
        for ent in rare_entities:
            _.write(ent + '\n')
