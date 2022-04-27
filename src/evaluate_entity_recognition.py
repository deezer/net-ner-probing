import os
import re
import pickle
import numpy as np
from Levenshtein import distance as levenshtein_distance

OUTPUT_DIR = os.path.join('output/entity_recognition/')
if not os.path.isdir(OUTPUT_DIR):
    print(f"{OUTPUT_DIR} with the NER results does not exist")
    exit(1)
BASELINE_DIR = os.path.join('../data/entity_recognition/WNUT2017/')

ent_types = {}
ent_types['wnut'] = ['person', 'location', 'corporation', 'group', 'product', 'creative-work']
ent_types['conll'] = ['PER', 'LOC', 'ORG']
ent_types['mitmovie'] = ['person', 'title']


def count_words_in_text(ent, text, is_wnut=False):
    """
    Checks if the generated entity words are in text
    Used as helper for negative entity matching
    """
    ent_words = ent.lower().split()
    if is_wnut:
        if not ent.isalpha():
            return 1
        else:
            count = np.sum([1 for x in ent_words if x in text.lower()])
    else:
        count = np.sum([1 for x in ent_words if x in text.lower().split()])
    return count


def eval(data, is_wnut):
    """
    Compute accuracy for positive and negative ner examples in test set
    """
    all_negatives = 0
    all_positives = 0
    true_positives = 0
    true_negatives = 0
    for t, p, text in zip(data['test_labels'], data['all_reweighted_ans'], data['test_sentences']):
        if text.strip() == "":
            continue
        p = p.split('\n')[0]
        if t == 'none':
            count_common_words = count_words_in_text(p, text, is_wnut)
            if p.lower() == 'none' or count_common_words == 0:
                true_negatives += 1
            all_negatives += 1
        else:
            ents = t.split('\t')
            found_ent = False
            for e in ents:
                d = levenshtein_distance(e.lower(), p.lower())
                if d == 0 or d / len(e) < 0.2:
                    found_ent = True
                    break
            if found_ent:
                true_positives += 1
            all_positives += 1

    false_positives = all_negatives - true_negatives
    false_negatives = all_positives - true_positives
    print("tp {:.2f}, fp {:.2f}, tn {:.2f}, fn {:.2f}".format(true_positives, false_positives, true_negatives, false_negatives))
    f1_score = true_positives / (true_positives + 1 / 2 * (false_positives + false_negatives))
    return true_positives / all_positives, true_negatives / all_negatives, all_positives + all_negatives, f1_score


def print_results(file_name, ents, extra=''):
    """
    Print NER results
    """
    is_wnut = 'wnut' in file_name
    for ent in ents:
        print('***', ent, '***')
        results = []
        for i in range(3):
            fname = ''.join([file_name, ent, '_seed', str(i), extra, '.pkl'])
            with open(fname, 'rb') as file:
                data = pickle.load(file)
                results.append(eval(data, is_wnut))

        sep = ' '  # '$\\pm$'
        acc_p = [x[0] for x in results]
        acc_n = [x[1] for x in results]
        f1score = [x[3] for x in results]
        print('support {}'.format(results[0][2]))
        print("acc@p: {:.2f}{}{:.2f}".format(np.mean(acc_p), sep, np.std(acc_p)))
        print("acc@n: {:.2f}{}{:.2f}".format(np.mean(acc_n), sep, np.std(acc_n)))
        print("f1score: {:.2f}{}{:.2f}".format(np.mean(f1score), sep, np.std(f1score)))
        print('\n')

for exp_name in ent_types:
    print(exp_name)
    print_results(os.path.join(OUTPUT_DIR, exp_name + '_'), ent_types[exp_name])

print('SEEN')
for exp_name in ['wnut']:
    print(exp_name)
    print_results(os.path.join(OUTPUT_DIR, exp_name + '_'), ent_types[exp_name], extra='_seen')

print('RARE/UNSEEN')
for exp_name in ['wnut']:
    print(exp_name)
    print_results(os.path.join(OUTPUT_DIR, exp_name + '_'), ent_types[exp_name], extra='_unseen')

# Baseline evaluation

def read_input_file(input_file):
    """
        Read file with BIO annotations
        (each line contains a token and a label)
    """
    sentences = {}
    with open(input_file, 'r') as _:
        lines = [line.replace('\n', '') for line in _.readlines()]
        current_sent = []
        for line in lines:
            if not line:
                sent = ' '.join([x[0] for x in current_sent])
                sentences[sent] = current_sent
                current_sent = []
            else:
                word, tag = re.split(r'\s+', line)
                current_sent.append((word, tag))
    sent = ' '.join([x[0] for x in current_sent])  # the last one
    sentences[sent] = current_sent
    return sentences


def entities_extracted(sentence, ent_type):
    """
        Return list of entities from a sentence of a given type
    """
    ents = set()
    current_ent = ''
    for word, tag in sentence:
        if tag == 'B-' + ent_type:
            current_ent = word
        elif tag == 'I-' + ent_type:
            current_ent += ' ' + word
        else:
            if current_ent != '':
                ents.add(current_ent)
            current_ent = ''
    if current_ent != '':
        ents.add(current_ent)
    return ents


def eval_wnut_baseline(predicted, true, ent_type):
    """
        Evaluate UH-RiTUAL
    """
    true_negatives = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for p, t in zip(predicted, true):
        #print(p)
        #print(t)
        predicted_ents = entities_extracted(p, ent_type)
        true_ents = entities_extracted(t, ent_type)

        if len(true_ents) == 0:
            if len(predicted_ents) == 0:
                true_negatives += 1
            else:
                false_positives += 1
        else:
            found = False
            for ent in predicted_ents:
                # at least one entity is discovered correctly
                if ent in true_ents:
                    true_positives += 1
                    found = True
                    break
            if not found:
                false_negatives += 1
    f1_score = true_positives / (true_positives + 1 / 2 * (false_positives + false_negatives))
    return true_positives / (true_positives + false_negatives), true_negatives / (true_negatives + false_positives), f1_score


print('WNUT2019 uh_ritual')
baseline = read_input_file(os.path.join(BASELINE_DIR, 'baseline'))
for et in ent_types['wnut']:
    print(et)
    et_file = os.path.join(BASELINE_DIR, et)
    with open(et_file, 'r') as _:
        lines = [line.replace('\n', '') for line in _.readlines()]
        ground_truth = {}
        for line in lines:
            tags = [x.split('>') for x in line.split(' ')]
            sent = ' '.join([tag[0] for tag in tags])
            ground_truth[sent] = tags

    sample_baseline = [baseline[gt] for gt in ground_truth]
    assert len(sample_baseline) == len(ground_truth)
    results = eval_wnut_baseline(sample_baseline, ground_truth.values(), et)
    print("acc@p: {:.2f}".format(results[0]))
    print("acc@n: {:.2f}".format(results[1]))
    print("f1score: {:.2f}".format(results[2]))
    print('\n')
