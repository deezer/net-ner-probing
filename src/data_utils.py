import numpy as np


wnut_all_fields = ['person', 'location', 'corporation', 'group', 'product', 'creative-work']
mitmovie_all_fields = ['person', 'title']
conll_all_fields = ['PER', 'LOC', 'ORG']


def sample_data(field_name, in_file, all_fields):
    import re
    assert field_name in all_fields
    all_fields.remove(field_name)
    filter_tags = [f"B-{field}" for field in all_fields] + [f"I-{field}" for field in all_fields] + ["O"]
    target_tags = [f"B-{field_name}", f"I-{field_name}"]

    with open(in_file + '/train', 'r') as f:
        lines = f.readlines()
    train_answers = []
    train_sentences = []
    train_neg_sents = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                has_b_tag = False
                for tag in target_tags:
                    if 'B-' in tag and tag in word:
                        has_b_tag = True
                    word = word.replace('>' + tag, '')
                if has_b_tag:
                    word = '\t' + word
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace('>' + tag, '')
            if '\t' in word:
                word = word.replace('\t', '')
            untagged_line += word + ' '

        untagged_line = re.sub(r"http\S+", "", untagged_line.strip())
        if answer != '':
            ents = answer.split('\t')
            ents.remove('')
            for e in set(ents):
                train_answers.append(e.strip())
                train_sentences.append(untagged_line.strip())
                #print(e, untagged_line.strip())
        else:
            train_neg_sents.append(untagged_line.strip())

    with open(in_file + '/test', 'r') as f:
        lines = f.readlines()
    test_answers = []
    test_sentences = []
    test_neg_sents = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                has_b_tag = False
                for tag in target_tags:
                    if 'B-' in tag and tag in word:
                        has_b_tag = True
                    word = word.replace('>' + tag, '')
                if has_b_tag:
                    word = '\t' + word
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace('>' + tag, '')
            if '\t' in word:
                word = word.replace('\t', '')
            untagged_line += word + ' '
        untagged_line = re.sub(r"http\S+", "", untagged_line)
        if answer.strip() != '':
            ents = set([a.strip() for a in answer.split('\t')])
            ents.remove('')
            #print(ents)
            answer = '\t'.join(ents)
            test_answers.append(answer)
            test_sentences.append(untagged_line.strip())
        else:
            test_neg_sents.append(untagged_line.strip())
    import random
    random.seed(1)
    for s in random.sample(test_neg_sents, int(len(test_sentences) / 2)):
        test_answers.append('none')
        test_sentences.append(s)
        #print(s)
    return train_sentences, train_answers, train_neg_sents, ['none'] * len(train_neg_sents), test_sentences, test_answers


def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """
    if params['dataset'][:4] == 'wnut':
        field_name = params['dataset'][5:]
        orig_train_sentences, orig_train_labels, orig_train_neg_sents, orig_train_neg_labels, orig_test_sentences, orig_test_labels = sample_data(field_name, f'../data/entity_recognition/WNUT2017', wnut_all_fields.copy())
    elif params['dataset'][:8] == 'mitmovie':
        field_name = params['dataset'][9:]
        orig_train_sentences, orig_train_labels, orig_train_neg_sents, orig_train_neg_labels, orig_test_sentences, orig_test_labels = sample_data(field_name, f'../data/entity_recognition/MIT-Movie', mitmovie_all_fields.copy())
    elif params['dataset'][:5] == 'conll':
        field_name = params['dataset'][6:]
        orig_train_sentences, orig_train_labels, orig_train_neg_sents, orig_train_neg_labels, orig_test_sentences, orig_test_labels = sample_data(field_name, f'../data/entity_recognition/CoNLL-2003', conll_all_fields.copy())
    else:
        raise NotImplementedError
    params['prompt_prefix'] = ""
    params["q_prefix"] = "Sentence: "
    params["a_prefix"] = f"{field_name}: "
    params['task_format'] = 'qa'
    params['num_tokens_to_predict'] = 1


    def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
        q_prefix = params["q_prefix"]
        a_prefix = params["a_prefix"]

        prompt = params['prompt_prefix']
        for x, y in zip(train_sentences, train_labels):
            prompt += f"{q_prefix}{x}\n{a_prefix}{y}"
            prompt += "\n\n"

        if test_label_option is None:
            prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1]
        else:
            prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1] + test_label_option
        return prompt

    params['prompt_func'] = prompt_func
    return orig_train_sentences, orig_train_labels, orig_train_neg_sents, orig_train_neg_labels, orig_test_sentences, orig_test_labels
