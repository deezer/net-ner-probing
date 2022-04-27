import os
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def get_prompts(dataset_id):
    """
    Get prompts per dataset
    """
    per_prompts = ['a person', 'a character', 'an animal']
    loc_prompts = ['a country', 'a city', 'a place', 'a location']
    org_prompts = ['an organization', 'a company', 'a group', 'an institution', 'a club', 'a corporation']
    woa_prompts = ['a movie', 'a book', 'a song', 'a title', 'a work']
    if dataset_id == 'conll':
        return per_prompts + org_prompts + loc_prompts
    elif dataset_id == 'wnut':
        return per_prompts + loc_prompts + ['a corporation', 'a group', 'a product'] + woa_prompts
    elif dataset_id == 'dbp':
        return per_prompts + org_prompts + loc_prompts + woa_prompts
    elif dataset_id == 'mitmovie':
        return per_prompts + woa_prompts
    return []


def ppl_batch(lines, model, tokenizer, device='cpu'):
    """
    Compute perplexity
    """
    lines = [''.join([tokenizer.bos_token, line]) for line in lines]
    tokenizer.pad_token = tokenizer.eos_token
    tok_res = tokenizer.batch_encode_plus(lines, return_tensors='pt', padding=True)
    input_ids = tok_res['input_ids']
    attention_mask = tok_res['attention_mask']
    lines_len = torch.sum(tok_res['attention_mask'], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=input_ids.to(device))
        logits = outputs[1].detach().cpu()

    ppls = []
    for line_ind in range(len(lines)):
        line_log_prob = 0.0
        for token_ind in range(lines_len[line_ind] - 1):
            token_prob = softmax(logits[line_ind, token_ind])
            token_id = input_ids[line_ind, token_ind + 1]
            line_log_prob -= np.log(token_prob[token_id])
        ppls.append(np.exp(line_log_prob / (lines_len[line_ind] - 1)).item())
    return ppls


if __name__ == "__main__":
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--in_file', dest='in_file', action='store', required=True, help='the path of the input file')
    parser.add_argument('--device', dest='device', action='store', required=True, help='cpu or cuda')
    args = parser.parse_args()
    args = vars(args)
    in_file = args['in_file']
    device = args['device']

    if 'conll' in in_file.lower():
        dataset_id = 'conll'
    elif 'wnut' in in_file.lower():
        dataset_id = 'wnut'
    elif 'dbp' in in_file.lower():
        dataset_id = 'dbp'
    elif 'mit' in in_file.lower():
        dataset_id = 'mitmovie'
    else:
        print('Dataset id not found')
        exit(1)
    print(dataset_id)

    entities = []
    with open(in_file, 'r') as _:
        lines = [line.replace('\n', '') for line in _.readlines()]
        for line in lines:
            entities.append(line.split('\t')[0])

    model_id = 'gpt2-medium'
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    ppls = {}
    prompts = get_prompts(dataset_id)
    for e in tqdm(entities):
        lines = []
        for p in prompts:
            lines.append(e + ' is ' + p)
        ppls[e] = dict(zip(prompts, ppl_batch(lines, model, tokenizer, device)))

    df = pd.DataFrame(ppls).T
    out_file = os.path.splitext(in_file)[0] + '_typing.tsv'
    df.to_csv(out_file, header=True, index=True, sep='\t')
    print("--- %s seconds ---" % (time.time() - start_time))
