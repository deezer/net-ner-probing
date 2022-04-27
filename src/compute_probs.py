import os
import argparse
import csv
import torch
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def probs_batch(lines, model, tokenizer, device='cpu'):
    """
    Compute string probablities and perplexity
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

    probs = []
    ppls = []
    for line_ind in range(len(lines)):
        line_probs = []
        line_log_prob = 0
        for token_ind in range(lines_len[line_ind] - 1):
            token_prob = softmax(logits[line_ind, token_ind])
            token_id = input_ids[line_ind, token_ind + 1]
            line_probs.append(token_prob[token_id].item())
            line_log_prob -= np.log(token_prob[token_id])
        tokens = tokenizer.convert_ids_to_tokens(input_ids[line_ind])[1:]
        probs.append(list(zip(tokens, line_probs)))
        ppls.append(np.exp(line_log_prob / (lines_len[line_ind] - 1)).item())

    return probs, ppls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--input_file', dest='input_file', action='store', required=True, help='the path of the input file')
    parser.add_argument('--output_file', dest='output_file', action='store', required=True, help='the path of the output file')
    parser.add_argument('--device', dest='device', action='store', required=True, help='cpu or cuda')
    args = parser.parse_args()
    args = vars(args)
    in_file = args['input_file']
    out_file = args['output_file']
    device = args['device']

    with open(in_file, 'r') as _:
        entities = [line.replace('\n', '').replace('"', '') for line in _.readlines()]

    model_id = 'gpt2-medium'
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    probs = []
    ppls = []
    bsize = 32
    no_ents = len(entities)
    for i in tqdm(range(0, no_ents, bsize)):
        batch = entities[i: min(no_ents, i + bsize)]
        result = probs_batch(batch, model, tokenizer, device)
        probs.extend(result[0])
        ppls.extend(result[1])

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        for key, ppl, values in zip(entities, ppls, probs):
            writer.writerow([key, ppl] + values)
