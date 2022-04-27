import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dbp_per_file = 'output/entity_typing/dbp.person.tsv'
dbp_loc_file = 'output/entity_typing/dbp.location.tsv'
dbp_org_file = 'output/entity_typing/dbp.organisation.tsv'


def plot(df, x_col, y_col, title):
    """
    Plot error bars using data from the dataframe df
    """
    x = sorted(pd.unique(df[x_col]))
    mean = df[[y_col, x_col]].groupby([x_col]).mean()[y_col].to_numpy()
    std = df[[y_col, x_col]].groupby([x_col]).std(ddof=0)[y_col].to_numpy()
    x = [el for el in x if el <= 25]
    plt.errorbar(x, mean[:len(x)], std[:len(x)], linestyle='None', marker='^')
    plt.title(title)
    plt.show()


def load_data(in_file):
    """
    Load data in a dataframe
    """
    data = []
    with open(in_file, 'r') as _:
        for line in _.readlines():
            row = line.replace('\n', '').split('\t')
            data.append(row[:2] + [len(row[2:])])
    df = pd.DataFrame(data, columns=['name', 'ppl_tok', 'no_toks'])
    df['log_ppl_tok'] = np.log(df['ppl_tok'].astype(float))
    df['no_chars'] = df['name'].map(lambda name: len(name))
    df['log_ppl_char'] = df['log_ppl_tok'] * df['no_toks'] / df['no_chars']
    df['no_words'] = df['name'].map(lambda name: len(name.split(' ')))
    df['log_ppl_word'] = df['log_ppl_tok'] * df['no_toks'] / df['no_words']
    return df


df_per = load_data(dbp_per_file)
df_loc = load_data(dbp_loc_file)
df_org = load_data(dbp_org_file)
plot(df_per, 'no_toks', 'log_ppl_tok', 'person')
plot(df_loc, 'no_toks', 'log_ppl_tok', 'location')
plot(df_org, 'no_toks', 'log_ppl_tok', 'organisation')

# plot(df_per, 'no_words', 'log_ppl_word', 'person')
# plot(df_loc, 'no_words', 'log_ppl_word', 'location')
# plot(df_org, 'no_words', 'log_ppl_word', 'organisation')
