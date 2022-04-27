import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# conll filepaths
conll_all = {
    'person': 'output/entity_typing/conll.person_typing.tsv',
    'location': 'output/entity_typing/conll.location_typing.tsv',
    'organization': 'output/entity_typing/conll.organisation_typing.tsv'
}
conll_seen = {
    'person': 'output/entity_typing/conll.person_seen_0.8_0_typing.tsv',
    'location': 'output/entity_typing/conll.location_seen_0.8_0_typing.tsv',
    'organization': 'output/entity_typing/conll.organisation_seen_0.8_0_typing.tsv'
}
conll_rare = {
    'person': 'output/entity_typing/conll.person_rare_unseen_0.0001_0_typing.tsv',
    'location': 'output/entity_typing/conll.location_rare_unseen_0.0001_0_typing.tsv',
    'organization': 'output/entity_typing/conll.organisation_rare_unseen_0.0001_0_typing.tsv'
}

# wnut filepaths
wnut_all = {
    'person': 'output/entity_typing/wnut.person_typing.tsv',
    'location': 'output/entity_typing/wnut.location_typing.tsv',
    'corporation': 'output/entity_typing/wnut.corporation_typing.tsv',
    'group': 'output/entity_typing/wnut.group_typing.tsv',
    'product': 'output/entity_typing/wnut.product_typing.tsv',
    'work': 'output/entity_typing/wnut.creative-work_typing.tsv'
}

# mitmove filepaths
mitmovie_all = {
    'person': 'output/entity_typing/mitmovie.person_typing.tsv',
    'work': 'output/entity_typing/mitmovie.creative-work_typing.tsv'
}
mitmovie_seen = {
    'person': 'output/entity_typing/mitmovie.person_seen_0_0.001_typing.tsv',
    'work': 'output/entity_typing/mitmovie.creative-work_seen_0_0.001_typing.tsv'
}
mitmovie_rare = {
    'person': 'output/entity_typing/mitmovie.person_rare_unseen_0_1e-05_typing.tsv',
    'work': 'output/entity_typing/mitmovie.creative-work_rare_unseen_0_1e-05_typing.tsv'
}

# dbpedia filepaths
dbp_seen_word_prunning = {
    'person': 'output/entity_typing/dbp.person_seen_1.0_0_typing.tsv',
    'location': 'output/entity_typing/dbp.location_seen_1.0_0_typing.tsv',
    'organization': 'output/entity_typing/dbp.organisation_seen_1.0_0_typing.tsv',
    'work': 'output/entity_typing/dbp.creative-work_seen_1.0_0_typing.tsv'
}
dbp_seen_trans_prunning = {
    'person': 'output/entity_typing/dbp.person_seen_0_0.01_typing.tsv',
    'location': 'output/entity_typing/dbp.location_seen_0_0.01_typing.tsv',
    'organization': 'output/entity_typing/dbp.organisation_seen_0_0.01_typing.tsv',
    'work': 'output/entity_typing/dbp.creative-work_seen_0_0.01_typing.tsv'
}
dbp_rare_word_prunning = {
    'person': 'output/entity_typing/dbp.person_rare_unseen_1e-06_0_typing.tsv',
    'location': 'output/entity_typing/dbp.location_rare_unseen_1e-06_0_typing.tsv',
    'organization': 'output/entity_typing/dbp.organisation_rare_unseen_1e-06_0_typing.tsv',
    'work': 'output/entity_typing/dbp.creative-work_rare_unseen_1e-06_0_typing.tsv'
}
dbp_rare_trans_prunning = {
    'person': 'output/entity_typing/dbp.person_rare_unseen_0_1e-06_typing.tsv',
    'location': 'output/entity_typing/dbp.location_rare_unseen_0_1e-06_typing.tsv',
    'organization': 'output/entity_typing/dbp.organisation_rare_unseen_0_1e-06_typing.tsv',
    'work': 'output/entity_typing/dbp.creative-work_rare_unseen_0_1e-06_typing.tsv'
}

per_prompts = ['a person', 'a character', 'an animal']
loc_prompts = ['a country', 'a city', 'a place', 'a location']
org_prompts = ['an organization', 'a company', 'a group', 'an institution', 'a club', 'a corporation']
woa_prompts = ['a movie', 'a book', 'a song', 'a title', 'a work']


def process_predictions(dfs, key_to_id, f_map_pred, drop_labels=[]):
    df_index = []
    df_predicted = []
    df_T = []
    df_P = []
    for ent in dfs:
        dfs[ent] = dfs[ent].drop(drop_labels, axis=1, errors='ignore')
        df_index.extend(dfs[ent].index.tolist())

        if 'predicted' not in dfs[ent].columns:
            dfs[ent]['predicted'] = dfs[ent].idxmin(axis=1)
        df_predicted.extend(dfs[ent]['predicted'].tolist())

        if 'T' not in dfs[ent].columns:
            dfs[ent]['T'] = key_to_id[ent]
        df_T.extend(dfs[ent]['T'].tolist())

        if 'P' not in dfs[ent].columns:
            dfs[ent]['P'] = dfs[ent].predicted.map(f_map_pred)
        df_P.extend(dfs[ent]['P'].tolist())

    df = pd.DataFrame(index=df_index)
    df['predicted'] = df_predicted
    df['T'] = df_T
    df['P'] = df_P
    return df


def show_report(df, key_to_id):
    print(classification_report(df['T'].tolist(), df['P'].tolist(), target_names=key_to_id.keys()))
    cm = confusion_matrix(df['T'].tolist(), df['P'].tolist())
    cm_df = pd.DataFrame(cm, index=key_to_id.keys(), columns=key_to_id.keys())
    print(cm_df)


print("*** Conll dataset ***")
dfs = {}
for ent_type in conll_all:
    dfs[ent_type] = pd.read_csv(conll_all[ent_type], sep='\t', index_col=0)

key_to_id_conll = dict(zip(dfs.keys(), range(len(dfs))))
def pred_class_conll(s, key_to_id=key_to_id_conll):
    if s in per_prompts:
        return key_to_id['person']
    elif s in loc_prompts:
        return key_to_id['location']
    elif s in org_prompts:
        return key_to_id['organization']

df = process_predictions(dfs, key_to_id_conll, pred_class_conll, drop_labels=[])
show_report(df, key_to_id_conll)

print("\n*** Conll dataset SEEN ***")
dfs = {}
for ent_type in conll_seen:
    dfs[ent_type] = pd.read_csv(conll_seen[ent_type], sep='\t', index_col=0)
df = process_predictions(dfs, key_to_id_conll, pred_class_conll, drop_labels=[])
show_report(df, key_to_id_conll)

print("\n*** Conll dataset RARE ***")
dfs = {}
for ent_type in conll_rare:
    dfs[ent_type] = pd.read_csv(conll_rare[ent_type], sep='\t', index_col=0)
df = process_predictions(dfs, key_to_id_conll, pred_class_conll, drop_labels=[])
show_report(df, key_to_id_conll)

print("\n*** Wnut dataset ***")
dfs = {}
for ent_type in wnut_all:
    dfs[ent_type] = pd.read_csv(wnut_all[ent_type], sep='\t', index_col=0)

key_to_id_wnut = dict(zip(dfs.keys(), range(len(dfs))))
def pred_class_wnut(s, key_to_id=key_to_id_wnut):
    if s in per_prompts:
        return key_to_id_wnut['person']
    elif s in loc_prompts:
        return key_to_id_wnut['location']
    elif s == 'a corporation':
        return key_to_id_wnut['corporation']
    elif s == 'a group':
        return key_to_id_wnut['group']
    elif s == 'a product':
        return key_to_id_wnut['product']
    elif s in woa_prompts:
        return key_to_id_wnut['work']

df = process_predictions(dfs, key_to_id_wnut, pred_class_wnut, drop_labels=[])
show_report(df, key_to_id_wnut)

print("\n*** Mitmovie dataset ***")
dfs = {}
for ent_type in  mitmovie_all:
    dfs[ent_type] = pd.read_csv(mitmovie_all[ent_type], sep='\t', index_col=0)

key_to_id_mitmovie = dict(zip(dfs.keys(), range(len(dfs))))
def pred_class_mitmovie(s, key_to_id=key_to_id_mitmovie):
    if s in per_prompts:
        return key_to_id['person']
    else:
        return key_to_id['work']

df = process_predictions(dfs, key_to_id_mitmovie, pred_class_mitmovie, drop_labels=[])
show_report(df, key_to_id_mitmovie)

print("\n*** Mitmovie dataset SEEN ***")
dfs = {}
for ent_type in  mitmovie_seen:
    dfs[ent_type] = pd.read_csv(mitmovie_seen[ent_type], sep='\t', index_col=0)
df = process_predictions(dfs, key_to_id_mitmovie, pred_class_mitmovie, drop_labels=[])
show_report(df, key_to_id_mitmovie)

print("\n*** Mitmovie dataset RARE ***")
dfs = {}
for ent_type in  mitmovie_rare:
    dfs[ent_type] = pd.read_csv(mitmovie_rare[ent_type], sep='\t', index_col=0)
df = process_predictions(dfs, key_to_id_mitmovie, pred_class_mitmovie, drop_labels=[])
show_report(df, key_to_id_mitmovie)

print("\n*** DBpedia dataset SEEN WORD ***")
dfs = {}
for ent_type in dbp_seen_word_prunning:
    dfs[ent_type] = pd.read_csv(dbp_seen_word_prunning[ent_type], sep='\t', index_col=0)

key_to_id_dbp = dict(zip(dfs.keys(), range(len(dfs))))
def pred_class_dbp(s, key_to_id=key_to_id_dbp):
    if s in per_prompts:
        return key_to_id['person']
    elif s in loc_prompts:
        return key_to_id['location']
    elif s in org_prompts:
        return key_to_id['organization']
    else:
        return key_to_id['work']

df = process_predictions(dfs, key_to_id_dbp, pred_class_dbp, drop_labels=[])
show_report(df, key_to_id_dbp)

print("\n*** DBpedia dataset RARE WORD ***")
dfs = {}
for ent_type in dbp_rare_word_prunning:
    dfs[ent_type] = pd.read_csv(dbp_rare_word_prunning[ent_type], sep='\t', index_col=0)
df = process_predictions(dfs, key_to_id_dbp, pred_class_dbp, drop_labels=[])
show_report(df, key_to_id_dbp)

print("\n*** DBpedia dataset SEEN TRANSITION ***")
dfs = {}
for ent_type in dbp_seen_trans_prunning:
    dfs[ent_type] = pd.read_csv(dbp_seen_trans_prunning[ent_type], sep='\t', index_col=0)
df = process_predictions(dfs, key_to_id_dbp, pred_class_dbp, drop_labels=[])
show_report(df, key_to_id_dbp)

print("\n*** DBpedia dataset RARE TRANSITION ***")
dfs = {}
for ent_type in dbp_rare_trans_prunning:
    dfs[ent_type] = pd.read_csv(dbp_rare_trans_prunning[ent_type], sep='\t', index_col=0)
df = process_predictions(dfs, key_to_id_dbp, pred_class_dbp, drop_labels=[])
show_report(df, key_to_id_dbp)
