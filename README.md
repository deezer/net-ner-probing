# Probing Pre-trained Auto-regressive Language Models for Named Entity Typing and Recognition

This repository contains the code and data to reproduce the experiments from the article **Probing Pre-trained Auto-regressive Language Models for Named Entity Typing and Recognition** presented at [LREC 2022](https://lrec2022.lrec-conf.org/en/).

## Data

The directory `data` contains the data required for the named entity typing (NET) and named entity recognition (NER) experiments from four corpora: *CoNLL-2003*, *MIT-Movie*, *WNUT2017* and *DBpedia* (only for NET).

In `data/entity_typing`, we can find files containing lists of named entities for each named entity type per corpus.

In `data/entity_typing`, we can find the train and test splits of the three corpora, as provided by the original works. The train set is used only to sample examples provided during few-shot NER.


## Installation

```bash
git clone git@github.com:deezer/net-ner-probing.git
cd net-ner-probing
```

Most of the experiments can be run on CPU. However, the time is significantly longer. Therefore, we encourage the use of a GPU environment with CUDA installed, especially for the NER experiments.

In our experiments, we used cuda11.0 on a 1x GTX 1080 with 11GB RAM (Driver Version: 460.84).

The directory `docker` contains a docker image (`Dockerfile`), extra packages required for the experiments (`requirements.txt`) and two example scripts that should be modified accordingly for building the docker image (`build.sh`) and for running the docker image as a container (`run.sh`).

## NET Experiments

Select seen and rare/unseen named entities for CoNLL-2003 dataset:
```bash
cd ../src
python compute_probs.py --input_file=../data/entity_typing/CoNLL-2003/person.tsv --output_file=output/entity_typing/conll.person.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/CoNLL-2003/location.tsv --output_file=output/entity_typing/conll.location.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/CoNLL-2003/organisation.tsv --output_file=output/entity_typing/conll.organisation.tsv --device=cuda
python select_seen_unseen.py --input_file=output/entity_typing/conll.person.tsv --seen_word_exp=.8 --rare_word_exp=1e-04
python select_seen_unseen.py --input_file=output/entity_typing/conll.location.tsv --seen_word_exp=.8 --rare_word_exp=1e-04
python select_seen_unseen.py --input_file=output/entity_typing/conll.organisation.tsv --seen_word_exp=.8 --rare_word_exp=1e-04
```

Select seen and rare/unseen named entities for Mit Movie dataset:
```bash
cd ../src
python compute_probs.py --input_file=../data/entity_typing/MIT-Movie/person.tsv --output_file=output/entity_typing/mitmovie.person.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/MIT-Movie/creative-work.tsv --output_file=output/entity_typing/mitmovie.creative-work.tsv --device=cuda
python select_seen_unseen.py --input_file=output/entity_typing/mitmovie.person.tsv --seen_tran_exp=.001 --rare_tran_exp=1e-05
python select_seen_unseen.py --input_file=output/entity_typing/mitmovie.creative-work.tsv --seen_tran_exp=.001 --rare_tran_exp=1e-05
```

Compute probabilities for WNUT2017 dataset:
```bash
cd ../src
python compute_probs.py --input_file=../data/entity_typing/WNUT2017/corporation.tsv --output_file=output/entity_typing/wnut.corporation.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/WNUT2017/creative-work.tsv --output_file=output/entity_typing/wnut.creative-work.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/WNUT2017/group.tsv --output_file=output/entity_typing/wnut.group.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/WNUT2017/location.tsv --output_file=output/entity_typing/wnut.location.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/WNUT2017/person.tsv --output_file=output/entity_typing/wnut.person.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/WNUT2017/product.tsv --output_file=output/entity_typing/wnut.product.tsv --device=cuda
```

Select seen and rare/unseen named entities for DBpedia dataset:
```bash
cd ../src
python compute_probs.py --input_file=../data/entity_typing/DBpedia/person.tsv --output_file=output/entity_typing/dbp.person.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/DBpedia/location.tsv --output_file=output/entity_typing/dbp.location.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/DBpedia/organisation.tsv --output_file=output/entity_typing/dbp.organisation.tsv --device=cuda
python compute_probs.py --input_file=../data/entity_typing/DBpedia/creative-work.tsv --output_file=output/entity_typing/dbp.creative-work.tsv --device=cuda
python select_seen_unseen.py --input_file=output/entity_typing/dbp.person.tsv --seen_word_exp=1 --rare_word_exp=1e-06
python select_seen_unseen.py --input_file=output/entity_typing/dbp.location.tsv --seen_word_exp=1 --rare_word_exp=1e-06
python select_seen_unseen.py --input_file=output/entity_typing/dbp.organisation.tsv --seen_word_exp=1 --rare_word_exp=1e-06
python select_seen_unseen.py --input_file=output/entity_typing/dbp.creative-work.tsv --seen_word_exp=1 --rare_word_exp=1e-06
python select_seen_unseen.py --input_file=output/entity_typing/dbp.person.tsv --seen_tran_exp=0.01 --rare_tran_exp=1e-06
python select_seen_unseen.py --input_file=output/entity_typing/dbp.location.tsv --seen_tran_exp=0.01 --rare_tran_exp=1e-06
python select_seen_unseen.py --input_file=output/entity_typing/dbp.organisation.tsv --seen_tran_exp=0.01 --rare_tran_exp=1e-06
python select_seen_unseen.py --input_file=output/entity_typing/dbp.creative-work.tsv --seen_tran_exp=0.01 --rare_tran_exp=1e-06

```

Reproducing Figure 1 (the filepaths declared in the beginning of the file `analyze_dbp_ppl.py` may need to be updated to your case):
```bash
python analyze_dbp_ppl.py
```

Named entity typing for CoNLL-2003 dataset:
```bash
python entity_typing.py --in_file=output/entity_typing/conll.person.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/conll.person_rare_unseen_0.0001_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/conll.person_seen_0.8_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/conll.location.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/conll.location_rare_unseen_0.0001_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/conll.location_seen_0.8_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/conll.organisation.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/conll.organisation_rare_unseen_0.0001_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/conll.organisation_seen_0.8_0.tsv --device=cuda
```

Named entity typing for Mit Movie dataset:
```bash
python entity_typing.py --in_file=output/entity_typing/mitmovie.person.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/mitmovie.person_rare_unseen_0_1e-05.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/mitmovie.person_seen_0_0.001.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/mitmovie.creative-work.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/mitmovie.creative-work_rare_unseen_0_1e-05.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/mitmovie.creative-work_seen_0_0.001.tsv --device=cuda
```

Named entity typing for WNUT2017 dataset:
```bash
python entity_typing.py --in_file=output/entity_typing/wnut.corporation.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/wnut.creative-work.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/wnut.group.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/wnut.location.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/wnut.person.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/wnut.product.tsv --device=cuda
```

Named entity typing for DBpedia dataset:
```bash
python entity_typing.py --in_file=output/entity_typing/dbp.person_rare_unseen_0_1e-06.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.person_rare_unseen_1e-06_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.location_rare_unseen_0_1e-06.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.location_rare_unseen_1e-06_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.organisation_rare_unseen_0_1e-06.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.organisation_rare_unseen_1e-06_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.creative-work_rare_unseen_0_1e-06.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.creative-work_rare_unseen_1e-06_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.person_seen_0_0.01.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.person_seen_1.0_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.location_seen_0_0.01.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.location_seen_1.0_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.organisation_seen_0_0.01.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.organisation_seen_1.0_0.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.creative-work_seen_0_0.01.tsv --device=cuda
python entity_typing.py --in_file=output/entity_typing/dbp.creative-work_seen_1.0_0.tsv --device=cuda
```

Evaluating NET (the filepaths declared in the beginning of the file `evaluate_entity_typing.py` may need to be updated to your case):
```bash
python evaluate_entity_typing.py
```

## NER Experiments

NER for WNUT2017 dataset:
```bash
python run_extraction.py --models='gpt2-medium' --datasets='wnut_corporation' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
python run_extraction.py --models='gpt2-medium' --datasets='wnut_corporation' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='seen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_corporation' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='unseen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_creative-work' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
python run_extraction.py --models='gpt2-medium' --datasets='wnut_creative-work' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='seen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_creative-work' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='unseen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_group' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
python run_extraction.py --models='gpt2-medium' --datasets='wnut_group' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='seen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_group' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='unseen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_location' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
python run_extraction.py --models='gpt2-medium' --datasets='wnut_location' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='seen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_location' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='unseen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_person' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
python run_extraction.py --models='gpt2-medium' --datasets='wnut_person' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='seen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_person' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='unseen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_product' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
python run_extraction.py --models='gpt2-medium' --datasets='wnut_product' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='seen'
python run_extraction.py --models='gpt2-medium' --datasets='wnut_product' --num_seeds=3 --all_shots=16 --api_num_log_prob=20 --modify_test='unseen'
```

Example of NER for CoNLL-2003 dataset:
```bash
python run_extraction.py --models='gpt2-medium' --datasets='conll_PER' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
python run_extraction.py --models='gpt2-medium' --datasets='conll_LOC' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
python run_extraction.py --models='gpt2-medium' --datasets='conll_ORG' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
```

Example of NER for Mit Movie dataset:
```bash
python run_extraction.py --models='gpt2-medium' --datasets='mitmovie_person' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
python run_extraction.py --models='gpt2-medium' --datasets='mitmovie_title' --num_seeds=3 --all_shots=16 --api_num_log_prob=20
```

Evaluating NER (the filepaths declared in the beginning of the file `evaluate_entity_recognition.py` may need to be updated to your case):
```bash
python evaluate_entity_recognition.py
```

## Cite

Please cite our paper if you use this code in your own work:

```BibTeX
@inproceedings{epure2022probing,
  title={Probing Pre-trained Auto-regressive Language Models for Named Entity Typing and Recognition},
  author={Epure, Elena V. and Hennequin, Romain},
  booktitle={the 13th Edition of Language Resources and Evaluation Conference (LREC2022)},
  year={2022}
}
```

