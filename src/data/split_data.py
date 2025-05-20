import torch as t
from torch.utils.data import random_split
from src.data.dataset import load_data
import os
import json
import sys
from pathlib import Path
import numpy as np
import random


def load_data_and_split(
    data_dir: str,
    proportions: list[float] = [1.0, 0.0],
    seed: int = 0,
):
    data = load_data(data_dir)

    n_data = len(data)
    n_train_data = int(n_data*proportions[0])
    n_val_data = int(n_data*proportions[1])
    n_test_data = n_data - n_train_data - n_val_data    

    train_data, val_data, test_data = random_split(
        dataset= data,
        lengths= [n_train_data, n_val_data, n_test_data],
        generator=t.Generator().manual_seed(seed)
    )
    return train_data, val_data, test_data


def split_and_generate(
    task1: str,
    task2: str,
    seed: int = 0
) -> tuple[list, list, list]:
    if f'{task1}_{task2}' in ['antonym_uppercase', 'synonym_uppercase']:
        s1_train, s1_val, _ = load_data_and_split(f'data/base/{task1}.json', proportions=[0.3, 0.7], seed= seed)
        _, s2_val, _ = load_data_and_split(f'data/base/{task2}.json', proportions=[0.3, 0.7], seed= seed)
        composite_data = []
        for d in s1_train:
            entry = [
                d[0],
                d[1].capitalize(),
                d[1],
                d[0].capitalize(),
            ]
            composite_data.append(entry)
        return s1_val, s2_val, composite_data
    
    elif f'{task1}_{task2}' in ['country_capital_lowercase', 'product_company_lowercase', 'landmark_country_lowercase']:
        s1_train, s1_val, _ = load_data_and_split(f'data/base/{task1}.json', proportions=[0.3, 0.7], seed= seed)
        _, s2_val, _ = load_data_and_split(f'data/base/{task2}.json', proportions=[0.3, 0.7], seed= seed)
        composite_data = []
        for d in s1_train:
            entry = [
                d[0],
                d[1].lower(),
                d[1],
                d[0].lower(),
            ]
            composite_data.append(entry)
        return s1_val, s2_val, composite_data
    
    elif f'{task1}_{task2}' in ['choose_last_landmark_country', 'choose_last_country_capital']:
        _, s1_val, _ = load_data_and_split(f'data/base/{task1}.json', proportions=[0.3, 0.7], seed= seed)
        s2_train, s2_val, _ = load_data_and_split(f'data/base/{task2}.json', proportions=[0.3, 0.7], seed= seed)
        composite_data = []
        for d in s2_train:
            random_pairs = np.random.choice(len(s2_train), 2, replace=False)
            entry = [
                f"{s2_train[random_pairs[0]][0]}, {s2_train[random_pairs[1]][0]}, {d[0]}",
                d[1],
                d[0],
                s2_train[random_pairs[0]][1],
            ]
            composite_data.append(entry)
        return s1_val, s2_val, composite_data
    
    elif f'{task1}_{task2}' == 'choose_last_uppercase':
        s1_train, s1_val, _ = load_data_and_split(f'data/base/{task1}.json', proportions=[0.3, 0.7], seed= seed)
        _, s2_val, _ = load_data_and_split(f'data/base/{task2}.json', proportions=[0.3, 0.7], seed= seed)
        composite_data = []
        for d in s1_train:
            split_word = d[0].split(', ')
            entry = [
                d[0],
                d[1].capitalize(),
                d[1],
                split_word[0].capitalize(),
            ]
            composite_data.append(entry)
        return s1_val, s2_val, composite_data
    
    elif f'{task1}_{task2}' == 'choose_last_first_letter': 
        s1_train, s1_val, _ = load_data_and_split(f'data/base/{task1}.json', proportions=[0.3, 0.7], seed= seed)
        _, s2_val, _ = load_data_and_split(f'data/base/{task2}.json', proportions=[0.3, 0.7], seed= seed)
        composite_data = []
        for d in s1_train:
            split_word = d[0].split(', ')
            entry = [
                d[0],
                d[1][0],
                d[1],
                split_word[0][0],
            ]
            composite_data.append(entry)
        return s1_val, s2_val, composite_data

    elif f'{task1}_{task2}' in ['adjective_v_verb_antonym', 'adjective_v_verb_synonym']: 
        _, s1_val, _ = load_data_and_split(f'data/base/{task1}.json', proportions=[0.3, 0.7], seed= seed)
        s2_train, s2_val, _ = load_data_and_split(f'data/base/{task2}.json', proportions=[0.3, 0.7], seed= seed)
        with open('data/base/vocab/adjective.json', "r") as f:
            adj_vocab = json.load(f)
        with open('data/base/vocab/noun.json', "r") as f:
            n_vocab = json.load(f)
        with open('data/base/vocab/verb.json', "r") as f:
            v_vocab = json.load(f)

        adj_data = []
        v_data = []
        for d in s2_train:
            if (d[0] in adj_vocab) and (d[0] not in n_vocab) and (d[0] not in v_vocab) and (d[1] in adj_vocab) and (d[1] not in n_vocab) and (d[1] not in v_vocab):
                adj_data.append(d)
            if (d[0] in v_vocab) and (d[0] not in n_vocab) and (d[0] not in adj_vocab) and (d[1] in v_vocab) and (d[1] not in n_vocab) and (d[1] not in adj_vocab):
                v_data.append(d)

        composite_data = []
        for d in adj_data:
            chosen_two = random.sample(v_data, 2)
            combined = [d] + chosen_two
            random.shuffle(combined)
            entry = [
                f"{combined[0][0]}, {combined[1][0]}, {combined[2][0]}",
                d[1],
                d[0],
                combined[0][1],
            ]
            composite_data.append(entry)
        return s1_val, s2_val, composite_data
    
    elif f'{task1}_{task2}' in ['antonym_english_french', 'landmark_country_english_french']:
        s1_train, s1_val, _ = load_data_and_split(f'data/base/{task1}.json', proportions=[0.3, 0.7], seed= seed)
        _, s2_val, _ = load_data_and_split(f'data/base/{task2}.json', proportions=[0.3, 0.7], seed= seed)
        with open("data/translations.json", "r") as f:
            translations = json.load(f)
        composite_data = []
        for d in s1_train:
            entry = [
                d[0],
                translations[d[1]]['fr'],
                d[1],
                translations[d[0]]['fr'],
            ]
            composite_data.append(entry)
        return s1_val, s2_val, composite_data
    
    elif f'{task1}_{task2}' in ['antonym_english_spanish', 'landmark_country_english_spanish']:
        s1_train, s1_val, _ = load_data_and_split(f'data/base/{task1}.json', proportions=[0.3, 0.7], seed= seed)
        _, s2_val, _ = load_data_and_split(f'data/base/{task2}.json', proportions=[0.3, 0.7], seed= seed)
        with open("data/translations.json", "r") as f:
            translations = json.load(f)
        composite_data = []
        for d in s1_train:
            entry = [
                d[0],
                translations[d[1]]['es'],
                d[1],
                translations[d[0]]['es'],
            ]
            composite_data.append(entry)
        return s1_val, s2_val, composite_data
    
    return None, None, None