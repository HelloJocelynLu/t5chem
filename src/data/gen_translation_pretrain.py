# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import pandas as pd
from EFGs import standize
from rdkit import Chem
from tqdm import tqdm

from .data_utils import MolTokenizer


def add_args(parser):
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="The raw data file (source csv file).",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Whether the input *.csv file contains a header",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Number of samples as validation. If smaller than 1, would be len(data)*ratio, default=0.05",
    )

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args()
    if args.header:
        all_data = pd.read_csv(args.data_file)
    else:
        all_data = pd.read_csv(args.data_file, header=None)

    shuffled_idx = np.random.permutation(len(all_data))
    if args.val_ratio < 1:
        num_val = int(len(shuffled_idx)*args.val_ratio)
    else:
        num_val = int(args.val_ratio)

    eval_idx = shuffled_idx[:num_val]
    train_idx = shuffled_idx[num_val:]

    train_samples = all_data.iloc[train_idx]
    eval_samples = all_data.iloc[eval_idx]

    dir_name = os.path.dirname(args.data_file)
    error = open(os.path.join(dir_name,'error.txt'), 'w')
    phases = {'train':train_samples, 'val':eval_samples}

    tokenizer = MolTokenizer()
    for phase in phases:
        source = open(os.path.join(dir_name, phase+'.source'), 'w')
        target = open(os.path.join(dir_name, phase+'.target'), 'w')
#        for smiles in tqdm(phases[phase]['smiles'], desc=phase):
        for smiles in tqdm(phases[phase][0], desc=phase):
            try:
                can_smiles = standize(smiles)
                _ = tokenizer.tokenize(can_smiles)
                if not can_smiles: continue
                random_subsmiles = []
                for sub_smiles in smiles.split('.'):
                    random_subsmiles.append(gen_randomSmiles(sub_smiles))
                _ = tokenizer.tokenize('.'.join(random_subsmiles))
                print('.'.join(random_subsmiles), file=source)
                print(can_smiles, file=target)
            except:
                print(smiles, file=error)
        source.close()
        target.close()
    error.close()

def gen_randomSmiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)


if __name__ == "__main__":                                                      
        main()
