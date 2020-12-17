# -*- coding: utf-8 -*-
import argparse
import os
import random

import rdkit
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from data_utils import MolTokenizer


def add_args(parser):
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="The raw data file (train.source)",
    )
    parser.add_argument(
        "--multiple",
        type=int,
        default=3,
        help="How many times raw data should be augmented.",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Whether the input *.csv file contains a header",
    )

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args()
    if args.header:
        all_data = pd.read_csv(args.data_file)
        target_file = pd.read_csv(args.data_file.replace('.source', '.target'))
    else:
        all_data = pd.read_csv(args.data_file, header=None)
        target_file = pd.read_csv(args.data_file.replace('.source', '.target'),
                                  header=None)

    dir_name = os.path.dirname(args.data_file)
    target_dir = dir_name+'_X'+str(args.multiple)
    os.makedirs(target_dir, exist_ok=True)
    error = open(os.path.join(target_dir,'error.txt'), 'w')
    source = open(os.path.join(target_dir, 'train.source'), 'w')
    target = open(os.path.join(target_dir, 'train.target'), 'w')

    tokenizer = MolTokenizer()
    for i, rxn in enumerate(tqdm(all_data[0], desc=args.data_file)):
        rct, rgs, prod = rxn.split('>')
        random_rxns = {rxn}
        for _ in range(args.multiple):
            mols = rct.split('.')
            random.shuffle(mols)
            rct = '.'.join([gen_randomSmiles(x) for x in mols])
            mols = rgs.split('.')
            random.shuffle(mols)
            rgs = '.'.join([gen_randomSmiles(x) for x in mols])
            mols = prod.split('.')
            random.shuffle(mols)
            prod = '.'.join([gen_randomSmiles(x) for x in mols])
            random_rxns.add('>'.join([rct, rgs, prod]))
        for rxn in random_rxns:
            try:
                _ = tokenizer.tokenize(rxn)
                print(rxn, file=source)
                print(target_file[0][i], file=target)
            except:
                print(rxn, file=error)
    source.close()
    target.close()
    error.close()

def gen_randomSmiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    main()
