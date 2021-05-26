import os
import argparse
import subprocess
import pandas as pd

from rdkit import Chem

def add_args(parser):
    parser.add_argument(
        "prediction",
        type=str,
        require=True,
        help="The path to prediction result to be evaluate."
    )
    parser.add_argument(
        "data",
        type=str,
        require=True,
        help="The path to ground truth folder."
    )


def cansmiles(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except:
        return 'error:'+smiles


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    sources, targets = [], []
    with open(os.path.join(args.data, 'test.source')) as rf:
        for line in rf:
            sources.append(Chem.CanonSmiles(line.strip()))

    with open(os.path.join(args.data, 'test.target')) as rf:
        for line in rf:
            target.append(Chem.CanonSmiles(line.strip()))

    with open('../synthesis/reference.can', 'w') as wf:
        for x,y in zip(sources, targets):
            print(x+','+y, file=wf)

    preds = pd.read_csv(args.prediction)
    n_preds = len(preds.columns)
    preds.columns = ['input', 'target']+['target_'+str(i) for i in range(n_preds)]
    preds['input'] = sources
    preds = preds.applymap(cansmiles)
    preds.to_csv('../synthesis/results.can', index=False)
    
    subprocess.call(['perl', '../synthesis/compare.pl',
        '../synthesis/reference.can',
        '../synthesis/results.can', 1])

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
