import argparse
import pandas as pd

from preprocess_utils import FakeRxnChecker, StripTrivalProd, GetYield

def main(opts):
    raw_csv = pd.read_csv(opts.file_name, sep='\t')
    ori_rxn = open(opts.save+'.txt', 'w')
    yield_rxn = open(opts.save+'_yield.txt', 'w')
    qualified_idx = []
    for i,rxn in enumerate(raw_csv['OriginalReaction']):
        if not FakeRxnChecker(rxn.split()[0]):
            qualified_idx.append(i)

    for i,rxn in enumerate(raw_csv['OriginalReaction']):
        if i in qualified_idx:
            rxn = rxn.split()[0]
            reactants, ragents, prods = rxn.split('>')
            if len(prods.split('.')) > 1:
                rxn = StripTrivalProd(rxn)
            print(rxn, file=ori_rxn)
            yield_data = GetYield(raw_csv[['CalculatedYield', 'TextMinedYield']].iloc[i], tolerance=0.1)
            if yield_data:
                print(rxn,'{:.3f}'.format(yield_data), file=yield_rxn)

    ori_rxn.close()
    yield_rxn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse USPTO seq2seq data.') 
    parser.add_argument("file_name", help="file name")
    # Note: Header lines may need to be deleted for a well-formatted csv file.
    parser.add_argument("save", help="file name to save as")
    args = parser.parse_args()
    main(args)

