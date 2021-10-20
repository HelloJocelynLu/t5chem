import argparse

import pandas as pd
import rdkit
import scipy
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error


def add_args(parser):
    parser.add_argument(
        "--prediction",
        type=str,
        required=True,
        help="The path to prediction result to be evaluate.",
    )
    parser.add_argument(
        "--average",
        default=0,
        type=int,
        help="whether to use average if multiple predictions exist."
    )
    parser.add_argument(
        "--type",
        type=str,
        default='text',
        help="The type of predictions (text/value)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Whether to compare simple string only",
    )


def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0

def standize(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except:
        return ''
        
def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    lg = rdkit.RDLogger.logger()  
    lg.setLevel(rdkit.RDLogger.CRITICAL) 

    print('evaluating {}'.format(args.prediction))
    predictions = pd.read_csv(args.prediction)
    predictions.fillna('', inplace=True)
    num_preds = len([x for x in predictions.columns if 'prediction' in x])

    if args.type == 'text':
        predictions = predictions.astype(str)
        if not args.simple:
            for i in range(1, num_preds+1):
                predictions['prediction_{}'.format(i)] = predictions['prediction_{}'.format(i)].apply(standize)
        predictions['rank'] = predictions.apply(lambda row: get_rank(row, 'prediction_', num_preds), axis=1)

        correct = 0
        invalid_smiles = 0
        for i in range(1, num_preds+1):
            correct += (predictions['rank'] == i).sum()
            invalid_smiles += (predictions['prediction_{}'.format(i)] == '').sum()
            print('Top-{}: {:.1f}% || Invalid {:.2f}%'.format(i, correct/len(predictions)*100,
                                                                     invalid_smiles/len(predictions)/i*100))
    
    else:
        MAE = mean_absolute_error(predictions['target'], predictions['prediction'])      
        MSE = mean_squared_error(predictions['target'], predictions['prediction'])
        slope, intercept, r_value, p_value, std_err = \
            scipy.stats.linregress(predictions['prediction'], predictions['target'])
        print("MAE: {}    RMSE: {}    r2: {}    r:{}".format(MAE, MSE**0.5, r_value**2, r_value))

if __name__ == "__main__":
    main()
