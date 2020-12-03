from EFGs import standize
import argparse
import scipy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np


def add_args(parser):
    parser.add_argument(
        "--prediction",
        type=str,
        required=True,
        help="The path to prediction result to be evaluate.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default='text',
        help="The type of predictions (text/value)",
    )


def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == standize(row['{}{}'.format(base, i)]):
            return i
    return 0


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    print('evaluating {}'.format(args.prediction))
    predictions = pd.read_csv(args.prediction)
    num_preds = len(predictions.columns)-1

    if args.type == 'text':
        predictions['rank'] = predictions.apply(lambda row: get_rank(row, 'prediction_', num_preds), axis=1)

        correct = 0
        for i in range(1, num_preds+1):
            correct += (predictions['rank'] == i).sum()
            invalid_smiles = (predictions['prediction_{}'.format(i)] == '').sum()
            print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct/len(predictions)*100,
                                                                     invalid_smiles/len(predictions)*100))
    
    else:
        predictions['prediction_1'] = pd.to_numeric(predictions['prediction_1'], errors='coerce')
        predictions = predictions.replace(np.nan, 0, regex=True)
        MAE = mean_absolute_error(predictions['target'], predictions['prediction_1'])      
        MSE = mean_squared_error(predictions['target'], predictions['prediction_1'])
        slope, intercept, r_value, p_value, std_err = \
            scipy.stats.linregress(predictions['prediction_1'], predictions['target'])
        print("MAE: {}    MSE: {}    r2: {}".format(MAE, MSE**0.5, r_value**2))

if __name__ == "__main__":
    main()
