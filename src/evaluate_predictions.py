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


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    print('evaluating {}'.format(args.prediction))
    predictions = pd.read_csv(args.prediction)
    predictions.fillna('', inplace=True)
    num_preds = len(predictions.columns)-1

    if args.type == 'text':
        predictions = predictions.astype(str)
        if not args.simple:
            for i in range(1, num_preds+1):
                predictions['prediction_{}'.format(i)] = predictions['prediction_{}'.format(i)].apply(standize)
        predictions['rank'] = predictions.apply(lambda row: get_rank(row, 'prediction_', num_preds), axis=1)

        correct = 0
        for i in range(1, num_preds+1):
            correct += (predictions['rank'] == i).sum()
            invalid_smiles = (predictions['prediction_{}'.format(i)] == '').sum()
            print('Top-{}: {:.1f}% || Invalid {:.2f}%'.format(i, correct/len(predictions)*100,
                                                                     invalid_smiles/len(predictions)*100))
    
    else:
        for i in range(1, num_preds+1):
            predictions['prediction_{}'.format(i)] = pd.to_numeric(predictions['prediction_{}'.format(i)], errors='coerce')
        predictions = predictions.replace(np.nan, 0, regex=True)
        if args.average > num_preds:
            print("WARNING: only {} predictions exists, but {} required. Will use all.".format(num_preds, args.average))
            args.average = num_preds
        if args.average == 0:
            args.average = num_preds
        predictions['prediction'] = predictions[['prediction_{}'.format(i) for i in range(1,args.average+1)]].mean(1)
        MAE = mean_absolute_error(predictions['target'], predictions['prediction'])      
        MSE = mean_squared_error(predictions['target'], predictions['prediction'])
        slope, intercept, r_value, p_value, std_err = \
            scipy.stats.linregress(predictions['prediction'], predictions['target'])
        print("MAE: {}    RMSE: {}    r2: {}".format(MAE, MSE**0.5, r_value**2))

if __name__ == "__main__":
    main()
