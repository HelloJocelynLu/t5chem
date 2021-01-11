from EFGs import standize
import argparse
import pandas as pd
import numpy as np

from evaluate_predictions import get_rank


def add_args(parser):
    parser.add_argument(
        "--prediction",
        type=str,
        required=True,
        help="The path to prediction result to be evaluate.",
    )
    parser.add_argument(
        "--topk",
        default=5,
        type=int,
        help="evaluate top-k score rxnk."
    )

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    print('evaluating {}'.format(args.prediction))
    predictions = pd.read_csv(args.prediction)
    num_preds = len(predictions.columns)-1

    def gather_topk(df):
        scores = {}
        for i in range(1, num_preds+1):
            for pred in df['prediction_{}'.format(i)]:
                pred = standize(pred)
                scores[pred] = scores.get(pred, 0) + 1/(1+0.001*(i-1))
        if '' in scores: del scores['']
        return pd.Series(sorted(scores, key=scores.get, reverse=True)[:args.topk])
    
    predictions['target'] = predictions['target'].apply(standize)
    group = predictions.groupby(by='target')
    score_rank = group.apply(gather_topk).unstack()
    score_rank.reset_index(level=0, inplace=True)
    score.columns = ['target']+['prediction_{}'.format(i) for i in range(1, num_preds+1)]
    score_rank['rank'] = score_rank.apply(lambda row: get_rank(row, 'prediction_', num_preds), axis=1)

    correct = 0
    for i in range(1, num_preds+1):
        correct += (score_rank['rank'] == i).sum()
        invalid_smiles = (score_rank['prediction_{}'.format(i)] == '').sum()
        print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct/len(score_rank)*100,
                                                                 invalid_smiles/len(predictions)*100))

if __name__ == "__main__":
    main()
