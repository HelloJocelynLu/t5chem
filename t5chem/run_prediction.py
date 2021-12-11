import argparse
import os
from functools import partial

import pandas as pd
import rdkit
import scipy
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from .data_utils import T5ChemTasks, TaskPrefixDataset, data_collator
from .evaluation import get_rank, standize
from .model import T5ForProperty
from .mol_tokenizers import AtomTokenizer, SelfiesTokenizer, SimpleTokenizer


def add_args(parser):
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The model path to be loaded.",
    )
    parser.add_argument(
        "--prediction",
        default='',
        type=str,
        help="The file name for prediction.",
    )
    parser.add_argument(
        "--prefix",
        default='',
        type=str,
        help="When provided, use it instead of read from trained model. (Especially useful when trained on a mixed\
            dataset, but want to test on seperate tasks)",
    )
    parser.add_argument(
        "--num_beams",
        default=10,
        type=int,
        help="Number of beams for beam search.",
    )
    parser.add_argument(
        "--num_preds",
        default=5,
        type=int,
        help="The number of independently computed returned sequences for each element in the batch.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for training and validation.",
    )


def predict(args):
    lg = rdkit.RDLogger.logger()  
    lg.setLevel(rdkit.RDLogger.CRITICAL) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = T5Config.from_pretrained(args.model_dir)
    task = T5ChemTasks[config.task_type]
    tokenizer_type = getattr(config, "tokenizer")
    if tokenizer_type == "simple":
        Tokenizer = SimpleTokenizer
    elif tokenizer_type == 'atom':
        Tokenizer = AtomTokenizer
    else:
        Tokenizer = SelfiesTokenizer

    tokenizer = Tokenizer(vocab_file=os.path.join(args.model_dir, 'vocab.pt'))

    if os.path.isfile(args.data_dir):
        args.data_dir, base = os.path.split(args.data_dir)
        base = base.split('.')[0]
    else:
        base = "test"

    testset = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix=args.prefix or task.prefix,
                                    max_source_length=task.max_source_length,
                                    max_target_length=task.max_target_length,
                                    separate_vocab=(task.output_layer != 'seq2seq'),
                                    type_path=base)
    data_collator_padded = partial(data_collator, pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(
        testset, 
        batch_size=args.batch_size,
        collate_fn=data_collator_padded
    )

    targets = []
    if task.output_layer == 'seq2seq':
        task_specific_params = {
            "Reaction": {
            "early_stopping": True,
            "max_length": task.max_target_length,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_preds,
            "decoder_start_token_id": tokenizer.pad_token_id,
            }
        }
        model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
        model.eval()
        model = model.to(device)

        with open(os.path.join(args.data_dir, base+".target")) as rf:
            for line in rf:
                targets.append(standize(line.strip()[:task.max_target_length]))
    
        predictions = [[] for i in range(args.num_preds)]
        for batch in tqdm(test_loader, desc="prediction"):
            for k, v in batch.items():
                batch[k] = v.to(device)
            del batch['labels']
            with torch.no_grad():
                outputs = model.generate(**batch, **task_specific_params['Reaction'])
            for i,pred in enumerate(outputs):
                prod = tokenizer.decode(pred, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                predictions[i % args.num_preds].append(prod)

    else:
        predictions = []
        model = T5ForProperty.from_pretrained(args.model_dir)
        model.eval()
        model = model.to(device)

        for batch in tqdm(test_loader, desc="prediction"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            with torch.no_grad():
                outputs = model(**batch)
                targets.extend(batch['labels'].view(-1).to(outputs.logits).tolist())
                predictions.extend((outputs.logits).tolist())

    test_df = pd.DataFrame(targets, columns=['target'])

    if isinstance(predictions[0], list):
        for i, preds in enumerate(predictions):
            test_df['prediction_{}'.format(i + 1)] = preds
            test_df['prediction_{}'.format(i + 1)] = \
                test_df['prediction_{}'.format(i + 1)].apply(standize)
        test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'prediction_', args.num_preds), axis=1)

        correct = 0
        invalid_smiles = 0
        for i in range(1, args.num_preds+1):
            correct += (test_df['rank'] == i).sum()
            invalid_smiles += (test_df['prediction_{}'.format(i)] == '').sum()
            print('Top-{}: {:.1f}% || Invalid {:.2f}%'.format(i, correct/len(test_df)*100, \
                invalid_smiles/len(test_df)/i*100))
    elif task.output_layer == 'regression':
        test_df['prediction'] = predictions
        MAE = mean_absolute_error(test_df['target'], test_df['prediction'])      
        MSE = mean_squared_error(test_df['target'], test_df['prediction'])
        slope, intercept, r_value, p_value, std_err = \
            scipy.stats.linregress(test_df['prediction'], test_df['target'])
        print("MAE: {}    RMSE: {}    r2: {}    r:{}".format(MAE, MSE**0.5, r_value**2, r_value))
    else:
        test_df['prediction_1'] = predictions
        correct = sum(test_df['prediction_1'] == test_df['target'])
        print('Accuracy: {:.1f}%'.format(correct/len(test_df)*100))

    if not args.prediction:
        args.prediction = os.path.join(args.model_dir, 'predictions.csv')
    test_df.to_csv(args.prediction, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    predict(args)
