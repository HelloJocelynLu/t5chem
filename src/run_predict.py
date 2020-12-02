import argparse
import os
from functools import partial

import pandas as pd
import torch
from EFGs import standize
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration

from data import T5MolTokenizer, TaskPrefixDataset, data_collator


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
        "--vocab",
        default='',
        help="Vocabulary file to load.",
    )
    parser.add_argument(
        "--prediction",
        default='',
        type=str,
        help="The file name for prediction.",
    )
    parser.add_argument(
        "--max_length",
        default=300,
        type=int,
        help="The maximum length (for both source and target) after tokenization.",
    )
    parser.add_argument(
        "--num_beams",
        default=5,
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
        "--rep_penalty",
        default=1.0,
        type=float,
        help="The parameter for repetition penalty. 1.0 means no penalty.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--invalid_smiles",
        default=True,
        type=bool,
        help="Whether to print invalid smiles statistics.",
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.vocab:
        tokenizer = T5MolTokenizer(vocab_file=args.vocab)
    else:
        tokenizer = T5MolTokenizer(vocab_file=os.path.join(args.model_dir, 'vocab.pt'))

    testset = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix='Product:',
                                    max_source_length=args.max_length,
                                    max_target_length=args.max_length,
                                    type_path="test")
    data_collator_pad1 = partial(data_collator, pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             collate_fn=data_collator_pad1)

    task_specific_params = {
        "Reaction": {
          "early_stopping": True,
          "max_length": args.max_length,
          "num_beams": args.num_beams,
          "num_return_sequences": args.num_preds,
          "prefix": "Predict reaction outcomes",
          "decoder_start_token_id": tokenizer.pad_token_id,
          "repetition_penalty": args.rep_penalty,
        }
    }

    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    targets = []
    with open(os.path.join(args.data_dir, "test.target")) as rf:
        for line in rf:
            targets.append(line.strip())

    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']

    predictions = [[] for i in range(args.num_preds)]
    if not args.prediction:
        args.prediction = os.path.join(args.model_dir, 'predictions.txt')
    
    loss = 0.0
    out_file = open(args.prediction, "w")
    for batch in tqdm(test_loader, desc="prediction"):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        with torch.no_grad():
            outputs = model.generate(**batch, **task_specific_params['Reaction'])
            loss += model(**batch)[0].item()
        for i,pred in enumerate(outputs):
            prod = tokenizer.decode(pred, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            #print(prod, file=out_file)
            predictions[i % args.num_preds].append(standize(prod))
            print(standize(prod), file=out_file)

    print(loss/len(test_loader.dataset), file=out_file)
    out_file.close()
    torch.save(predictions,'USPTO_STEREO_predictions.pt')
    for i, preds in enumerate(tqdm(predictions, desc='ranking')):
        test_df['prediction_{}'.format(i + 1)] = preds

    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'prediction_', args.num_preds), axis=1)

    correct = 0

    for i in range(1, args.num_preds+1):
        correct += (test_df['rank'] == i).sum()
        invalid_smiles = (test_df['prediction_{}'.format(i)] == '').sum()
        if args.invalid_smiles:
            print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct/len(test_df)*100,
                                                                     invalid_smiles/len(test_df)*100))
        else:
            print('Top-{}: {:.1f}%'.format(i, correct / len(test_df) * 100))
    test_df.to_csv(args.prediction.rsplit('.',1)[0]+'.log.csv')


if __name__ == "__main__":
    main()
