import argparse
import os
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import T5Config

from data import T5MolTokenizer, T5SelfiesTokenizer, T5SimpleTokenizer, MultiTaskPrefixDataset, data_collator
from models import T5ForMultitaskLabel

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
        "--task_prefix",
        default='Product:',
        help="Prefix of current task. ('Product:', 'Yield:', 'Fill-Mask:')",
    )
    parser.add_argument(
        "--tokenizer",
        default='simple',
        help="Tokenizer to use. (Default: 'smiles'. 'selfies')",
    )
    parser.add_argument(
        "--max_source_length",
        default=500,
        type=int,
        help="The maximum source length after tokenization.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for training and validation.",
    )


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = T5ForMultitaskLabel.from_pretrained(args.model_dir)
#    archive_file = os.path.join(args.model_dir, "pytorch_model.bin")
#    model.load_state_dict(torch.load(archive_file, map_location="cpu"))
    model.eval()

    if args.tokenizer == "smiles":
        Tokenizer = T5MolTokenizer
    elif args.tokenizer == 'simple':
        Tokenizer = T5SimpleTokenizer
    else:
        Tokenizer = T5SelfiesTokenizer

    tokenizer = Tokenizer(vocab_file=os.path.join(args.model_dir, 'vocab.pt'), max_size=model.config.vocab_size)

    if os.path.isfile(args.data_dir):
        args.data_dir, base = os.path.split(args.data_dir)
        base = base.split('.')[0]
    else:
        base = "test"

    testset = MultiTaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix=args.task_prefix,
                                    max_source_length=args.max_source_length,
                                    type_path=base)
    data_collator_pad1 = partial(data_collator, pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             collate_fn=data_collator_pad1)

    yield_label, yield_pred = [], []
    class_label, class_pred = [], []
    pred_base = args.prediction or os.path.join(args.model_dir, 'predictions.csv')
    yield_prediction = pred_base.split('.')[0]+"_yield.csv"
    class_prediction = pred_base.split('.')[0]+"_class.csv"

    model = model.to(device)
    for batch in tqdm(test_loader, desc="prediction"):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            class_label.extend(batch['labels'][:,0].long().tolist())
            yield_label.extend(batch['labels'][:,1].tolist())
            class_pred.extend(outputs.logits[:,0].long().tolist())
            yield_pred.extend(outputs.logits[:,1].tolist())

    test_df1 = pd.DataFrame(class_label, columns=['target'])
    test_df1['prediction_1'] = class_pred
    test_df1.to_csv(class_prediction, index=False)

    test_df2 = pd.DataFrame(yield_label, columns=['target'])
    test_df2['prediction_1'] = yield_pred
    test_df2.to_csv(yield_prediction, index=False)

if __name__ == "__main__":
    main()
