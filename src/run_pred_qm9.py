import argparse
import os
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
from EFGs import standize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from .data import MolTokenizer
from .models import T5ForRegression
from .run_trainer_qm9 import QM9Dataset, data_collator


def data_collator(batch, pad_token_id):
    whole_batch = {}
    ex = batch[0]
    for key in ex.keys():
        if 'mask' in key:
            padding_value = 0
        else:
            padding_value = pad_token_id
        whole_batch[key] = pad_sequence([x[key] for x in batch],
                                        batch_first=True,
                                        padding_value=padding_value)
    return whole_batch


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
        "--vocab",
        default='',
        help="Vocabulary file to load.",
    )
    parser.add_argument(
        "--max_length",
        default=300,
        type=int,
        help="The maximum length (for both source and target) after tokenization.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training and validation.",
    )


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = MolTokenizer(vocab_file=args.vocab)

    testset = QM9Dataset(tokenizer, data_dir=args.data_dir, type_path="test")
    
    model = T5ForRegression.pretrained_config(args.model_dir)
    model.set_lm_head(nn.Linear(model.config.d_model, 1))
#     model.set_lm_head(nn.Sequential(nn.Linear(model.config.d_model, model.config.d_model), 
#                       nn.ReLU(), nn.Linear(model.config.d_model,1)))
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir,'pytorch_model.bin'),
                                     map_location=lambda storage, loc: storage))

    data_collator_pad1 = partial(data_collator, pad_token_id=tokenizer.pad_token_id)

    test_loader = DataLoader(testset, batch_size=args.batch_size,               
                             collate_fn=data_collator_pad1)
    loss = 0
    out_file = open(os.path.join(args.data_dir, args.prediction), "w")
    print('Predicted,Label', file=out_file)
    for batch in tqdm(test_loader, desc="prediction"):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        outputs = model(**batch)
        loss += outputs[0].item()*len(batch['labels'])
        for pred, label in zip(outputs[1], batch['labels']):
            print(str(pred.item())+','+str(label.item()), file=out_file)

    print('Loss={}'.format(loss/len(test_loader.dataset)))
    out_file.close()

if __name__ == "__main__":
    main()
