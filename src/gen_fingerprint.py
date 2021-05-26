import argparse
import os
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import T5Config

from data import T5MolTokenizer, T5SelfiesTokenizer, T5SimpleTokenizer, TaskPrefixDataset, data_collator
from models import T5ForSoftLabel

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
        "--output",
        default='',
        type=str,
        help="The file name for output file.",
    )
    parser.add_argument(
        "--task_prefix",
        default='Product:',
        help="Prefix of current task. ('Product:', 'Yield:', 'Fill-Mask:')",
    )
    parser.add_argument(
        "--tokenizer",
        default='smiles',
        help="Tokenizer to use. (Default: 'smiles'. 'selfies')",
    )
    parser.add_argument(
        "--max_source_length",
        default=300,
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

    model = T5ForSoftLabel.from_pretrained(args.model_dir)
    model.lm_head = None
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

    testset = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix=args.task_prefix,
                                    max_source_length=args.max_source_length,
                                    max_target_length=1,
                                    separate_vocab=True,
                                    type_path=base)
    data_collator_pad1 = partial(data_collator, pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             collate_fn=data_collator_pad1)

    predictions = []
    if not args.output:
        args.output = os.path.join(args.data_dir, base+'_ff.pt')
    
    model = model.to(device)
    for batch in tqdm(test_loader, desc="prediction"):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            predictions.append(outputs.logits)

    predictions = torch.cat(predictions, dim=0)
    torch.save(predictions.cpu().numpy(), args.output)

if __name__ == "__main__":
    main()
