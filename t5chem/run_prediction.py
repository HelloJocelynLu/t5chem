import argparse
import os
from functools import partial

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from .data_utils import T5ChemTasks, TaskPrefixDataset, data_collator
from .model import T5ForProperty
from .tokenizers import AtomTokenizer, SelfiesTokenizer, SimpleTokenizer


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


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

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
                                    prefix=task.prefix,
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
                targets.append(line.strip()[:args.max_target_length])
    
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
    else:
        test_df['prediction_1'] = predictions

    if not args.prediction:
        args.prediction = os.path.join(args.model_dir, 'predictions.csv')
    test_df.to_csv(args.prediction, index=False)

if __name__ == "__main__":
    main()
