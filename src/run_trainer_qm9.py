import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from EFGs import standize
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from .data import MolTokenizer
from transformers import T5Config, Trainer, TrainingArguments

from torch.utils.data import Dataset
from typing import Callable, Iterable, List

class QM9Dataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path="train",
        max_source_length=500,
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        # FIXME: the rstrip logic strips all the chars, it seems.
        tok_name = tokenizer.__class__.__name__.lower().rstrip("tokenizer")

        data_path = os.path.join(data_dir, type_path + ".csv")
        dataframe = pd.read_csv(data_path, header=None)
        self.source = []
        for text in tqdm(dataframe[1], desc=f"Tokenizing {data_path}"):
            can_smiles = standize(text)
            tokenized = tokenizer(
                [can_smiles],
                max_length=max_source_length,
                padding="do_not_pad",
                truncation=True,
                return_tensors='pt',
            )
            self.source.append(tokenized)

        self.target = [float(x) for x in dataframe[2]]
        
        if n_obs is not None:
            self.source = self.source[:n_obs]
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": torch.LongTensor([self.pad_token_id]),
                "labels": torch.tensor([target_ids])}

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


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
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--vocab",
        default='',
        help="Vocabulary file to load.",
    )
    parser.add_argument(                                                        
        "--pretrain",                                                           
        default='',                                                             
        help="Load from a pretrained model.",                                   
    ) 
    parser.add_argument(
        "--max_length",
        default=300,
        type=int,
        help="The maximum length (for both source and target) after tokenization.",
    )
    parser.add_argument(
        "--num_layers",
        default=4,
        type=int,
        help="Number of hidden layers in the Transformer encoder.",
    )
    parser.add_argument(
        "--num_heads",
        default=8,
        type=int,
        help="Number of attention heads for each attention layer in the Transformer encoder.",
    )
    parser.add_argument(
        "--d_model",
        default=256,
        type=int,
        help="Size of the encoder layers and the pooler layer.",
    )
    parser.add_argument(
        "--num_epoch",
        default=150,
        type=int,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--save_steps",
        default=10000,
        type=int,
        help="Checkpoints of model would be saved every setting number of steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        default=20,
        type=int,
        help="The maximum number of chackpoints to be kept.",
    )


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    torch.manual_seed(8570) 
    # this one is needed for torchtext random call (shuffled iterator) 
    # in multi gpu it ensures datasets are read in the same order 
    random.seed(8570) 
    # some cudnn methods can be random even after fixing the seed 
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

    if args.vocab:
        tokenizer = MolTokenizer(vocab_file=args.vocab)
    else:
        tokenizer = MolTokenizer(source_files=[os.path.join(args.data_dir,
                                 'train.txt')])
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer.save_vocabulary(os.path.join(args.data_dir, 'vocab.pt'))

    dataset = QM9Dataset(tokenizer, args.data_dir, type_path="train")
    eval_iter = QM9Dataset(tokenizer, data_dir=args.data_dir, type_path="val")

    if not args.pretrain:                                                       
        task_specific_params = {                                                
            "Reaction": {                                                       
              "early_stopping": True,                                           
              "max_length": args.max_length,                                    
              "num_beams": 5,                                                   
              "prefix": "Predict reaction outcomes"                             
            }                                                                   
        }                                                                       
        config = T5Config(                                                      
            vocab_size=len(tokenizer.vocab),                                    
            pad_token_id=tokenizer.pad_token_id,                                
            decoder_input_ids=tokenizer.bos_token_id,                           
            eos_token_id=tokenizer.eos_token_id,                                
            bos_token_id=tokenizer.bos_token_id,                                
            task_specific_params=task_specific_params,                          
            output_past=True,                                                   
            num_layers=args.num_layers,                                         
            num_heads=args.num_heads,                                           
            d_model=args.d_model,                                               
            )                                                                   
                                                                                
        model = T5ForRegression(config)
        model.set_lm_head(nn.Linear(config.d_model, 1))
    else:                                                                       
        model = T5ForRegression.from_pretrained(args.pretrain)       
        model.resize_token_embeddings(len(tokenizer.vocab))
        model.set_lm_head(nn.Linear(args.d_model, 1))
#         model.set_lm_head(nn.Sequential(nn.Linear(args.d_model, args.d_model), 
#                           nn.ReLU(), nn.Linear(args.d_model,1)))

    data_collator_pad1 = partial(data_collator, pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator_pad1,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        prediction_loss_only=True,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
