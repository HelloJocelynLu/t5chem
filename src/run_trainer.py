import argparse
import os
import random
from functools import partial

import torch
import torch.nn as nn
from transformers import (T5Config, T5ForConditionalGeneration,
                          TrainingArguments)
from transformers.optimization import AdamW, get_constant_schedule

from data import T5MolTokenizer, TaskPrefixDataset, data_collator
from models import EarlyStopTrainer

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
        "--pretrain",
        default='',
        help="Load from a pretrained model.",
    )
    parser.add_argument(
        "--vocab",
        default='',
        help="Vocabulary file to load.",
    )
    parser.add_argument(
        "--continued", action="store_true",
        help="Whether this training should be resumed from pretrained model.",
    )
    parser.add_argument(
        "--task_prefix",
        default='Product:',
        help="Prefix of current task. ('Product:', 'Yield:', 'Fill-Mask:')",
    )
    parser.add_argument(
        "--max_source_length",
        default=300,
        type=int,
        help="The maximum source length after tokenization.",
    )
    parser.add_argument(
        "--max_target_length",
        default=100,
        type=int,
        help="The maximum target length after tokenization.",
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
        default=30,
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
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--constant_lr",
        action="store_true",
        help="Whether to apply constant learning rate.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.",
    )
    parser.add_argument(
        "--save_steps",
        default=100000,
        type=int,
        help="Checkpoints of model would be saved every setting number of steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        default=2,
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
        tokenizer = T5MolTokenizer(vocab_file=args.vocab)
    else:
        tokenizer = T5MolTokenizer(source_files=[os.path.join(args.data_dir, x) for
                                 x in ('train.target', 'train.source')])
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))

    dataset = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix=args.task_prefix,
                                    max_source_length=args.max_source_length,
                                    max_target_length=args.max_target_length,
                                    type_path="train")

    eval_iter = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix=args.task_prefix,
                                    max_source_length=args.max_source_length,
                                    max_target_length=args.max_target_length,
                                    type_path="val")

    config = T5Config(
        vocab_size=len(tokenizer.vocab),
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_past=True,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        )

    if not args.pretrain:
        model = T5ForConditionalGeneration(config)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
        if model.config.vocab_size != len(tokenizer):
            model.config = config
            model.resize_token_embeddings(len(tokenizer))

    data_collator_pad1 = partial(data_collator, pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        fp16=args.fp16,
        evaluate_during_training=True,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
    )

    if args.constant_lr:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=training_args.learning_rate,
            eps=training_args.adam_epsilon,
        )
        lr_scheduler = get_constant_schedule(
            optimizer
        )
    else:
        optimizer = None
        lr_scheduler = None

    trainer = EarlyStopTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator_pad1,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        prediction_loss_only=True,
        optimizers=(optimizer, lr_scheduler),
    )
    
    if not args.continued:
        trainer.train()
    else:
        trainer.train(model_path=args.pretrain)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
