import argparse
import os
import random
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from transformers import (T5Config, T5ForConditionalGeneration,
                          TrainingArguments)

from data import T5MolTokenizer, T5SelfiesTokenizer, T5SimpleTokenizer, TaskPrefixDataset, data_collator
from models import EarlyStopTrainer, EMA, T5ForSoftLabel

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
        "--lr_scheduler",
        default='linear',
        help="learning rate scheduler to use (linear, cosine or constant)",
    )
    parser.add_argument(
        "--task_prefix",
        default='Product:',
        help="Prefix of current task. ('Product:', 'Yield:', 'Fill-Mask:')",
    )
    parser.add_argument(
        "--tokenizer",
        default='simple',
        help="Tokenizer to use. (Default: 'simple'. Options: 'smiles', 'selfies')",
    )
    parser.add_argument(
        "--vocab_size",
        default=2400,
        type=int,
        help="The max_size of vocabulary.",
    )
    parser.add_argument(
        "--max_source_length",
        default=300,
        type=int,
        help="The maximum source length after tokenization.",
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
        "--EMA",
        action="store_true",
        help="Whether to use EMA shadow model.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.",
    )
    parser.add_argument(
        "--log_steps",
        default=500,
        type=int,
        help="Number of update steps between two logs.",
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


def CalMSELoss(PredictionOutput, tokenizer):
    predictions = PredictionOutput.predictions
    label_ids = PredictionOutput.label_ids/100
    loss_fcn = nn.MSELoss()
    loss = loss_fcn(torch.Tensor(preds), torch.Tensor(labels))
    return {'mse_loss': loss.item()}

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    torch.cuda.manual_seed(8570)
    np.random.seed(8570)
    torch.manual_seed(8570)
    # this one is needed for torchtext random call (shuffled iterator)
    # in multi gpu it ensures datasets are read in the same order
    random.seed(8570)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

    if args.tokenizer == "smiles":
        Tokenizer = T5MolTokenizer
    elif args.tokenizer == 'simple':
        Tokenizer = T5SimpleTokenizer
    else:
        Tokenizer = T5SelfiesTokenizer

    tokenizer = Tokenizer(args.vocab, max_size=args.vocab_size)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))

    dataset = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix=args.task_prefix,
                                    max_source_length=args.max_source_length,
                                    separate_vocab=True,
                                    type_path="train")

    do_eval = os.path.exists(os.path.join(args.data_dir, 'val.source'))
    if do_eval:
        eval_strategy = "steps"
        eval_iter = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                      prefix=args.task_prefix,
                                      max_source_length=args.max_source_length,
                                      separate_vocab=True,
                                      type_path="val")
    else:
        eval_strategy = "no"
        eval_iter = None

    model = T5ForSoftLabel.from_pretrained(args.pretrain)

    model.config.tie_word_embeddings=False

    if args.EMA:
        model = EMA(model, 0.999)

    compute_metrics = partial(CalMSELoss, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        fp16=args.fp16,
        evaluation_strategy=eval_strategy,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.log_steps,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        prediction_loss_only=(compute_metrics is None),
    )

    trainer = EarlyStopTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator_pad1,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        compute_metrics = compute_metrics,
        optimizers = (None, args.lr_scheduler)
    )

    trainer.train()
    print(args)
    print("logging dir: {}".format(training_args.logging_dir))
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
