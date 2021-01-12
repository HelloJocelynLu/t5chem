import argparse
import os
import random
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from transformers import (T5Config, T5ForConditionalGeneration,
                          TrainingArguments)

from data import T5MolTokenizer, T5SelfiesTokenizer, TaskPrefixDataset, data_collator
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
        "--lr_scheduler",
        default='linear',
        help="learning rate scheduler to use (linear, cosine or constant)",
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


def CalMSELoss(PredictionOutput, tokenizer):
    predictions = PredictionOutput.predictions
    predictions = torch.argmax(torch.Tensor(predictions[0]), -1)
    label_ids = PredictionOutput.label_ids
    preds, labels = [], []
    for pred, label in zip(predictions, label_ids):
        try:
            num_pred = float(tokenizer.decode(pred, skip_special_tokens=True, 
                clean_up_tokenization_spaces=False))
        except ValueError:
            num_pred = 0
        preds.append(num_pred)
        labels.append(float(tokenizer.decode(label, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)))
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
    else:
        Tokenizer = T5SelfiesTokenizer

    tokenizer = Tokenizer(args.vocab)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))

    dataset = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix=args.task_prefix,
                                    max_source_length=args.max_source_length,
                                    max_target_length=args.max_target_length,
                                    type_path="train")

    do_eval = os.path.exists(os.path.join(args.data_dir, 'val.source'))
    if do_eval:
        eval_strategy = "steps"
        eval_iter = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                      prefix=args.task_prefix,
                                      max_source_length=args.max_source_length,
                                      max_target_length=args.max_target_length,
                                      type_path="val")
    else:
        eval_strategy = "no"
        eval_iter = None

    config = T5Config(
        vocab_size=len(tokenizer),
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
    if args.task_prefix == 'Yield:':
        compute_metrics = partial(CalMSELoss, tokenizer=tokenizer)
    else:
        compute_metrics = None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        fp16=args.fp16,
        evaluation_strategy=eval_strategy,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
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
    
    if not args.continued:
        trainer.train()
    else:
        trainer.train(model_path=args.pretrain)
    print(args)
    print("logging dir: {}".format(training_args.logging_dir))
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
