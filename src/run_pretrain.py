import os
import argparse
import torch
from data import T5MolTokenizer, T5SelfiesTokenizer, T5SimpleTokenizer, LineByLineTextDataset
from transformers import T5Config, T5ForConditionalGeneration, Trainer, TrainingArguments,\
    DataCollatorForLanguageModeling
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
        "--continued", action="store_true",
        help="Whether this training should be resumed from pretrained model.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        type=str,
        help="The path to pretrained vocabulary.",
    )
    parser.add_argument(
        "--vocab_size",
        default=2400,
        type=int,
        help="The max_size of vocabulary.",
    )
    parser.add_argument(
        "--tokenizer",
        default='smiles',
        help="Tokenizer to use. (Default: 'smiles'. 'selfies', 'simple')",
    )
    parser.add_argument(
        "--max_length",
        default=150,
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
        default=1,
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

    if args.tokenizer == "smiles":
        Tokenizer = T5MolTokenizer
    elif args.tokenizer == 'simple':
        Tokenizer = T5SimpleTokenizer
    else:
        Tokenizer = T5SelfiesTokenizer

    tokenizer = Tokenizer(vocab_file=args.vocab, max_size=args.vocab_size)

    dataset = LineByLineTextDataset(tokenizer=tokenizer, 
                                  file_path=os.path.join(args.data_dir,'train.txt'),
                                  block_size=args.max_length,
                                  prefix='Fill-Mask:',
                                  )
    eval_iter = LineByLineTextDataset(tokenizer=tokenizer, 
                                    file_path=os.path.join(args.data_dir,'val.txt'),
                                    block_size=args.max_length,
                                    prefix='Fill-Mask:',
                                    )

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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        evaluation_strategy="steps",
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=2000,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        prediction_loss_only=True,
    )

    trainer = EarlyStopTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_iter,
    )

    if not args.continued:
        trainer.train()
    else:
        trainer.train(model_path=args.pretrain)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
