import os
import argparse
import torch
from .data import T5MolTokenizer, LineByLineTextDataset
from transformers import T5Config, T5ForConditionalGeneration, Trainer, TrainingArguments,\
    DataCollatorForLanguageModeling


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

    if args.vocab:
        tokenizer = T5MolTokenizer(vocab_file=args.vocab, mask_token='<mask>')
    else:
        files = [os.path.join(args.data_dir,x+'.txt') for x in ['train', 'val']]
        tokenizer = T5MolTokenizer(source_files=files, mask_token='<mask>')

    dataset = LineByLineTextDataset(tokenizer=tokenizer, 
                                  file_path=os.path.join(args.data_dir,'train.txt'),
                                  block_size=args.max_length,
                                  )
    eval_iter = LineByLineTextDataset(tokenizer=tokenizer, 
                                    file_path=os.path.join(args.data_dir,'val.txt'),
                                    block_size=args.max_length,
                                    )

    if args.continued:
        model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
    else:
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

        model = T5ForConditionalGeneration(config)
        if args.pretrain:
            model.load_state_dict(torch.load(os.path.join(args.pretrain,
                'pytorch_model.bin'), map_location=lambda storage, loc: storage))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

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
        learning_rate=args.learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        prediction_loss_only=True,
    )

    if not args.continued:
        trainer.train()
    else:
        trainer.train(model_path=args.pretrain)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
