import argparse
import os
import random
from functools import partial

import torch
import torch.nn as nn
from transformers import T5Config, TrainingArguments

from data import MolTokenizer, YieldDataset, data_collator_yield
from models import T5ForRegression, EarlyStopTrainer


def add_args(parser):
    parser.add_argument(
        "--data_files",
        required=True,
        nargs='+',
        help="The input data files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--val_file",
        default='',
        help="Optional. If additional validation set is available.",
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
        "--copy_all_w", action="store_true",
        help="Whether to copy all weights from pretrain.",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Whether to evaluate.",
    )
    parser.add_argument(
        "--only_readout", action="store_true",
        help="Only train read out layer",
    )
    parser.add_argument(
        "--mode",
        default='sigmoid',
        type=str,
        help="lm_head to set. (sigmoid/linear1/linear2)",
    )
    parser.add_argument(
        "--split_idx",
        default=[2767],
        type=int,
        nargs='+',
        help="The index to separate training(, validation) and test set.",
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
        default=80,
        type=int,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
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
        "--save_steps",
        default=1000,
        type=int,
        help="Checkpoints of model would be saved every setting number of steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        default=5,
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

    lm_heads_layer = {
        'sigmoid': nn.Sequential(nn.Linear(args.d_model, 1), nn.Sigmoid()),
        'sigmoid2': nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.ReLU(), nn.Linear(args.d_model,1), nn.Sigmoid()),
        'linear2': nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.ReLU(), nn.Linear(args.d_model,1)),
        'linear1': nn.Sequential(nn.Linear(args.d_model, 1)),
    }

    if args.vocab:
        tokenizer = MolTokenizer(vocab_file=args.vocab)
    else:
        tokenizer = MolTokenizer(source_files=[os.path.join(os.path.dirname(
            args.data_files[0]), 'train.txt')])
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))

    result_log = open(os.path.join(args.output_dir, 'test.log'), 'w')
    print(args, file=result_log)
    for file in args.data_files:
        dataset = YieldDataset(tokenizer, file, type_path="train", sep_id=args.split_idx)
        if args.val_file:
            eval_iter = YieldDataset(tokenizer, args.val_file, sep_id=[0])
        else:
            eval_iter = YieldDataset(tokenizer, file, type_path="val", sep_id=args.split_idx)

        if not args.pretrain:
            config = T5Config(                                                    
                vocab_size=len(tokenizer.vocab),                                  
                pad_token_id=tokenizer.pad_token_id,                              
                decoder_input_ids=tokenizer.bos_token_id,                         
                eos_token_id=tokenizer.eos_token_id,                              
                bos_token_id=tokenizer.bos_token_id,                                                     
                output_past=True,                                                 
                num_layers=args.num_layers,                                       
                num_heads=args.num_heads,                                         
                d_model=args.d_model,                                             
                )                                                                 

            model = T5ForRegression(config)
            model.set_lm_head(lm_heads_layer[args.mode])
        else:                                                                     
            model = T5ForRegression.from_pretrained(args.pretrain)     
            model.resize_token_embeddings(len(tokenizer.vocab))
            model.set_lm_head(lm_heads_layer[args.mode])
            if args.copy_all_w:
                model.load_state_dict(torch.load(os.path.join(args.pretrain,
                    'pytorch_model.bin'), map_location=lambda storage, loc: storage))

        data_collator_pad1 = partial(data_collator_yield,
                                     pad_token_id=tokenizer.pad_token_id,
                                     percentage=('sigmoid' in args.mode),
                                    )

        output_dir = os.path.join(args.output_dir, args.mode, os.path.basename(file).split('.')[0])
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=args.eval,
            evaluate_during_training=args.eval,
            num_train_epochs=args.num_epoch,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            learning_rate=args.learning_rate,
        )

        if args.only_readout:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.lm_head.parameters():
                param.requires_grad = True

        trainer = EarlyStopTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator_pad1,
            train_dataset=dataset,
            eval_dataset=eval_iter,
            prediction_loss_only=True,
        )

        trainer.train()
        pred_output = trainer.predict(eval_iter)
        print(file, pred_output.metrics['eval_loss'], file=result_log)
        trainer.save_model(output_dir)

    result_log.close()

if __name__ == "__main__":
    main()
