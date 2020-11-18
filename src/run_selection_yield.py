import argparse
import os
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import T5Config, TrainingArguments

from data import MolTokenizer, YieldDataset, data_collator_yield
from models import EarlyStopTrainer, T5ForRegression


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
        "--mode",
        default='sigmoid',
        type=str,
        help="lm_head to set. (sigmoid/linear1/linear2)",
    )
    parser.add_argument(
        "--split_idx",
        default=[2768],
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
        default=30,
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
        default=10000,
        type=int,
        help="Checkpoints of model would be saved every setting number of steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        default=0,
        type=int,
        help="The maximum number of chackpoints to be kept.",
    )


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    print(args)
    portion_dict = {0:'Ideal: ', 1:'5%: ', 2:'10%: ', 3:'20%: '}
    for file in args.data_files:
        # 5%: 198 10%: 396 20%: 791
        df = pd.read_csv(file, sep='\s+', header=None)
        len_05 = [int(len(df)*0.05)]
        len_10 = [int(len(df)*0.1)]
        len_20 = [int(len(df)*0.2)]
        results = torch.zeros(len(df), 4)
        results[:, 0]=torch.from_numpy(df[1].to_numpy())
        for idx, sep in enumerate([len_05, len_10, len_20]):
            dataset = YieldDataset(tokenizer, file, type_path="train", sep_id=sep)
            testset = YieldDataset(tokenizer, file, type_path="test", sep_id=sep)

            if not args.pretrasin:
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
            test_loader = DataLoader(testset, batch_size=args.batch_size,               
                         collate_fn=data_collator_pad1)

            output_dir = os.path.join(args.output_dir, args.mode, os.path.basename(file).split('.')[0])
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                do_train=True,
                num_train_epochs=args.num_epoch,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                save_steps=args.save_steps,
                save_total_limit=args.save_total_limit,
                learning_rate=args.learning_rate,
            )

            trainer = EarlyStopTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator_pad1,
                train_dataset=dataset,
                prediction_loss_only=True,
            )

            trainer.train()
            trainer.save_model(output_dir)
            model = model.eval()
            x = []
            for batch in tqdm(test_loader, desc="prediction"):

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                outputs = model(**batch)
    #            loss += outputs[0].item()*len(batch['labels'])
                for pred in outputs[1]:
                    x.append(pred.item())
            results[sep[0]:,idx+1] = torch.tensor(x)
        # top 10 reactions
        print("------------- Select 10 reactions --------------")
        val, idx = torch.topk(results, 10, dim=0)
        for i in range(4):
            mean = torch.index_select(results[:,0], 0, idx[:,i]).mean().item()
            std = torch.index_select(results[:,0], 0, idx[:,i]).std().item()
            print('train', portion_dict[i], mean, '±', std, '%')
        # top 50 reactions
        print("------------- Select 50 reactions --------------")
        val, idx = torch.topk(results, 50, dim=0)
        for i in range(4):
            mean = torch.index_select(results[:,0], 0, idx[:,i]).mean().item()
            std = torch.index_select(results[:,0], 0, idx[:,i]).std().item()
            print('train', portion_dict[i], mean, '±', std, '%')
        # top 100 reactions
        print("------------- Select 100 reactions --------------")
        val, idx = torch.topk(results, 100, dim=0)
        for i in range(4):
            mean = torch.index_select(results[:,0], 0, idx[:,i]).mean().item()
            std = torch.index_select(results[:,0], 0, idx[:,i]).std().item()
            print('train', portion_dict[i], mean, '±', std, '%')

if __name__ == "__main__":
    main()
