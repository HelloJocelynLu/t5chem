import argparse
import os
from functools import partial

import pandas as pd
import scipy
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import T5Config, TrainingArguments, T5ForConditionalGeneration

from data import T5MolTokenizer, YieldDatasetFromList, data_collator
from models import EarlyStopTrainer

def add_args(parser): 
    parser.add_argument(
        "--data_file",
        required=True,
        help="The training data file.",
    )
    parser.add_argument(
        "--test_files",
        required=True,
        nargs='+',
        help="The test data files.",
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
        "--num_epoch",
        default=500,
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

    torch.manual_seed(8570)                                                                           
    torch.backends.cudnn.deterministic = True  
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = T5MolTokenizer(vocab_file=args.vocab)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))

    print(args)
    with open(args.data_file) as rf:
        dataset = YieldDatasetFromList(tokenizer, rf.readlines(), prefix='Yield:')

    if not args.pretrain:                                                      
        config = T5Config(                                                          
            vocab_size=len(tokenizer),                                        
            pad_token_id=tokenizer.pad_token_id,                                    
            decoder_start_token_id=tokenizer.pad_token_id,                               
            eos_token_id=tokenizer.eos_token_id,                                                                                            
            output_past=True,                                                       
            num_layers=4,
            num_heads=8,
            d_model=256,
            )                                                                                                                              
        model = T5ForConditionalGeneration(config)                         
    else:                                                                           
        model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
        if model.config.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    data_collator_pad1 = partial(data_collator,
                                 pad_token_id=tokenizer.pad_token_id,
                                )

    model = model.to(device)

    output_dir = os.path.join(args.output_dir, os.path.basename(args.data_file).split('.')[0])
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
    all_results = []
    for file in args.test_files:
        print(file+':')
        with open(file) as rf:
            testset = YieldDatasetFromList(tokenizer, rf.readlines(), prefix='Yield:')
        results = torch.zeros(len(testset), 3)
        results[:, 0]=torch.Tensor(testset.num_target)
        test_loader = DataLoader(testset, batch_size=args.batch_size,               
                     collate_fn=data_collator_pad1)
        results[:, 1]=results[:,0][torch.randperm(len(testset))]
        
        x = []
        for batch in tqdm(test_loader, desc="prediction"):

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            with torch.no_grad():
                outputs = model.generate(**batch, max_length=5)
                for i,pred in enumerate(outputs):
                    pred = tokenizer.decode(pred, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                    x.append(float(pred))
        results[:,2] = torch.tensor(x)
        torch.save(results, os.path.join(args.output_dir,
            os.path.basename(file).split('.')[0]+'_results.pt'))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                results[:,0], results[:,2])
        print('MAE:',mean_absolute_error(results[:,0], results[:,2]), 'R^2:', r_value**2)

        # top 10 reactions
        val, idx = torch.topk(results, 10, dim=0)
        for i in range(3):
            mean = torch.index_select(results[:,0], 0, idx[:,i]).mean().item()
            std = torch.index_select(results[:,0], 0, idx[:,i]).std().item()
            all_results.append(file, 10, 'mean', mean)
            all_results.append(file, 10, 'std', std)
        # top 50 reactions
        if results.size()[0]>50:
            val, idx = torch.topk(results, 50, dim=0)
            for i in range(3):
                mean = torch.index_select(results[:,0], 0, idx[:,i]).mean().item()
                std = torch.index_select(results[:,0], 0, idx[:,i]).std().item()
                all_results.append(file, 50, 'mean', mean)
                all_results.append(file, 50, 'std', std)
        if results.size()[0]>100:
            # top 100 reactions
            val, idx = torch.topk(results, 100, dim=0)
            for i in range(3):
                mean = torch.index_select(results[:,0], 0, idx[:,i]).mean().item()
                std = torch.index_select(results[:,0], 0, idx[:,i]).std().item()
                all_results.append(file, 100, 'mean', mean)
                all_results.append(file, 100, 'std', std)

    pd.DataFrame(all_results, columns=['file_name','top-k','type','value'])\
        .to_csv(os.path.join(args.output_dir, 'results.csv'))

if __name__ == "__main__":
    main()
