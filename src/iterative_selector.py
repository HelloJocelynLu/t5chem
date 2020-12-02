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
        "--test_file",
        required=True,
        help="The test data file.",
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

    if args.vocab:
        tokenizer = T5MolTokenizer(vocab_file=args.vocab)
    else:
        tokenizer = T5MolTokenizer(source_files=[os.path.join(os.path.dirname(
            args.data_files[0]), 'train.txt')])
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))

    ori_train = pd.read_csv(args.data_file, sep='\s+', header=None)
    train_list = ori_train.values.tolist()
    ori_test = pd.read_csv(args.test_file, sep='\s+', header=None)
    test_list = ori_test.values.tolist()

    print(args)
    for turn in range(5):
        dataset = YieldDatasetFromList(tokenizer, train_list)
        testset = YieldDatasetFromList(tokenizer, test_list)
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
    
        output_dir = args.output_dir
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

        model = model.eval()
        results = torch.zeros(len(test_list), 3)
        results[:, 0]=torch.Tensor(testset.num_target)

        results[:, 1]=results[:,0][torch.randperm(len(test_list))]
        test_loader = DataLoader(testset, batch_size=args.batch_size,               
                     collate_fn=data_collator_pad1)
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

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                results[:,0], results[:,2])
        print('MAE:',mean_absolute_error(results[:,0], results[:,2]), 'R^2:', r_value**2)
        print('training size:{} test size:{}'.format(len(train_list), len(test_list)))
        # top 10 reactions
        print("------------- Select 10 reactions --------------Round "+str(turn))
        val, idx = torch.topk(results, 10, dim=0)
        for i in range(3):
            mean = torch.index_select(results[:,0], 0, idx[:,i]).mean().item()
            std = torch.index_select(results[:,0], 0, idx[:,i]).std().item()
            print('train', portion_dict[i], mean, '±', std, '%')
        selected_idx = set(idx[:,2].tolist())
        train_list += [x for i,x in enumerate(test_list) if i in selected_idx]
        test_list = [x for i,x in enumerate(test_list) if not i in selected_idx]
        print('*'*30)

if __name__ == "__main__":
    main()
