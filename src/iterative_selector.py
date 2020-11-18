import os
import argparse
import scipy
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from EFGs import standize
import torch.nn.functional as F
from functools import partial
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from torch.nn.utils.rnn import pad_sequence
from data_utils import MolTokenizer
from early_stop_trainer import EarlyStopTrainer
from transformers import T5Config, TrainingArguments, T5PreTrainedModel
from transformers.modeling_t5 import T5Stack

from torch.utils.data import Dataset
from typing import Callable, Iterable, List

from run_trainer_qm9 import T5ForRegression
from run_trainer_yield import data_collator


class YieldDatasetFromList(Dataset):
    def __init__(
        self,
        tokenizer,
        list_data,
        max_source_length=500,
    ):
        super().__init__()
        # FIXME: the rstrip logic strips all the chars, it seems.
        tok_name = tokenizer.__class__.__name__.lower().rstrip("tokenizer")

        self.source, self.target = [], []
        for text in tqdm(list_data, desc=f"Tokenizing"):
            tokenized = tokenizer(
                [text[0]],
                max_length=max_source_length,
                padding="do_not_pad",
                truncation=True,
                return_tensors='pt',
            )
            self.source.append(tokenized)
            self.target.append(float(text[1]))
        
        self.bos_token_id = tokenizer.bos_token_id

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": torch.LongTensor([self.bos_token_id]),
                "labels": torch.tensor([target_ids])}

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


def add_args(parser): 
    parser.add_argument(
        "--data_file",
        required=True,
        help="The training data file.",
    )
    parser.add_argument(
        "--test_file",
        required=True,
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
    # this one is needed for torchtext random call (shuffled iterator)             
    # in multi gpu it ensures datasets are read in the same order                  
    # some cudnn methods can be random even after fixing the seed                  
    # unless you tell it to be deterministic                                       
    torch.backends.cudnn.deterministic = True  
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    portion_dict = {0:'Ideal:', 1:'random:', 2:'out-of-sample:'}
    lm_heads_layer = {
        'sigmoid': nn.Sequential(nn.Linear(256, 1), nn.Sigmoid()),
        'sigmoid2': nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256,1), nn.Sigmoid()),
        'linear2': nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256,1)),
        'linear1': nn.Sequential(nn.Linear(256, 1)),
    }

    if args.vocab:
        tokenizer = MolTokenizer(vocab_file=args.vocab)
    else:
        tokenizer = MolTokenizer(source_files=[os.path.join(os.path.dirname(
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
                vocab_size=len(tokenizer.vocab),                                        
                pad_token_id=tokenizer.pad_token_id,                                    
                decoder_input_ids=tokenizer.bos_token_id,                               
                eos_token_id=tokenizer.eos_token_id,                                    
                bos_token_id=tokenizer.bos_token_id,                                                           
                output_past=True,                                                       
                num_layers=4,
                num_heads=8,
                d_model=256,
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
    
        data_collator_pad1 = partial(data_collator,
                                     pad_token_id=tokenizer.pad_token_id,
                                     percentage=('sigmoid' in args.mode),
                                    )
    
        model = model.to(device)
    
        output_dir = os.path.join(args.output_dir, args.mode, os.path.basename(args.data_file).split('.')[0])
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
        trainer.save_model(output_dir+str(turn))

        model = model.eval()
        results = torch.zeros(len(test_list), 3)
        results[:, 0]=torch.Tensor(testset.target)

        results[:, 1]=results[:,0][torch.randperm(len(test_list))]
        test_loader = DataLoader(testset, batch_size=args.batch_size,               
                     collate_fn=data_collator_pad1)
        x = []
        for batch in tqdm(test_loader, desc="prediction"):

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(**batch)
            for pred in outputs[1]:
                x.append(pred.item())
        results[:,2] = torch.tensor(x)*100
        torch.save(results, os.path.join(output_dir+str(turn),
            os.path.basename(args.test_file).split('.')[0]+'_results.pt'))

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
