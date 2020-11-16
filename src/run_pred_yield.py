import argparse
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from .data import MolTokenizer, YieldDataset, data_collator_yield
from .models import T5ForRegression


def add_args(parser):
    parser.add_argument(
        "--data_files",
        required=True,
        nargs='+',
        help="The input data files.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The model path to be loaded.",
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
        "--checkpoint",
        default=0,
        type=int,
        help="Checkpoint to be loaded. Default to load latest one.",
    )
    parser.add_argument(
        "--load_best", action="store_true",
        help="Whether to load best trained weights. (best_cp*)",
    )
    parser.add_argument(
        "--max_length",
        default=300,
        type=int,
        help="The maximum length (for both source and target) after tokenization.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size for training and validation.",
    )


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = MolTokenizer(vocab_file=os.path.join(args.model_dir, 'vocab.pt'))

    losses, r2 = [], []
    for i,file in enumerate(args.data_files):
        testset = YieldDataset(tokenizer, file, type_path="test", sep_id=args.split_idx)
        model_dir = os.path.join(args.model_dir, args.mode, os.path.basename(file).split('.')[0])
        if args.checkpoint:
            model_dir = os.path.join(model_dir, 'checkpoint-'+str(args.checkpoint))
        if args.load_best:
            model_dirs = [f for f in Path(model_dir).glob('best_cp-*')]
            assert len(model_dirs) == 1
            model_dir = model_dirs[0]
        model = T5ForRegression.pretrained_config(model_dir)
        d_model = model.config.d_model
        lm_heads_layer = {
            'sigmoid': nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid()),
            'sigmoid2': nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model,1), nn.Sigmoid()),
            'linear2': nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model,1)),
            'linear1': nn.Sequential(nn.Linear(d_model, 1)),
        }
        model.set_lm_head(lm_heads_layer[args.mode])
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir,'pytorch_model.bin'),
                                         map_location=lambda storage, loc: storage))
        
        model = model.eval()
        data_collator_pad1 = partial(data_collator,
                                     pad_token_id=tokenizer.pad_token_id,
                                     percentage=('sigmoid' in args.mode),
                                    )

        test_loader = DataLoader(testset, batch_size=args.batch_size,               
                                 collate_fn=data_collator_pad1)
        loss = 0

        x,y = [],[]
        out_file = open(os.path.join(model_dir, "performance.log"), 'w')

        for batch in tqdm(test_loader, desc="prediction"):

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(**batch)
#            loss += outputs[0].item()*len(batch['labels'])
            for pred, label in zip(outputs[1], batch['labels']):
                x.append(pred.item())
                y.append(label.item())
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
#        losses.append(loss/len(test_loader.dataset))
        r2.append(r_value**2)
        losses.append(mean_absolute_error(y,x))
        print(file,mean_absolute_error(y,x), r_value**2, file=out_file)
        pd.DataFrame([x,y]).T.to_csv(os.path.join(model_dir, 'predictions.txt'),
                                     index=False, header=['predict','label'])
        out_file.close()
    print("MAE: {}±{}    r2: {}±{}".format(np.mean(losses), np.var(losses), np.mean(r2), np.var(r2)))

if __name__ == "__main__":
    main()
