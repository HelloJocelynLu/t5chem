import linecache
import os
import subprocess
from typing import Dict, List, NamedTuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput

TOKENS = {"mask_token" : "<mask>",
            "unk_token" : "<unk>",
            "pad_token" : "<pad>",
            "bos_token" : "<pad>",
            "sos_token" : "<pad>",
            "eos_token" : "</s>"}
DEFAULT_VOCAB = os.path.join(os.getcwd(), "t5chem","vocab","tokenizer.json")

class TaskSettings(NamedTuple):
    prefix: str
    max_source_length: int
    max_target_length: int
    output_layer: str


T5ChemTasks: Dict[str, TaskSettings] = {
    'product': TaskSettings('Product:', 400, 200, 'seq2seq'),
    'reactants': TaskSettings('Reactants:', 200, 300, 'seq2seq'),
    'reagents': TaskSettings('Reagents:', 400, 200, 'seq2seq'),
    'classification': TaskSettings('Classification:', 500, 1, 'classification'),
    'regression': TaskSettings('Yield:', 500, 1, 'regression'),
    'pretrain': TaskSettings('Fill-Mask:', 400, 200, 'seq2seq'),
    'mixed': TaskSettings('', 400, 300, 'seq2seq'),
}


class LineByLineTextDataset(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        file_path: str, 
        block_size: int, 
        prefix: str = ''
    ) -> None:
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        
        self.prefix: str = prefix
        self._file_path: str = file_path
        self._len: int = int(subprocess.check_output("wc -l " + file_path, shell=True).split()[0])
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = block_size
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        line: str = linecache.getline(self._file_path, idx + 1).strip()
        sample = self.tokenizer(
                        self.prefix+line,
                        max_length=self.max_length,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        # Assert sample is BatchEncoding
        assert isinstance(sample, BatchEncoding) # Should be a batchEncoding.
        return sample['input_ids'].squeeze(0)
      
    def __len__(self) -> int:
        return self._len


class TaskPrefixDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_dir: str,
        prefix: str='',
        type_path: str="train",
        max_source_length: int=300,
        max_target_length: int=100,
        separate_vocab: bool=False,
    ) -> None:
        super().__init__()

        self.prefix: str = prefix
        self._source_path: str = os.path.join(data_dir, type_path + ".source")
        self._target_path: str = os.path.join(data_dir, type_path + ".target")
        self._len_source: int = int(subprocess.check_output("wc -l " + self._source_path, shell=True).split()[0])
        self._len_target: int = int(subprocess.check_output("wc -l " + self._target_path, shell=True).split()[0])
        assert self._len_source == self._len_target, "Source file and target file don't match!"
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_source_len: int = max_source_length
        self.max_target_len: int = max_target_length
        self.sep_vocab: bool = separate_vocab

    def __len__(self) -> int:
        return self._len_source

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source_line: str = linecache.getline(self._source_path, idx + 1).strip()
        source_sample: BatchEncoding = self.tokenizer(
                        self.prefix+source_line,
                        max_length=self.max_source_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        target_line: str = linecache.getline(self._target_path, idx + 1).strip()
        if self.sep_vocab:
            try:
                target_value: float = float(target_line)
                target_ids: torch.Tensor = torch.Tensor([target_value])
            except TypeError:
                print("The target should be a number, \
                        not {}".format(target_line))
                raise AssertionError
        else:
            target_sample: BatchEncoding = self.tokenizer(
                            target_line,
                            max_length=self.max_target_len,
                            padding="do_not_pad",
                            truncation=True,
                            return_tensors='pt',
                        )
            target_ids = target_sample["input_ids"].squeeze(0)
        source_ids: torch.Tensor = source_sample["input_ids"].squeeze(0)
        src_mask: torch.Tensor = source_sample["attention_mask"].squeeze(0)
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": target_ids}

    def sort_key(self, ex: BatchEncoding) -> int:
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


def data_collator(batch: List[BatchEncoding], pad_token_id: int) -> Dict[str, torch.Tensor]:
    whole_batch: Dict[str, torch.Tensor] = {}
    ex: BatchEncoding = batch[0]
    for key in ex.keys():
        if 'mask' in key:
            padding_value = 0
        else:
            padding_value = pad_token_id
        whole_batch[key] = pad_sequence([x[key] for x in batch],
                                        batch_first=True,
                                        padding_value=padding_value)
    source_ids, source_mask, y = \
        whole_batch["input_ids"], whole_batch["attention_mask"], whole_batch["decoder_input_ids"]
    return {'input_ids': source_ids, 'attention_mask': source_mask,
            'labels': y}


def CalMSELoss(model_output: PredictionOutput) -> Dict[str, float]:
    predictions: np.ndarray = model_output.predictions # type: ignore
    label_ids: np.ndarray = model_output.label_ids.squeeze() # type: ignore
    loss: float = ((predictions - label_ids)**2).mean().item()
    return {'mse_loss': loss}

def AccuracyMetrics(model_output: PredictionOutput) -> Dict[str, float]:
    label_ids: np.ndarray = model_output.label_ids # type: ignore
    predictions: np.ndarray = model_output.predictions.reshape(-1, label_ids.shape[1]) # type: ignore
    correct: int = np.all(predictions==label_ids, 1).sum()
    return {'accuracy': correct/len(predictions)}
