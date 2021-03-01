import linecache
import os
import subprocess
from collections import Counter
from itertools import groupby
from typing import List, Optional

import torch
import torchtext
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

#from .selfies import split_selfies

TASK_PREFIX = ['Yield:', 'Product:', 'Fill-Mask:', 'Retrosynthesis:', '>', 'Classification:']

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, prefix=''):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        
        self.prefix = prefix
        self._file_path = file_path
        self._len = int(subprocess.check_output("wc -l " + file_path, shell=True).split()[0])
        self.tokenizer = tokenizer
        self.max_length = block_size
        
    def __getitem__(self, idx):
        line = linecache.getline(self._file_path, idx + 1).strip()
        sample = self.tokenizer(
                        self.prefix+line,
                        max_length=self.max_length,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        return sample['input_ids'].squeeze(0)
      
    def __len__(self):
        return self._len


class MolTranslationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path="train",
        max_source_length=500,
        max_target_length=500,
    ):
        super().__init__()

        self._source_path = os.path.join(data_dir, type_path + ".source")
        self._target_path = os.path.join(data_dir, type_path + ".target")
        self._len_source = int(subprocess.check_output("wc -l " + self._source_path, shell=True).split()[0])
        self._len_target = int(subprocess.check_output("wc -l " + self._target_path, shell=True).split()[0])
        assert self._len_source == self._len_target, "Source file and target file don't match!"
        self.tokenizer = tokenizer
        self.max_source_len = max_source_length
        self.max_target_len = max_target_length

    def __len__(self):
        return self._len_source

    def __getitem__(self, idx):
        source_line = linecache.getline(self._source_path, idx + 1).strip()
        source_sample = self.tokenizer(
                        source_line,
                        max_length=self.max_source_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        target_line = linecache.getline(self._target_path, idx + 1).strip()
        target_sample = self.tokenizer(
                        target_line,
                        max_length=self.max_target_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        source_ids = source_sample["input_ids"].squeeze()
        target_ids = target_sample["input_ids"].squeeze()
        src_mask = source_sample["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": target_ids}

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


class TaskPrefixDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        prefix='',
        type_path="train",
        max_source_length=300,
        max_target_length=100,
        separate_vocab=False,
    ):
        super().__init__()

        self.prefix = prefix
        self._source_path = os.path.join(data_dir, type_path + ".source")
        self._target_path = os.path.join(data_dir, type_path + ".target")
        self._len_source = int(subprocess.check_output("wc -l " + self._source_path, shell=True).split()[0])
        self._len_target = int(subprocess.check_output("wc -l " + self._target_path, shell=True).split()[0])
        assert self._len_source == self._len_target, "Source file and target file don't match!"
        self.tokenizer = tokenizer
        self.max_source_len = max_source_length
        self.max_target_len = max_target_length
        self.sep_vocab = separate_vocab

    def __len__(self):
        return self._len_source

    def __getitem__(self, idx):
        source_line = linecache.getline(self._source_path, idx + 1).strip()
        source_sample = self.tokenizer(
                        self.prefix+source_line,
                        max_length=self.max_source_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        target_line = linecache.getline(self._target_path, idx + 1).strip()
        if self.sep_vocab:
            target_line = target_line[:self.max_target_len]
            try:
                target_line = int(target_line)
                target_ids = torch.LongTensor([target_line])
            except TypeError:
                print("The target should be integer representing a class, \
                        not {}".format(target_line))
                raise AssertionError
        else:
            target_sample = self.tokenizer(
                            target_line,
                            max_length=self.max_target_len,
                            padding="do_not_pad",
                            truncation=True,
                            return_tensors='pt',
                        )
            target_ids = target_sample["input_ids"].squeeze(0)
        source_ids = source_sample["input_ids"].squeeze(0)
        src_mask = source_sample["attention_mask"].squeeze(0)
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": target_ids}

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


class YieldDatasetFromList(Dataset):
    def __init__(
        self,
        tokenizer,
        list_data,
        prefix='',
        max_source_length=300,
        max_target_length=5,
    ):
        super().__init__()

        self.source, self.target = [], []
        self.num_target = []
        for text in tqdm(list_data, desc=f"Tokenizing..."):
            rxn, _yield = text.strip().split()
            tokenized = tokenizer(
                [prefix+rxn],
                max_length=max_source_length,
                padding="do_not_pad",
                truncation=True,
                return_tensors='pt',
            )
            tokenized_target = tokenizer(
                ['{:.2f}'.format(float(_yield))],
                max_length=max_target_length,
                padding="do_not_pad",
                truncation=True,
                return_tensors='pt',
            )
            self.source.append(tokenized)
            self.num_target.append(float(_yield))
            self.target.append(tokenized_target)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze(0)
        target_ids = self.target[index]["input_ids"].squeeze(0)
        src_mask = self.source[index]["attention_mask"].squeeze(0)
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "labels": target_ids}

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


class MolTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a Molecular tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`, `optional`, defaults to ''):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<blank>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """

    def __init__(
        self,
        vocab_file='',
        source_files='',
        unk_token='<unk>',
        bos_token='<s>',
        pad_token="<blank>",
        eos_token='</s>',
        vocab_size=None,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs)

        self.create_vocab(vocab_file=vocab_file, source_files=source_files, vocab_size=vocab_size)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def merge_vocabs(self, vocabs, vocab_size=None):
        """
        Merge individual vocabularies (assumed to be generated from disjoint
        documents) into a larger vocabulary.
        Args:
            vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
            vocab_size: `int` the final vocabulary size. `None` for no limit.
        Return:
            `torchtext.vocab.Vocab`
        """
        merged = sum([vocab.freqs for vocab in vocabs], Counter())
        return torchtext.vocab.Vocab(merged,
                                     specials=list(self.special_tokens_map.values()),
                                     max_size=vocab_size)

    def create_vocab(self, vocab_file=None, source_files=None, vocab_size=None):
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
            source_files (:obj:`string`, `optional`, defaults to ''):
                File containing source data files, vocabulary would be built based on the source file(s).
        """
        if (not vocab_file) and (not source_files):
            self.vocab = []
        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                self.vocab = torch.load(vocab_file)

        if source_files:
            if isinstance(source_files, str):
                if not os.path.isfile(source_files):
                    raise ValueError(
                        "Can't find a source file at path '{}'.".format(source_files)
                    )
                else:
                    source_files = [source_files]
            counter = {}
            vocabs = {}
            for i, source_file in enumerate(source_files):
                counter[i] = Counter()
                with open(source_file) as rf:
                    for line in tqdm(rf, desc='Generating {}'.format(source_file)):
                        try:
                            items = self._tokenize(line.strip())
                            counter[i].update(items)
                        except AssertionError:
                            print(line.strip())
                specials = list(self.special_tokens_map.values())
                vocabs[i] = torchtext.vocab.Vocab(counter[i], specials=specials)
            self.vocab = self.merge_vocabs([vocabs[i] for i in range(len(source_files))], vocab_size=vocab_size)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        """
        Tokenize a SMILES molecule or reaction
        """
        import re  
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(text)]
        assert text == ''.join(tokens), 'Error when parsing {}'.format(text)
        return tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = "".join(tokens).strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        torch.save(self.vocab, vocab_path)

class SelfiesTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a SELFIES tokenizer. Based on SELFIES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`, `optional`, defaults to ''):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<blank>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """

    def __init__(
        self,
        vocab_file='',
        source_files='',
        unk_token='<unk>',
        bos_token='<s>',
        pad_token="<blank>",
        eos_token='</s>',
        vocab_size=None,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs)

        self.create_vocab(vocab_file=vocab_file, source_files=source_files, vocab_size=vocab_size)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def merge_vocabs(self, vocabs, vocab_size=None):
        """
        Merge individual vocabularies (assumed to be generated from disjoint
        documents) into a larger vocabulary.
        Args:
            vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
            vocab_size: `int` the final vocabulary size. `None` for no limit.
        Return:
            `torchtext.vocab.Vocab`
        """
        merged = sum([vocab.freqs for vocab in vocabs], Counter())
        return torchtext.vocab.Vocab(merged,
                                     specials=list(self.special_tokens_map.values()),
                                     max_size=vocab_size)

    def create_vocab(self, vocab_file=None, source_files=None, vocab_size=None):
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
            source_files (:obj:`string`, `optional`, defaults to ''):
                File containing source data files, vocabulary would be built based on the source file(s).
        """
        if (not vocab_file) and (not source_files):
            self.vocab = []
        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                self.vocab = torch.load(vocab_file)

        if source_files:
            if isinstance(source_files, str):
                if not os.path.isfile(source_files):
                    raise ValueError(
                        "Can't find a source file at path '{}'.".format(source_files)
                    )
                else:
                    source_files = [source_files]
            counter = {}
            vocabs = {}
            for i, source_file in enumerate(source_files):
                counter[i] = Counter()
                with open(source_file) as rf:
                    for line in tqdm(rf, desc='Generating {}'.format(source_file)):
                        try:
                            items = self._tokenize(line.strip())
                            counter[i].update(items)
                        except AssertionError:
                            print(line.strip())
                specials = list(self.special_tokens_map.values())
                vocabs[i] = torchtext.vocab.Vocab(counter[i], specials=specials)
            self.vocab = self.merge_vocabs([vocabs[i] for i in range(len(source_files))], vocab_size=vocab_size)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        """
        Tokenize a SMILES molecule or reaction
        """


        return list(split_selfies(text))

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """

        return ''.join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        torch.save(self.vocab, vocab_path)

class SimpleTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a basic tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`, `optional`, defaults to ''):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<blank>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """

    def __init__(
        self,
        vocab_file='',
        source_files='',
        unk_token='<unk>',
        bos_token='<s>',
        pad_token="<blank>",
        eos_token='</s>',
        vocab_size=None,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs)

        self.create_vocab(vocab_file=vocab_file, source_files=source_files, vocab_size=vocab_size)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def merge_vocabs(self, vocabs, vocab_size=None):
        """
        Merge individual vocabularies (assumed to be generated from disjoint
        documents) into a larger vocabulary.
        Args:
            vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
            vocab_size: `int` the final vocabulary size. `None` for no limit.
        Return:
            `torchtext.vocab.Vocab`
        """
        merged = sum([vocab.freqs for vocab in vocabs], Counter())
        return torchtext.vocab.Vocab(merged,
                                     specials=list(self.special_tokens_map.values()),
                                     max_size=vocab_size)

    def create_vocab(self, vocab_file=None, source_files=None, vocab_size=None):
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
            source_files (:obj:`string`, `optional`, defaults to ''):
                File containing source data files, vocabulary would be built based on the source file(s).
        """
        if (not vocab_file) and (not source_files):
            self.vocab = []
        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                self.vocab = torch.load(vocab_file)

        if source_files:
            if isinstance(source_files, str):
                if not os.path.isfile(source_files):
                    raise ValueError(
                        "Can't find a source file at path '{}'.".format(source_files)
                    )
                else:
                    source_files = [source_files]
            counter = {}
            vocabs = {}
            for i, source_file in enumerate(source_files):
                counter[i] = Counter()
                with open(source_file) as rf:
                    for line in tqdm(rf, desc='Generating {}'.format(source_file)):
                        try:
                            items = self._tokenize(line.strip())
                            counter[i].update(items)
                        except AssertionError:
                            print(line.strip())
                specials = list(self.special_tokens_map.values())
                vocabs[i] = torchtext.vocab.Vocab(counter[i], specials=specials)
            self.vocab = self.merge_vocabs([vocabs[i] for i in range(len(source_files))], vocab_size=vocab_size)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        """
        Tokenize a SMILES molecule or reaction
        """
        return list(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = "".join(tokens).strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        torch.save(self.vocab, vocab_path)

class T5MolTokenizer(MolTokenizer):
    def __init__(self, vocab_file, task_prefixs=TASK_PREFIX, max_size=2400, **kwargs):
        super().__init__(
                unk_token='<unk>',
                bos_token='<s>',
                pad_token='<pad>',
                eos_token='</s>',
                mask_token='<mask>',
                **kwargs)
        raw_vocab = torch.load(vocab_file)
        max_size = min(max_size, len(raw_vocab)+100-len(task_prefixs))
        self.vocab = torchtext.vocab.Vocab(raw_vocab.freqs, specials=['<s>', '</s>', '<unk>', '<pad>', '<mask>'],
                           max_size=max_size-len(task_prefixs))
        extra_to_add = max_size - len(self.vocab)
        cur_added_len = len(task_prefixs)
        for i in range(cur_added_len, extra_to_add):
            task_prefixs.append('<extra_id_{}>'.format(str(i)))
        self.add_tokens(task_prefixs, special_tokens=True)
        self.unique_no_split_tokens = sorted(
            set(self.unique_no_split_tokens).union(set(self.all_special_tokens))
        )
    
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1 = None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

class T5SimpleTokenizer(SimpleTokenizer):
    def __init__(self, vocab_file, task_prefixs=TASK_PREFIX, max_size=2400, **kwargs):
        super().__init__(
                unk_token='<unk>',
                bos_token='<s>',
                pad_token='<pad>',
                eos_token='</s>',
                mask_token='<mask>',
                **kwargs)
        raw_vocab = torch.load(vocab_file)
        max_size = min(max_size, len(raw_vocab)+100-len(task_prefixs))
        self.vocab = torchtext.vocab.Vocab(raw_vocab.freqs, specials=['<s>', '</s>', '<unk>', '<pad>', '<mask>'],
                           max_size=max_size-len(task_prefixs))
        extra_to_add = max_size - len(self.vocab)
        cur_added_len = len(task_prefixs)
        for i in range(cur_added_len, extra_to_add):
            task_prefixs.append('<extra_id_{}>'.format(str(i)))
        self.add_tokens(task_prefixs, special_tokens=True)
        self.unique_no_split_tokens = sorted(
            set(self.unique_no_split_tokens).union(set(self.all_special_tokens))
        )
    
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1 = None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

class T5SelfiesTokenizer(SelfiesTokenizer):
    def __init__(self, vocab_file, task_prefixs=TASK_PREFIX, max_size=2400, **kwargs):
        super().__init__(
                unk_token='<unk>',
                bos_token='<s>',
                pad_token='<pad>',
                eos_token='</s>',
                mask_token='<mask>',
                **kwargs)
        raw_vocab = torch.load(vocab_file)
        max_size = min(max_size, len(raw_vocab)+100-len(task_prefixs))
        self.vocab = torchtext.vocab.Vocab(raw_vocab.freqs, specials=['<s>', '</s>', '<unk>', '<pad>', '<mask>'],
                           max_size=max_size-len(task_prefixs))
        extra_to_add = max_size - len(self.vocab)
        cur_added_len = len(task_prefixs)
        for i in range(cur_added_len, extra_to_add):
            task_prefixs.append('<extra_id_{}>'.format(str(i)))
        self.add_tokens(task_prefixs, special_tokens=True)
        self.unique_no_split_tokens = sorted(
            set(self.unique_no_split_tokens).union(set(self.all_special_tokens))
        )
    
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1 = None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

def data_collator(batch, pad_token_id):
    whole_batch = {}
    ex = batch[0]
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
    # y_ids = y[:, :-1].contiguous()
    # lm_labels = y[:, 1:].clone()
    # lm_labels[y[:, 1:] == padding_value] = -100
    return {'input_ids': source_ids, 'attention_mask': source_mask,
            # 'decoder_input_ids': y_ids, 
            'labels': y}
