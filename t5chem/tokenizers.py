from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer
from typing import Dict, Optional, List
from collections import Counter
from tqdm import tqdm
import re
import os
import importlib
import torch
from torchtext.vocab import Vocab


is_selfies_available: bool = False
if importlib.util.find_spec("selfies"):
    from selfies import split_selfies
    is_selfies_available = True
pattern: str = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex: re.Pattern = re.compile(pattern)

class MolTokenizer(ABC, PreTrainedTokenizer):
    r"""
    An abstract class for all tokenizers. Other tokenizer should
    inherit this class
    """

    def __init__(
        self,
        vocab_file: str='',
        source_files: str='',
        unk_token: str='<unk>',
        bos_token: str='<s>',
        pad_token: str="<blank>",
        eos_token: str='</s>',
        vocab_size: Optional[int]=None,
        **kwargs
    ) -> None:
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs)

        self.create_vocab(vocab_file=vocab_file, source_files=source_files, vocab_size=vocab_size)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def merge_vocabs(
        self, 
        vocabs: List[Vocab], 
        vocab_size: Optional[int]=None,
    ) -> Vocab:
        """
        Merge individual vocabularies (assumed to be generated from disjoint
        documents) into a larger vocabulary.
        Args:
            vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
            vocab_size: `int` the final vocabulary size. `None` for no limit.
        Return:
            `torchtext.vocab.Vocab`
        """
        merged: Counter = sum([vocab.freqs for vocab in vocabs], Counter())
        return Vocab(merged,
                    specials=list(self.special_tokens_map.values()),
                    max_size=vocab_size)

    def create_vocab(
        self, 
        vocab_file: str=None, 
        source_files: str=None, 
        vocab_size: Optional[int]=None,
        ) -> None:
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
            source_files (:obj:`string`, `optional`, defaults to ''):
                File containing source data files, vocabulary would be built based on the source file(s).
            vocab_size: (:obj:`int`, `optional`, defaults to `None`):
                The final vocabulary size. `None` for no limit.
        """
        if (not vocab_file) and (not source_files):
            self.vocab: list = []
        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                self.vocab: Vocab = torch.load(vocab_file)

        if source_files:
            if isinstance(source_files, str):
                if not os.path.isfile(source_files):
                    raise ValueError(
                        "Can't find a source file at path '{}'.".format(source_files)
                    )
                else:
                    source_files: List[str] = [source_files]
            counter: dict = {}
            vocabs: dict = {}
            for i, source_file in enumerate(source_files):
                counter[i] = Counter()
                with open(source_file) as rf:
                    for line in tqdm(rf, desc='Generating {}'.format(source_file)):
                        try:
                            items: List[str] = self._tokenize(line.strip())
                            counter[i].update(items)
                        except AssertionError:
                            print(line.strip())
                specials: List[str] = list(self.special_tokens_map.values())
                vocabs[i] = Vocab(counter[i], specials=specials)
            self.vocab: Vocab = self.merge_vocabs([vocabs[i] for i in range(len(source_files))], vocab_size=vocab_size)

    def get_vocab(self) -> Vocab:
        return self.vocab
    
    @abstractmethod
    def _tokenize(self, text: str):
        """
        Tokenize a molecule or reaction
        """
        pass

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        out_string: str = "".join(tokens).strip()
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
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def save_vocabulary(self, vocab_path: str) -> None:
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        torch.save(self.vocab, vocab_path)

class SimpleTokenizer(MolTokenizer):
    r"""
    Constructs a simple, character-level tokenizer. Based on SMILES.
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
        vocab_size: (:obj:`int`, `optional`, defaults to `None`):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _tokenize(self, text) -> List[str]:
        return list(text)

class AtomTokenizer(MolTokenizer):
    r"""
    Constructs an atom-level tokenizer. Based on SMILES.
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
        vocab_size: (:obj:`int`, `optional`, defaults to `None`):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _tokenize(self, text) -> List[str]:
        tokens: List[str] = [token for token in regex.findall(text)]
        assert text == ''.join(tokens), 'Error when parsing {}'.format(text)
        return tokens

class SelfiesTokenizer(MolTokenizer):
    r"""
    Constructs an SELFIES tokenizer. Based on SELFIES.
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
        vocab_size: (:obj:`int`, `optional`, defaults to `None`):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _tokenize(self, text) -> List[str]:
        """
        Tokenize a SMILES molecule or reaction
        """
        tokens: List[str] = [token for token in regex.findall(text)]
        assert text == ''.join(tokens), 'Error when parsing {}'.format(text)
        return tokens

    def _tokenize(self, text):
        """
        Tokenize a SMILES molecule or reaction
        """
        from selfies import split_selfies
        return list(split_selfies(text))

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
        self.add_tokens(task_prefixs+['>'], special_tokens=True)
        self.unique_no_split_tokens = sorted(
            set(self.unique_no_split_tokens).union(set(self.all_special_tokens))
        )
    
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

