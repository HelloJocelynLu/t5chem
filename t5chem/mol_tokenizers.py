import importlib
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Union, Tuple

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

is_selfies_available: bool = False
if importlib.util.find_spec("selfies"):
    from selfies import split_selfies
    is_selfies_available = True
pattern: str = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex: re.Pattern = re.compile(pattern)
TASK_PREFIX: List[str] = ['Yield:', 'Product:', 'Fill-Mask:', 'Classification:', 'Reagents:', 'Reactants:']


class Vocab:
    """Simple vocabulary class to replace torchtext.vocab.Vocab"""
    def __init__(
        self,
        tokens: List[str]
    ):
        """Initialize vocabulary from a list of tokens
        
        Args:
            tokens: List of tokens in desired order
        """
        self.itos = tokens  # index to string
        self.stoi = {s: i for i, s in enumerate(tokens)}  # string to index

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: str) -> None:
        """Save vocabulary to a text file, one token per line"""
        with open(path, 'w', encoding='utf-8') as f:
            for token in self.itos:
                f.write(f"{token}\n")

    @classmethod
    def load(cls, path: str) -> 'Vocab':
        """Load vocabulary from a text file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f]
        return cls(tokens)


class MolTokenizer(ABC, PreTrainedTokenizer):
    r"""
    An abstract class for all tokenizers. Other tokenizer should
    inherit this class
    """
    def __init__(
        self,
        vocab_file: Optional[str]=None,
        source_files: Optional[Union[str, List[str]]]=None,
        unk_token: str='<unk>',
        bos_token: str='<s>',
        pad_token: str="<pad>",
        eos_token: str='</s>',
        mask_token: str='<mask>',
        max_size: int=1000,
        task_prefixs: List[str]=[],
        **kwargs
    ) -> None:

        task_prefixs = TASK_PREFIX+task_prefixs
        self.create_vocab(vocab_file=vocab_file)
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs)
        if self.vocab:
            extra_to_add: int = max_size - len(self.vocab)
            cur_added_len: int = len(task_prefixs) + 9 # placeholder for smiles tokens
            for i in range(cur_added_len, extra_to_add):
                task_prefixs.append('<extra_task_{}>'.format(str(i)))
            self.add_tokens(['<extra_token_'+str(i)+'>' for i in range(9)]+task_prefixs+['>'], special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def create_vocab(self, vocab_file: Optional[str]=None) -> None:
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
        """
        self.vocab: Vocab = Vocab.load(vocab_file)

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    @abstractmethod
    def _tokenize(self, text: str, **kwargs) -> List[str]: 
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
        
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the vocabulary to a directory.

        Args:
            save_directory (str): 
                The directory where the vocabulary will be saved.
            filename_prefix (Optional[str], defaults to None): 
                An optional prefix to add to the filename.

        Returns:
            Tuple[str]: Tuple containing the path to the saved vocabulary file.

        Raises:
            ValueError: If save_directory is not a directory.
        """
        if not os.path.isdir(save_directory):
            raise ValueError(f"Vocabulary path ({save_directory}) should be a directory")

        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt"
        )
        
        self.vocab.save(vocab_file)
        return (vocab_file,)


class SimpleTokenizer(MolTokenizer):
    r"""
    Constructs a simple, character-level tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 100):
            The final vocabulary size. `None` for no limit.
        **kwargs:
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=100, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        return list(text)


class AtomTokenizer(MolTokenizer):
    r"""
    Constructs an atom-level tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 1000):
            The final vocabulary size. `None` for no limit.
        **kwargs:
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=1000, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        tokens: List[str] = [token for token in regex.findall(text)]
        assert text == ''.join(tokens), 'Error when parsing {}'.format(text)
        return tokens

class SelfiesTokenizer(MolTokenizer):
    r"""
    Constructs an SELFIES tokenizer. Based on SELFIES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 1000):
            The final vocabulary size. `None` for no limit.
        **kwargs:
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=1000, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)
        assert is_selfies_available, "You need to install selfies package to use SelfiesTokenizer"

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        """
        Tokenize a SELFIES molecule or reaction
        """
        return list(split_selfies(text))

