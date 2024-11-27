"""t5chem - A Unified Deep Learning Model for Multi-task Reaction Predictions"""
from t5chem.__version__ import __version__
from t5chem.data_utils import LineByLineTextDataset, TaskPrefixDataset, data_collator
from t5chem.model import T5ForProperty
from .mol_tokenizers import AtomTokenizer, SelfiesTokenizer, SimpleTokenizer
# from t5chem.trainer import EarlyStopTrainer

__author__ = 'Jocelyn Lu <jl8570@nyu.edu>'
__all__: list = [
    "TaskPrefixDataset",
    "data_collator",
    "LineByLineTextDataset",
    "T5ForProperty",
    "AtomTokenizer",
    "SelfiesTokenizer",
    "SimpleTokenizer",
    # "EarlyStopTrainer",
]
