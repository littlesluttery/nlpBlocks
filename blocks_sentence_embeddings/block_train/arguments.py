from dataclasses import dataclass,field
from typing import List,Tuple
from transformers import TrainingArguments


@dataclass
class SentenceTrainArguements(TrainingArguments):
    query_max_len:int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    temperature:float = 0.05
    embedding_model_name: str=field(default="qwen2")


@dataclass 
class ModelDataArguments:
    model_name_or_path:str 
    data_dir:str
    cache_dir_data:str
