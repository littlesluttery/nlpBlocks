import math
import os.path
import random
from dataclasses import dataclass
from typing import Any,List,Tuple


import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer,BatchEncoding
from itertools import chain
from .arguments import ModelDataArguments


class TrainDatasetForEmbedding(Dataset):
    def __init__(self,args:ModelDataArguments) -> None:
        super().__init__()
        if os.path.isdir(args.data_dir):
            train_datasets = []
            for file in os.listdir(args.data_dir):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(args.data_dir,file),
                    split="train",
                    cache_dir=args.cache_dir_data,
                )
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset(
                "json",
                data_files=args.train_data,
                split="train"
            )
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, index) -> Any:
        query = self.dataset[index]["query"]
        pos = self.dataset[index]["pos"][0]
        neg = self.dataset[index]["neg"][0]
        res = {
            "query":query,
            "pos":pos,
            "neg":neg
        }
        return res
    

class SentenceEmbeddingCollator:
    def __init__(self) -> None:
        pass

    def __call__(self, features) -> Any:
        query = []
        pos = []
        neg = []

        for i in features:
            query.append(i["query"])
            pos.append(i["pos"])
            neg.append(i["neg"])
        res = {
            "query":query,
            "pos":pos,
            "neg":neg
        }
        return res
    
