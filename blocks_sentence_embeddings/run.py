import logging
import os
import debugpy
from transformers import HfArgumentParser

from block_train.arguments import SentenceTrainArguements,ModelDataArguments
from block_train.data import TrainDatasetForEmbedding,SentenceEmbeddingCollator
from block_train.model import EmbeddingModel,EmbeddingModelQwen2
from block_train.trainer import SentenceTrainer



logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelDataArguments,SentenceTrainArguements))
    model_data_args,training_args = parser.parse_args_into_dataclasses()
    model_data_args:ModelDataArguments
    training_args:SentenceTrainArguements

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model Data parameters %s", model_data_args)

    if training_args.embedding_model_name == "qwen2":
        model = EmbeddingModelQwen2(
            model_name_or_path=model_data_args.model_name_or_path
        )
    else:
        model = EmbeddingModel(model_name_or_path=model_data_args.model_name_or_path)
    
    dataset = TrainDatasetForEmbedding(args=model_data_args)
    trainer = SentenceTrainer(
        model = model,
        args = training_args,
        data_collator=SentenceEmbeddingCollator(),
        train_dataset=dataset, 

    )
    trainer.train()

if __name__ == "__main__":
    main()
