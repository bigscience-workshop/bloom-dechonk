import logging
from itertools import chain
from typing import Iterable, Optional

import datasets
import torch
import torch.utils.data
import transformers.utils.logging
from transformers import TrainingArguments
from transformers.testing_utils import CaptureLogger
from transformers import DataCollatorForLanguageModeling

import src.overrides
from src.arguments import DataTrainingArguments

logger = logging.getLogger(__name__)

TEXT_COLUMN_NAME = 'text'


def get_tokenized_lm_datasets(tokenizer, cache_dir, data_args: DataTrainingArguments, training_args: TrainingArguments):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    assert data_args.dataset_name is not None

    # Downloading and loading a dataset from the hub.
    if src.overrides.load_raw_shuffled_datasets is None:
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=cache_dir
        )
    else:
        raw_datasets = src.overrides.load_raw_shuffled_datasets(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=cache_dir
        )
    assert "train" in raw_datasets.keys() and "validation" in raw_datasets.keys()

    # Preprocessing the datasets.
    # First we tokenize all the texts.

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer([sample + '\n' for sample in examples[TEXT_COLUMN_NAME]])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = {
            key: dataset.map(
                tokenize_function,
                batched=True,
            )
            for key, dataset in raw_datasets.items()
        }

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 2048 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 2048
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = {
            key: WrappedIterableDataset(dataset.map(group_texts, batched=True))
            for key, dataset in tokenized_datasets.items()
        }

    return lm_datasets, DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=block_size)


class WrappedIterableDataset(torch.utils.data.IterableDataset):
    """Wraps huggingface IterableDataset as pytorch IterableDataset, implement default methods for DataLoader"""

    def __init__(self, iterable: Iterable, cycle: bool = True, verbose: bool = True, num_examples: Optional[int] = None):
        self.iterable = iterable
        self.cycle, self.verbose = cycle, verbose
        self.num_examples = num_examples
        self._persistent_iter = None

    def __iter__(self):
        started = False
        logger.info("Pre-fetching training samples...")
        while True:
            for sample in self.iterable:
                if not started:
                    logger.info("Began iterating minibatches!")
                    started = True
                yield sample

            if not self.cycle:
                break

    def __len__(self):
        return self.num_examples

    def take_next_subset(self, k: int):
        """Create a WrappedIterableDataset that iterates over the next k samples from the current dataset (stateful)"""
        if self._persistent_iter is None:
            self._persistent_iter = iter(self)

        def _subset_iterator():
            for i in range(k):
                yield next(self._persistent_iter)

        return WrappedIterableDataset(_subset_iterator(), cycle=False, num_examples=k, verbose=self.verbose)
