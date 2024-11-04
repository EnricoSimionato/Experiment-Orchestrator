from __future__ import annotations

from abc import ABC, abstractmethod
import gc
from typing import Optional, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

import pytorch_lightning as pl

import transformers

import datasets
from datasets import load_dataset, concatenate_datasets

import re


class LanguageModelingDataset(ABC, Dataset):
    """
    Dataset for language modeling tasks.
    """

    def __init__(
            self,
            dataset_id: str,
            tokenizer: transformers.PreTrainedTokenizer
    ) -> None:
        """
        Initializes the SentimentAnalysisDataset.
        """

        self.dataset_id = dataset_id
        self.tokenizer = tokenizer

    @abstractmethod
    def __len__(
            self
    ) -> int:
        """
        Returns the length of the dataset.
        """

    @abstractmethod
    def __getitem__(
            self,
            idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int):
                Index of the sample to retrieve.
        """


class ConversationDataset(LanguageModelingDataset, ABC):
    """
    Dataset for conversation modeling tasks.
    """


# TODO MAKE A GENERAL CLASS FOR ALL DATASETS
class DataModule(pl.LightningDataModule):
    """
    DataModule.

    Args:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets. Defaults to (0.8, 0.1, 0.1).
        batch_size (int):
            Size of the batch. Defaults to 1.
        num_workers (int):
            Number of workers to use for loading the data. Defaults to 1.
        seed (int):
            Seed for the random number generator.

    Attributes:
        batch_size (int):
            Size of the batch.
        num_workers (int):
            Number of workers to use for loading the data.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets.
        seed (int):
            Seed for the random number generator.
        train (OpenAssistantGuanacoDataset):
            Training dataset.
        validation (OpenAssistantGuanacoDataset):
            Validation dataset.
        test (OpenAssistantGuanacoDataset):
            Test dataset.
    """

    def __init__(
            self,
            tokenizer,
            max_len: int,
            split: tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 1,
            num_workers: int = 1,
            seed: int = 42,
    ) -> None:
        super().__init__()
        if len(split) != 3:
            raise ValueError(
                "The split must have three elements (train, validation, test)."
            )
        if sum(split) != 1:
            raise ValueError(
                "The sum of the split elements must be equal to 1."
            )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train = None
        self.validation = None
        self.test = None

    def prepare_data(
            self,
            **kwargs
    ):
        """
        Downloads the data.
        Runs on a single GPU.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        load_dataset(
            "timdettmers/openassistant-guanaco",
        )

    def setup(
            self,
            stage: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Preprocesses data.
        Can run on multiple GPUs.


        Args:
            stage (Optional[str]):
                Stage of the experiment.
            kwargs:
                Additional keyword arguments.
        """

        raw_dataset = load_dataset(
            "timdettmers/openassistant-guanaco",
        )

        concatenated_raw_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]])

        first_split_raw_dataset = concatenated_raw_dataset.train_test_split(
            test_size=self.split[2]
        )
        second_split_raw_dataset = first_split_raw_dataset["train"].train_test_split(
            test_size=self.split[1] / ((1 - self.split[2]) if self.split[2] != 1 else 1)
        )

        self.train = OpenAssistantGuanacoDataset(
            second_split_raw_dataset["train"],
            self.tokenizer,
            self.max_len
        )
        self.validation = OpenAssistantGuanacoDataset(
            second_split_raw_dataset["test"],
            self.tokenizer,
            self.max_len
        )
        self.test = OpenAssistantGuanacoDataset(
            first_split_raw_dataset["test"],
            self.tokenizer,
            self.max_len
        )

    def train_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the training DataLoader.

        Returns:
            DataLoader:
                Training DataLoader.
        """

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader:
                Validation DataLoader.
        """

        return DataLoader(
            self.validation,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers
        )

    def test_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the test DataLoader.

        Returns:
            DataLoader:
                Test DataLoader.
        """

        return DataLoader(
            self.test,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers
        )


class OpenAssistantGuanacoDataset(ConversationDataset):
    """
    Dataset class to use OpenAssistant-Guanaco dataset with Pytorch and Pytorch Lightning.

    Args:
        raw_dataset (datasets.Dataset):
            Raw dataset.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_length (int):
            Maximum length of the input sequences.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_length (int):
            Maximum length of the input sequences.
        dataset (list):
            Tokenized dataset.
        sep_regex (re.Pattern):
            Regular expression to separate the different turns of the dialogues.
    """

    def __init__(
            self,
            raw_dataset: datasets.Dataset,
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 512,
    ):
        super().__init__(
            "timdettmers/openassistant-guanaco",
            tokenizer
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset = []

        self.sep_regex = re.compile(r"### (Human|Assistant)")
        self.preprocess(raw_dataset)

    def preprocess(
            self,
            raw_dataset,
    ) -> None:
        """
        Preprocesses the dataset.
        It splits the text in sequential turns and inserts the utterances and the role of the speaker in a list of
        dictionaries. Then the chat template of the tokenizer is applied to the list of dictionaries in order to obtain
        a text formatted in the proper way.
        Finally, the text is tokenized and padded to the maximum length.

        Args:
            raw_dataset (datasets.Dataset):
                Raw dataset.
        """

        texts = [
            self.tokenizer.decode(
                self.tokenizer.apply_chat_template([
                    {
                        "role": "user" if message.split(":", 1)[0].strip() == "Human" else "assistant",
                        "content": message.split(":", 1)[1].strip()
                    } for message in self.sep_regex.sub("<sep/> \\1", dialogue["text"]).split("<sep/>") if
                    len(message) > 0
                ]),
                skip_special_tokens=False).strip() + self.tokenizer.eos_token
            for dialogue in raw_dataset
        ]

        tokenized_texts = [self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True
        ) for text in texts]

        self.dataset = [
            {
                "input_ids": input_encoding["input_ids"].squeeze(0),
                "labels": input_encoding["input_ids"].clone().squeeze(0),
                "attention_mask": input_encoding["attention_mask"].squeeze(0)
            }
            for input_encoding in tokenized_texts]

        for idx, _ in enumerate(self.dataset):
            self.dataset[idx]["labels"][["attention_mask"] == 0] = -100

    def __len__(
            self
    ):
        """
        Returns the length of the dataset.

        Returns:
            int:
                Length of the dataset.
        """

        return len(self.dataset)

    def __getitem__(
            self,
            idx
    ) -> dict:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            dict:
                Dictionary containing the tokenized inputs.
        """

        return self.dataset[idx]


class OpenAssistantGuanacoDataModule(pl.LightningDataModule):
    """
    DataModule for the OpenAssistant-Guanaco dataset.

    Args:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets. Defaults to (0.8, 0.1, 0.1).
        batch_size (int):
            Size of the batch. Defaults to 1.
        num_workers (int):
            Number of workers to use for loading the data. Defaults to 1.
        seed (int):
            Seed for the random number generator.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets.
        batch_size (int):
            Size of the batch.
        num_workers (int):
            Number of workers to use for loading the data.
        seed (int):
            Seed for the random number generator.
        train (OpenAssistantGuanacoDataset):
            Training dataset.
        validation (OpenAssistantGuanacoDataset):
            Validation dataset.
        test (OpenAssistantGuanacoDataset):
            Test dataset.
    """

    def __init__(
            self,
            tokenizer,
            max_len: int,
            split: tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 1,
            num_workers: int = 1,
            seed: int = 42,
    ) -> None:
        super().__init__()
        if len(split) != 3:
            raise ValueError(
                "The split must have three elements (train, validation, test)."
            )
        if sum(split) != 1:
            raise ValueError(
                "The sum of the split elements must be equal to 1."
            )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train = None
        self.validation = None
        self.test = None

    def prepare_data(
            self,
            **kwargs
    ):
        """
        Downloads the data.
        Runs on a single GPU.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        load_dataset(
            "timdettmers/openassistant-guanaco",
        )

    def setup(
            self,
            stage: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Preprocesses data.
        Can run on multiple GPUs.


        Args:
            stage (Optional[str]):
                Stage of the experiment.
            kwargs:
                Additional keyword arguments.
        """

        raw_dataset = load_dataset(
            "timdettmers/openassistant-guanaco",
        )

        concatenated_raw_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]])

        first_split_raw_dataset = concatenated_raw_dataset.train_test_split(
            test_size=self.split[2]
        )
        second_split_raw_dataset = first_split_raw_dataset["train"].train_test_split(
            test_size=self.split[1]/((1-self.split[2]) if self.split[2] != 1 else 1)
        )

        self.train = OpenAssistantGuanacoDataset(
            second_split_raw_dataset["train"],
            self.tokenizer,
            self.max_len
        )
        self.validation = OpenAssistantGuanacoDataset(
            second_split_raw_dataset["test"],
            self.tokenizer,
            self.max_len
        )
        self.test = OpenAssistantGuanacoDataset(
            first_split_raw_dataset["test"],
            self.tokenizer,
            self.max_len
        )

    def train_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the training DataLoader.

        Returns:
            DataLoader:
                Training DataLoader.
        """

        if self.train is None:
            self.setup()

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader:
                Validation DataLoader.
        """

        if self.validation is None:
            self.setup()

        return DataLoader(
            self.validation,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers
        )

    def test_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the test DataLoader.

        Returns:
            DataLoader:
                Test DataLoader.
        """

        if self.test is None:
            self.setup()

        return DataLoader(
            self.test,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers
        )


class Wikitext2Dataset(LanguageModelingDataset):
    """
    Dataset class to use Wikitext-2 dataset with Pytorch and Pytorch Lightning.

    Args:
        raw_dataset (datasets.Dataset):
            Raw dataset.
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_length (int):
            Maximum length of the input sequences.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_length (int):
            Maximum length of the input sequences.
        dataset (list):
            Tokenized dataset.
    """

    def __init__(
            self,
            raw_dataset: datasets.Dataset,
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 512,
    ):
        super().__init__(
            "wikitext-2-raw-v1",
            tokenizer
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset = []

        self.preprocess(raw_dataset)

    def preprocess(
            self,
            raw_dataset,
    ) -> None:
        """
        Preprocesses the dataset.
        The text is tokenized and padded to the maximum length.

        Args:
            raw_dataset (datasets.Dataset):
                Raw dataset.
        """

        tokenized_texts = [self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True
        ) for text in raw_dataset["text"]]

        self.dataset = [
            {
                "input_ids": input_encoding["input_ids"].squeeze(0),
                "labels": input_encoding["input_ids"].clone().squeeze(0),
                "attention_mask": input_encoding["attention_mask"].squeeze(0)
            }
            for input_encoding in tokenized_texts]

        for idx, _ in enumerate(self.dataset):
            self.dataset[idx]["labels"][["attention_mask"] == 0] = -100

    def __len__(
            self
    ):
        """
        Returns the length of the dataset.

        Returns:
            int:
                Length of the dataset.
        """

        return len(self.dataset)

    def __getitem__(
            self,
            idx
    ) -> dict:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            dict:
                Dictionary containing the tokenized inputs.
        """

        return self.dataset[idx]


class Wikitext2DataModule(pl.LightningDataModule):
    """
    DataModule for Wikitext-2 dataset for language modeling tasks.

    Args:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets. Defaults to (0.8, 0.1, 0.1).
        batch_size (int):
            Size of the batch. Defaults to 1.
        num_workers (int):
            Number of workers to use for loading the data. Defaults to 1.
        seed (int):
            Seed for the random number generator.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets.
        batch_size (int):
            Size of the batch.
        num_workers (int):
            Number of workers to use for loading the data.
        seed (int):
            Seed for the random number generator.
        train (OpenAssistantGuanacoDataset):
            Training dataset.
        validation (OpenAssistantGuanacoDataset):
            Validation dataset.
        test (OpenAssistantGuanacoDataset):
            Test dataset.
    """

    def __init__(
            self,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
            max_len: int,
            split: tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 1,
            num_workers: int = 1,
            seed: int = 42,
    ) -> None:
        super().__init__()
        if len(split) != 3:
            raise ValueError(
                "The split must have three elements (train, validation, test)."
            )
        if sum(split) != 1:
            raise ValueError(
                "The sum of the split elements must be equal to 1."
            )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train = None
        self.validation = None
        self.test = None

    def prepare_data(
            self,
            **kwargs
    ):
        """
        Downloads the data.
        Runs on a single GPU.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        load_dataset(
            "wikitext", "wikitext-2-raw-v1",
        )

    def setup(
            self,
            stage: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Preprocesses data.
        Can run on multiple GPUs.


        Args:
            stage (Optional[str]):
                Stage of the experiment.
            kwargs:
                Additional keyword arguments.
        """

        raw_dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1",
        )

        concatenated_raw_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]])

        first_split_raw_dataset = concatenated_raw_dataset.train_test_split(
            test_size=self.split[2]
        )
        second_split_raw_dataset = first_split_raw_dataset["train"].train_test_split(
            test_size=self.split[1] / ((1 - self.split[2]) if self.split[2] != 1 else 1)
        )

        self.train = Wikitext2Dataset(
            second_split_raw_dataset["train"],
            self.tokenizer,
            self.max_len
        )
        self.validation = Wikitext2Dataset(
            second_split_raw_dataset["test"],
            self.tokenizer,
            self.max_len
        )
        self.test = Wikitext2Dataset(
            first_split_raw_dataset["test"],
            self.tokenizer,
            self.max_len
        )

    def train_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the training DataLoader.

        Returns:
            DataLoader:
                Training DataLoader.
        """

        if self.train is None:
            self.setup()

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader:
                Validation DataLoader.
        """

        if self.validation is None:
            self.setup()

        return DataLoader(
            self.validation,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers
        )

    def test_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the test DataLoader.

        Returns:
            DataLoader:
                Test DataLoader.
        """

        if self.test is None:
            self.setup()

        return DataLoader(
            self.test,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers
        )


class OpenWebTextIterableDataset(IterableDataset):
    """
    IterableDataset for streaming OpenWebText data with custom splits.

    Args:
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_length (int):
            Maximum length of the input sequences.
        start (float):
            Start of the split range.
        end (float):
            End of the split range.

    Attributes:
        dataset (datasets.Dataset):
            OpenWebText dataset.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_length (int):
            Maximum length of the input sequences.
        start (float):
            Start of the split range.
        end (float):
            End of the split range.
    """

    def __init__(
            self,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
            max_length: int = 512,
            start: float = 0.,
            end: float = 1.
    ) -> None:
        super().__init__()
        self.dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.start = start
        self.end = end

    def __len__(
            self
    ) -> int:
        """
        Returns the number of examples in the dataset.

        Returns:
            int:
                Number of examples in the dataset
        """

        return self.dataset.info.splits["train"].num_examples

    def __iter__(
            self
    ) -> Iterator[dict]:
        """
        Yields tokenized data samples based on split range.
        """

        for idx, item in enumerate(self.dataset):
            if not (self.start <= (idx % 1) < self.end):
                continue  # Skip items outside the split range

            # Tokenize and pad the text to the maximum length
            tokenized = self.tokenizer(
                item["text"],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                add_special_tokens=True
            )
            yield {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "labels": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            }


class OpenWebTextStreamingDataModule(pl.LightningDataModule):
    """
    DataModule for OpenWebText dataset for language modeling tasks.

    Args:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets. Defaults to (0.8, 0.1, 0.1).
        batch_size (int):
            Size of the batch. Defaults to 1.
        num_workers (int):
            Number of workers to use for loading the data. Defaults to 1.
        seed (int):
            Seed for the random number generator.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer to use to preprocess the data.
        max_len (int):
            Maximum length of the input sequences.
        split (tuple[float, float, float]):
            Split of the dataset into training, validation, and test sets.
        batch_size (int):
            Size of the batch.
        num_workers (int):
            Number of workers to use for loading the data.
        seed (int):
            Seed for the random number generator.
        train (OpenAssistantGuanacoDataset):
            Training dataset.
        validation (OpenAssistantGuanacoDataset):
            Validation dataset.
        test (OpenAssistantGuanacoDataset):
            Test dataset.
    """

    def __init__(
            self,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
            max_len: int,
            split: tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 1,
            num_workers: int = 1,
            seed: int = 42,
    ) -> None:
        super().__init__()

        super().__init__()
        if len(split) != 3:
            raise ValueError(
                "The split must have three elements (train, validation, test)."
            )
        if sum(split) != 1:
            raise ValueError(
                "The sum of the split elements must be equal to 1."
            )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train = None
        self.validation = None
        self.test = None

    def prepare_data(
            self,
            **kwargs
    ):
        """
        Downloads the data.
        Runs on a single GPU.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        load_dataset("Skylion007/openwebtext", split='train', streaming=True, trust_remote_code=True)

    def setup(
            self,
            stage: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Preprocesses data.
        Can run on multiple GPUs.
        OpenWebText is used for training, while Wikitext2 is used for validation and test.

        Args:
            stage (Optional[str]):
                Stage of the experiment.
            kwargs:
                Additional keyword arguments.
        """

        self.train = OpenWebTextIterableDataset(self.tokenizer, self.max_len, 0., self.split[0])

        # Validation and test are done on Wikitext2 dataset
        wikitext2_datamodule = Wikitext2DataModule(self.tokenizer, self.max_len, self.split, self.batch_size, self.num_workers, self.seed)
        wikitext2_datamodule.setup()
        self.validation = wikitext2_datamodule.validation
        self.test = wikitext2_datamodule.test
        del wikitext2_datamodule.train
        gc.collect()

        #self.validation = OpenWebTextIterableDataset(self.tokenizer, self.max_len, self.split[0], self.split[1])
        #self.test = OpenWebTextIterableDataset(self.tokenizer, self.max_len,self.split[1], 1.)

    def train_dataloader(
            self
    ) -> DataLoader:
        """
        Returns the train DataLoader.

        Returns:
            DataLoader:
                Train DataLoader.
        """

        if self.train is None:
            self.setup()

        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader:
                Validation DataLoader.
        """

        if self.validation is None:
            self.setup()

        return DataLoader(self.validation, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Returns the test DataLoader.

        Returns:
            DataLoader:
                Test DataLoader.
        """

        if self.test is None:
            self.setup()

        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
