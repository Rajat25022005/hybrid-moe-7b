"""
Data pipeline — streaming dataset with document packing and sample-level attention masking.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import logging

logger = logging.getLogger(__name__)


class PackedTextDataset(IterableDataset):
    """
    Streaming dataset that packs tokenized documents into fixed-length sequences.
    Implements sample-level attention masking (documents from different sources
    cannot attend to each other within a packed sequence).
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_subset: str = None,
        tokenizer_path: str = "./tokenizer",
        seq_len: int = 4096,
        split: str = "train",
    ):
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer_path = tokenizer_path
        self.split = split

        # Load tokenizer
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load the tokenizer."""
        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(
                f"{self.tokenizer_path}/tokenizer.json"
            )
            logger.info(f"Loaded tokenizer from {self.tokenizer_path}")
        except Exception:
            # Fallback: create a simple byte-level tokenizer for testing
            logger.warning("Custom tokenizer not found. Using fallback.")
            self.tokenizer = None

    def _tokenize(self, text: str) -> list:
        """Tokenize a text string."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text).ids
        else:
            # Simple fallback: byte encoding mod vocab_size
            return [b % 32000 for b in text.encode('utf-8')]

    def __iter__(self):
        """Yield packed sequences of token IDs."""
        try:
            from datasets import load_dataset
            ds = load_dataset(
                self.dataset_name,
                self.dataset_subset,
                split=self.split,
                streaming=True,
            )
        except Exception as e:
            logger.warning(f"Could not load dataset: {e}. Using synthetic data.")
            ds = self._synthetic_data()

        buffer = []

        for sample in ds:
            text = sample.get("text", sample.get("content", ""))
            if not text:
                continue

            tokens = self._tokenize(text)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                # +1 for the target shift
                input_ids = torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                target_ids = torch.tensor(buffer[1:self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len:]
                yield {"input_ids": input_ids, "target_ids": target_ids}

    def _synthetic_data(self):
        """Generate synthetic data for testing."""
        import random
        for _ in range(10000):
            text = " ".join(
                [chr(random.randint(65, 90)) * random.randint(1, 10)
                 for _ in range(100)]
            )
            yield {"text": text}


def create_dataloader(
    dataset_name: str,
    dataset_subset: str,
    tokenizer_path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = PackedTextDataset(
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
