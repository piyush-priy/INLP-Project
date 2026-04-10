import logging
from torch.utils.data import DataLoader
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_calibration_data(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    max_seq_length: int = 512,
    max_tokens: int = 50_000,
    batch_size: int = 2,
    split: str = "train",
):
    """
    Load and tokenize a calibration dataset.

    Returns:
        (train_dataloader, val_dataloader)
    """
    logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
    ds = load_dataset(dataset_name, dataset_config)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Filter empty strings
    train_ds = ds[split].filter(lambda x: len(x["text"].strip()) > 0)
    val_ds = ds["validation"].filter(lambda x: len(x["text"].strip()) > 0)

    # Estimate how many examples to keep based on max_tokens
    avg_tokens = max_seq_length * 0.7  # rough estimate
    max_examples = int(max_tokens / avg_tokens)
    if len(train_ds) > max_examples:
        train_ds = train_ds.select(range(max_examples))
        logger.info(f"Trimmed training set to {max_examples} examples")

    train_ds = train_ds.map(
        tokenize_fn, batched=True, remove_columns=train_ds.column_names
    )
    val_ds = val_ds.map(
        tokenize_fn, batched=True, remove_columns=val_ds.column_names
    )

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    logger.info(
        f"Data ready: {len(train_ds)} train, {len(val_ds)} val examples"
    )
    return train_loader, val_loader
