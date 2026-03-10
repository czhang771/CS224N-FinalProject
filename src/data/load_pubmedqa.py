"""Load and format PubMedQA examples."""

from datasets import load_dataset

# Fixed split sizes (must sum to 1000)
TOTAL_SAMPLES = 1000
TRAIN_SIZE = 700
VAL_SIZE = 150
TEST_SIZE = 150
N_TRAIN_SHARDS = 3


def load_pubmedqa(
    split: str = "train",
    seed: int = 42,
) -> list[dict]:
    """
    Load PubMedQA (pqa_labeled) examples for a given split.

    Args:
        split: one of "train_shard_0", "train_shard_1", "train_shard_2", "val", "test"
               or "train" (returns all 700 training examples)
        seed:  shuffle seed — must be the same across all team members

    Returns:
        List of dicts with keys:
            pubid, question, context_text, ground_truth_label, ground_truth_long_answer
    """
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    dataset = dataset.shuffle(seed=seed)
    all_examples = [_format(item) for item in dataset.select(range(TOTAL_SAMPLES))]

    # Define slice boundaries
    train_examples = all_examples[:TRAIN_SIZE]
    val_examples   = all_examples[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test_examples  = all_examples[TRAIN_SIZE + VAL_SIZE :]

    if split == "val":
        return val_examples
    elif split == "test":
        return test_examples
    elif split == "train":
        return train_examples
    elif split.startswith("train_shard_"):
        shard_idx = int(split.split("_")[-1])
        return _shard_slice(train_examples, shard_idx, N_TRAIN_SHARDS)
    else:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Use 'train', 'train_shard_0/1/2', 'val', or 'test'."
        )


def _shard_slice(examples: list[dict], shard: int, n_shards: int) -> list[dict]:
    chunk = len(examples) // n_shards
    start = shard * chunk
    end = start + chunk if shard < n_shards - 1 else len(examples)
    return examples[start:end]


def _format(item: dict) -> dict:
    passages = item["context"]["contexts"]
    return {
        "pubid": str(item["pubid"]),
        "question": item["question"],
        "context_text": "\n\n".join(passages),
        "ground_truth_label": item["final_decision"],
        "ground_truth_long_answer": item["long_answer"],
    }
