"""Load and format PubMedQA examples."""

from datasets import load_dataset


def load_pubmedqa(n_samples: int = 100, seed: int = 42) -> list[dict]:
    """
    Load PubMedQA (pqa_labeled), shuffle, and return the first n_samples examples.

    Each returned dict has keys:
        pubid, question, context_text, ground_truth_label, ground_truth_long_answer
    """
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    examples = []
    for item in dataset:
        # context is a dict with key "contexts" (list of passage strings)
        passages = item["context"]["contexts"]
        context_text = "\n\n".join(passages)

        examples.append(
            {
                "pubid": str(item["pubid"]),
                "question": item["question"],
                "context_text": context_text,
                "ground_truth_label": item["final_decision"],
                "ground_truth_long_answer": item["long_answer"],
            }
        )

    return examples
