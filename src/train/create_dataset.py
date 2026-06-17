"""Generate a synthetic training dataset and save it (locally by default, or push to the HF Hub).

This is a task-agnostic skeleton: it parses CLI arguments and saves a DatasetDict (locally by
default, or pushes it to the Hub with --hub-name). You implement `build_dataset()` to produce
the actual data for your task.

`build_dataset` must return a `datasets.DatasetDict` with "train" and "validation"
splits. Each row must contain at least a "prompt" and an "answer" column; add any
extra metadata columns your training/eval needs.

A leakage-safe recipe to follow inside `build_dataset` (strongly recommended):
  1. Enumerate every atomic "item" your questions are built from (e.g. each (a, b)
     operand pair) and shuffle/split them into disjoint train/val pools BEFORE
     sampling any example. This is what prevents train/val leakage.
  2. For each row, sample the question item from the split's own pool, but draw any
     few-shot examples from the TRAIN pool only -- so a val question is never seen
     during training in any form, while val prompts still demonstrate the pattern.

A self-contained worked example (single-digit addition, 4-shot) is given in the
commented `build_dataset` body below; delete it and write your own.
"""

import argparse

from datasets import DatasetDict
import numpy as np
from datasets import Dataset, DatasetDict

DIGIT_MAPS = {
    "english": list("0123456789"),
    "hindi": list("०१२३४५६७८९"),
    "arabic": list("٠١٢٣٤٥٦٧٨٩"),
    "mandarin": list("零一二三四五六七八九"),
}
LANGUAGES = list(DIGIT_MAPS.keys())


def render_number(n: int, lang: str) -> str:
    """Render an integer digit-by-digit in the target language's numeral script."""
    digits = DIGIT_MAPS[lang]
    return "".join(digits[int(d)] for d in str(n))


def has_carry(a: int, b: int) -> bool:
    """True if a + b produces a carry in at least one column (base 10)."""
    carry, da, db = 0, str(a)[::-1], str(b)[::-1]
    for i in range(max(len(da), len(db))):
        da_i = int(da[i]) if i < len(da) else 0
        db_i = int(db[i]) if i < len(db) else 0
        carry = 1 if da_i + db_i + carry >= 10 else 0
    return bool(carry)


def render_equation(a: int, b: int, lang: str, solved: bool) -> str:
    """'x+y=z' (solved) or 'x+y=' (unsolved). No separator -- equations are concatenated
    directly, so example boundaries are recoverable only via the next '+' sign.

    ASSUMPTION to double-check: for Arabic we swap operand order ("y+x=z") as the
    operationalization of "RTL surface order." The '=' and answer never move, since
    generation is autoregressive and the answer must stay last regardless of language.
    """
    ra, rb = render_number(a, lang), render_number(b, lang)
    if lang == "arabic":
        ra, rb = rb, ra
    if solved:
        return f"{ra}+{rb}={render_number(a + b, lang)}"
    return f"{ra}+{rb}="


def build_dataset(train_size: int, val_size: int, seed: int, val_holdout: float) -> DatasetDict:
    """Build the train/validation DatasetDict for your task.

    Args:
        train_size: Number of training rows to generate.
        val_size: Number of validation rows to generate.
        seed: Random seed for reproducibility.
        val_holdout: Fraction of unique items to reserve for validation questions.

    Returns:
        A DatasetDict with "train" and "validation" splits. Each row must contain
        at least "prompt" and "answer" columns.
    """
    # ----------------------------------------------------------------------- #
    # TODO: implement dataset generation for your task.
    # ----------------------------------------------------------------------- #
    #
    # Example (single-digit addition, 4-shot, leakage-safe -- matches the commented examples
    # in src/train/train_tokenizer.py and src/utils/dataset.py):
    #
    #     import numpy as np
    #     from datasets import Dataset
    #
    #     few_shot = 4
    #
    #     # 1. Enumerate every (a, b) item, then split into disjoint train/val pools so a
    #     #    validation question never appears as a training question.
    #     items = [(a, b) for a in range(10) for b in range(10)]
    #     rng = np.random.default_rng(seed)
    #     rng.shuffle(items)
    #     split_idx = int(len(items) * (1 - val_holdout))
    #     pools = {"train": items[:split_idx], "val": items[split_idx:]}
    #
    #     def render(a: int, b: int) -> str:
    #         return f"{a}+{b}={a + b}"  # one solved example, e.g. "7+5=12"
    #
    #     # 2. Build one split: the question comes from `q_pool`, few-shot always from train.
    #     def generate_split(total: int, q_pool: list, split_name: str, split_seed: int) -> dict:
    #         srng = np.random.default_rng(split_seed)
    #         fs_pool = pools["train"]
    #         rows = []
    #         for i in range(total):
    #             a, b = q_pool[srng.integers(len(q_pool))]
    #             fs = [render(*fs_pool[j]) for j in srng.integers(len(fs_pool), size=few_shot)]
    #             prompt = "\n".join(fs) + f"\n{a}+{b}="  # ends at "=", the model produces the sum
    #             rows.append(
    #                 {"_id": f"{split_name}-{i}", "question": f"{a}+{b}", "answer": str(a + b), "prompt": prompt}
    #             )
    #         return {k: [r[k] for r in rows] for k in rows[0]}
    #
    #     train_data = generate_split(train_size, pools["train"], "train", seed)
    #     val_data = generate_split(val_size, pools["val"], "validation", seed + 1)
    #     return DatasetDict({
    #         "train": Dataset.from_dict(train_data),
    #         "validation": Dataset.from_dict(val_data),
    #     })
    #
    # ----------------------------------------------------------------------- #

    few_shot = 8
    max_operand = 99
    carry_few_shot_min = 2        # Section 5: "at least two of the eight ... involve a carry"
    carry_oversample_frac = 0.5   # Section 5: carry problems oversampled in the corpus

    rng = np.random.default_rng(seed)

    # Enumerate every (a, b) item once; split into disjoint train/val pools so a val
    # question is never seen as a training question. Reused across languages so they
    # see identical operand distributions.
    items = [(a, b) for a in range(max_operand + 1) for b in range(max_operand + 1)]
    rng.shuffle(items)
    split_idx = int(len(items) * (1 - val_holdout))
    pools = {"train": items[:split_idx], "val": items[split_idx:]}
    carry_pools = {split: [it for it in pool if has_carry(*it)] for split, pool in pools.items()}

    def sample_question(split: str, srng: np.random.Generator) -> tuple[int, int]:
        pool = carry_pools[split] if srng.random() < carry_oversample_frac else pools[split]
        return pool[srng.integers(len(pool))]

    def sample_few_shot(srng: np.random.Generator) -> list[tuple[int, int]]:
        carry_idx = srng.integers(len(carry_pools["train"]), size=carry_few_shot_min)
        carry_ex = [carry_pools["train"][i] for i in carry_idx]
        rest_idx = srng.integers(len(pools["train"]), size=few_shot - carry_few_shot_min)
        rest = [pools["train"][i] for i in rest_idx]
        examples = carry_ex + rest
        srng.shuffle(examples)
        return examples

    def generate_split(total: int, q_pool: str, split_name: str, split_seed: int) -> dict:
        srng = np.random.default_rng(split_seed)
        rows = []
        for i in range(total):
            lang = LANGUAGES[i % len(LANGUAGES)]  # equal distribution across languages
            a, b = sample_question(q_pool, srng)
            fs_segments = [render_equation(x, y, lang, solved=True) for x, y in sample_few_shot(srng)]
            prompt = "\n".join(fs_segments) + "\n" + render_equation(a, b, lang, solved=False)
            rows.append({
                "_id": f"{split_name}-{i}",
                "language": lang,
                "question": f"{a}+{b}",
                "answer": render_number(a + b, lang),
                "prompt": prompt,
                "has_carry": has_carry(a, b),
            })
        srng.shuffle(rows)
        return {k: [r[k] for r in rows] for k in rows[0]}

    train_data = generate_split(train_size, "train", "train", seed)
    val_data = generate_split(val_size, "val", "validation", seed + 1)
    return DatasetDict({"train": Dataset.from_dict(train_data), "validation": Dataset.from_dict(val_data)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset and save it locally or to the Hub")
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HF Hub dataset repo to push to (e.g. your-username/your-dataset-name). "
        "If omitted, the dataset is saved locally to --output-dir instead (no login needed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/dataset",
        help="Local directory to save the dataset to when --hub-name is not given.",
    )
    parser.add_argument("--train-size", type=int, default=1_000_000, help="Number of training rows")
    parser.add_argument("--val-size", type=int, default=100_000, help="Number of validation rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (val uses seed + 1)")
    parser.add_argument(
        "--val-holdout",
        type=float,
        default=0.1,
        help="Fraction of unique items held out for val questions (default: 0.1). "
        "Independent of --val-size, which controls the number of val rows.",
    )
    parser.add_argument("--shard-size", type=str, default="500MB", help="Max shard size per Parquet file on the Hub")
    args = parser.parse_args()

    print(f"Generating dataset | train={args.train_size:,} val={args.val_size:,}\n")

    dataset = build_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        val_holdout=args.val_holdout,
    )

    if args.hub_name:
        dataset.push_to_hub(args.hub_name, max_shard_size=args.shard_size)
        print(f"\nPushed dataset to hub: {args.hub_name}")
    else:
        dataset.save_to_disk(args.output_dir)
        print(f"\nSaved dataset to {args.output_dir}  (pass --hub-name to push to the Hub instead)")