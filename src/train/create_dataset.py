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
    "mandarin": list("零一二三四五六七八九"),
}
LANGUAGES = list(DIGIT_MAPS.keys())
 
PAD_WIDTH = 2  # zero-pad operands/answers to 3 digits
 
 
def render_number(n: int, lang: str, width: int = PAD_WIDTH) -> str:
    """Render an integer digit-by-digit in the target language's numeral script,
    zero-padded to `width` digits. Used for OPERANDS only (forward digit order)."""
    digits = DIGIT_MAPS[lang]
    padded = str(n).zfill(width)
    return "".join(digits[int(d)] for d in padded)
 
 
def render_answer(n: int, lang: str) -> str:
    """Render the answer with REVERSED digit order and NO zero-padding.
 
    e.g. 12 -> "21" (ones digit first, then tens digit). This aligns
    generation order with carry propagation (Lee et al., 2023) and means
    the LAST generated answer token is always the original most-significant
    digit -- i.e. the token whose hidden state has accumulated the most
    upstream autoregressive context, since it is generated after every
    other answer digit. This is a deliberate departure from render_number's
    forward + zero-padded convention, which is kept for operands only.
    """
<<<<<<< HEAD
    digits = DIGIT_MAPS[lang]
    return "".join(digits[int(d)] for d in str(n)[::-1])
 
 
def render_equation(a: int, b: int, lang: str, solved: bool) -> str:
    """'x+y=z' (solved) or 'x+y=' (unsolved). No separator, no few-shot context.
 
    Operands are forward digit order, zero-padded to PAD_WIDTH (render_number).
    The answer (when solved=True) is reversed digit order, unpadded (render_answer).
    """
    ra, rb = render_number(a, lang), render_number(b, lang)
    if solved:
        return f"{ra}+{rb}={render_answer(a + b, lang)}"
    return f"{ra}+{rb}="
 
 
def build_dataset(seed: int, val_holdout: float, max_operand: int) -> DatasetDict:
    """Build the train/validation DatasetDict for your task.
 
    Enumerates the exact coordinate universe based on max_operand
    (e.g. 100x100 = 10,000 unique (a, b) pairs per language at max_operand=99),
    shuffles independently per language, and splits evenly.
    """
    all_items = [(a, b) for a in range(max_operand + 1) for b in range(max_operand + 1)]
 
    train_rows = []
    val_rows = []
 
    for idx, lang in enumerate(LANGUAGES):
        # Distinct seed per language -> independent shuffle/split per script.
        lang_rng = np.random.default_rng(seed + idx)
 
        lang_items = all_items.copy()
        lang_rng.shuffle(lang_items)
 
        split_idx = int(len(lang_items) * (1 - val_holdout))
        train_pool = lang_items[:split_idx]
        val_pool = lang_items[split_idx:]
 
        for i, (a, b) in enumerate(train_pool):
            train_rows.append({
                "_id": f"train-{lang}-{i}",
                "language": lang,
                "eng_question": f"{a}+{b}",
                "eng_answer": str(a + b),
                "response": render_answer(a + b, lang),
                "prompt": render_equation(a, b, lang, solved=False),
            })
 
        for i, (a, b) in enumerate(val_pool):
            val_rows.append({
                "_id": f"val-{lang}-{i}",
                "language": lang,
                "eng_question": f"{a}+{b}",
                "eng_answer": str(a + b),
                "response": render_answer(a + b, lang),
                "prompt": render_equation(a, b, lang, solved=False),
            })
 
    # Shuffle the final global lists so that scripts are fully interwoven
    # across batches during a training epoch.
    global_rng = np.random.default_rng(seed)
    global_rng.shuffle(train_rows)
    global_rng.shuffle(val_rows)
 
    return DatasetDict({
        "train": Dataset.from_dict({k: [r[k] for r in train_rows] for k in train_rows[0]}),
        "validation": Dataset.from_dict({k: [r[k] for r in val_rows] for k in val_rows[0]}),
    })

=======
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
    #     # 2. Build one split: the question comes from `q_pool`, few-shot ALWAYS from train. Each
    #     #    shot looks like "7+5=12" and the prompt ends at "=", exactly the shape produced by
    #     #    src/utils/dataset.py's example -- the same prompts you'll later run inference on. The
    #     #    a/b/answer columns mirror the ground-truth dict that example stores under "metadata".
    #     def generate_split(total: int, q_pool: list, split_name: str, split_seed: int) -> dict:
    #         srng = np.random.default_rng(split_seed)
    #         fs_pool = pools["train"]
    #         rows = []
    #         for i in range(total):
    #             shots = []                                     # the solved examples shown first
    #             for j in srng.integers(len(fs_pool), size=few_shot):
    #                 fa, fb = fs_pool[j]
    #                 shots.append(f"{fa}+{fb}={fa + fb}")       # e.g. "7+5=12"
    #             a, b = q_pool[srng.integers(len(q_pool))]
    #             prompt = "\n".join(shots) + f"\n{a}+{b}="      # ends at "=", answer to come
    #             rows.append(
    #                 {"_id": f"{split_name}-{i}", "a": a, "b": b, "answer": str(a + b), "prompt": prompt}
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
    raise NotImplementedError("Implement build_dataset() for your task -- see the example in comments.")
>>>>>>> upstream/main


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
    #parser.add_argument("--train-size", type=int, default=20_000, help="Number of training rows")
    #parser.add_argument("--val-size", type=int, default=20_000, help="Number of validation rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (val uses seed + 1)")
    parser.add_argument(
        "--val-holdout",
        type=float,
        default=0.5,
        help="Fraction of unique items held out for val questions (default: 0.1). "
        "Independent of --val-size, which controls the number of val rows.",
    )
    parser.add_argument("--shard-size", type=str, default="500MB", help="Max shard size per Parquet file on the Hub")
    args = parser.parse_args()

    #print(f"Generating dataset | train={args.train_size:,} val={args.val_size:,}\n")

    dataset = build_dataset(
        max_operand = 99,
        seed=args.seed,
        val_holdout=args.val_holdout,
    )

    if args.hub_name:
        dataset.push_to_hub(args.hub_name, max_shard_size=args.shard_size)
        print(f"\nPushed dataset to hub: {args.hub_name}")
    else:
        dataset.save_to_disk(args.output_dir)
        print(f"\nSaved dataset to {args.output_dir}  (pass --hub-name to push to the Hub instead)")