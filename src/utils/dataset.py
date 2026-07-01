import random

from train.create_dataset import _apply_op, _format_answer, _format_operand


class PromptDataset:
    """A collection of prompts to run activation-extraction inference on."""

    def __init__(self) -> None:
        self.prompts: list[dict] = []

    def __len__(self) -> int:
        return len(self.prompts)

    @classmethod
    def generate_prompts(cls, num_prompts: int) -> "PromptDataset":
        """Generate 0-shot prompts for 3-digit addition."""

        instance = cls()

        digits = 3
        max_operand = 999

        for _ in range(num_prompts):
            a = random.randint(0, max_operand)
            b = random.randint(0, max_operand)

            prompt = (
                f"{_format_operand(a, digits)}"
                f"+"
                f"{_format_operand(b, digits)}="
            )

            answer = _format_answer(_apply_op(a, b, "+"))

            instance.prompts.append(
                {
                    "prompt": prompt,
                    "metadata": {
                        "base_operation": "+",
                        "target_operation": "+",
                        "a": a,
                        "b": b,
                        "answer": answer,
                    },
                }
            )

        return instance