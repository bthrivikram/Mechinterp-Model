import random

from train.create_dataset import render_number, render_answer


class PromptDataset:
    """A collection of prompts to run activation-extraction inference on."""

    def __init__(self) -> None:
        self.prompts: list[dict] = []

    def __len__(self) -> int:
        return len(self.prompts)

    @classmethod
    def generate_prompts(
        cls,
        num_prompts: int,
        language: str = "english",
        max_operand: int = 999,
    ) -> "PromptDataset":
        """Generate 0-shot prompts for multilingual addition."""

        instance = cls()

        for _ in range(num_prompts):
            a = random.randint(0, max_operand)
            b = random.randint(0, max_operand)

            prompt = (
                f"{render_number(a, language, width=3)}"
                f"+"
                f"{render_number(b, language, width=3)}="
            )

            answer = render_answer(a + b, language)

            instance.prompts.append(
                {
                    "prompt": prompt,
                    "metadata": {
                        "language": language,
                        "a": a,
                        "b": b,
                        "answer": answer,
                    },
                }
            )

        return instance