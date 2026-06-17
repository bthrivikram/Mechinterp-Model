import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./saved_models"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"Using device: {device}")

while True:
    prompt = input("\nEnter prompt: ").strip()

    if prompt.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)

    

    # Extract only the leading answer
    match = re.match(
        r"^[0-9०-९٠-٩零一二三四五六七八九]+",
        completion
    )

    if match:
        answer = match.group(0)
        print("Answer     :", answer)
    else:
        print("Answer     : <unable to parse>")