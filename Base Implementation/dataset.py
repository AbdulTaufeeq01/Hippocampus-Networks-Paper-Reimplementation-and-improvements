from datasets import load_dataset
from transformers import AutoTokeinizer
from config import Config

tokenizer=AutoTokeinizer.from_pretrained("gpt2")


def load_data():
    dataset=load_dataset("wikitext","wikitext-2-raw-v1")
    text=dataset["train"]["text"]

    tokens=tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=Config.seq_len,
        return_tensors="pt"
    )

    return tokens["input_ids"]