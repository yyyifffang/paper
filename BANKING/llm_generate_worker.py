#!/usr/bin/env python3
import argparse
import gc
import json
import os
import re
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def get_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model():
    print(f"Loading generator model: {LLM_MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=get_quant_config(),
        device_map="auto",
        max_memory={0: "20GiB", "cpu": "64GiB"},
        offload_folder="./offload_generate",
    )

    model.eval()
    return tokenizer, model


def get_input_device(model):
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_chat_response(tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_device = get_input_device(model)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def generate_variants(texts, labels, label_mapping, n_variants=1, log_file=None):
    tokenizer, model = load_model()

    augmented_texts = []
    augmented_labels = []
    source_texts = []
    records = []

    for text, label in zip(texts, labels):
        text = "" if text is None else str(text)
        label = int(label)
        label_name = label_mapping.get(str(label), label_mapping.get(label, f"Intent_{label}"))

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise banking customer service text augmentation assistant. "
                    f"Your task is to generate variations strictly belonging to: '{label_name}'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target Intent Category: {label_name}\n\n"
                    "Generate variations for the following customer service text.\n"
                    f"The variation MUST strictly preserve the exact semantic meaning of '{label_name}'.\n"
                    "Do not add new entities, numbers, dates, account statuses, transaction details, "
                    "or constraints that are not implied by the original text.\n"
                    "The variation should be a natural paraphrase of the original text, not a new scenario.\n"
                    f"Output exactly {n_variants} variation(s), each strictly enclosed within "
                    "<variation></variation> tags.\n"
                    "Do NOT include titles, numbers, markdown, or introductory text.\n\n"
                    f"Original text:\n{text}"
                ),
            },
        ]

        response_text = generate_chat_response(tokenizer, model, messages)
        matches = re.findall(
            r"<variation>(.*?)</variation>",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        variations = [m.strip() for m in matches if m.strip()]

        for var in variations[:n_variants]:
            augmented_texts.append(var)
            augmented_labels.append(label)
            source_texts.append(text)
            records.append(
                {
                    "label": label,
                    "label_name": label_name,
                    "original": text,
                    "generated": var,
                }
            )

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            for r in records:
                f.write(
                    f"GEN | [{r['label_name']}] | "
                    f"Orig: {r['original']} -> Aug: {r['generated']}\n"
                )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return {
        "generated_texts": augmented_texts,
        "generated_labels": augmented_labels,
        "source_texts": source_texts,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result = generate_variants(
        texts=payload["texts"],
        labels=payload["labels"],
        label_mapping=payload["label_mapping"],
        n_variants=payload.get("n_variants", 1),
        log_file=args.log_file,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()