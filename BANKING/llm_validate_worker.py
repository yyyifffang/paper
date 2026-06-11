#!/usr/bin/env python3
import argparse
import gc
import json
import os
import re
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


VALIDATOR_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def get_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model():
    print(f"Loading validator model: {VALIDATOR_MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(VALIDATOR_MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        VALIDATOR_MODEL_ID,
        quantization_config=get_quant_config(),
        device_map="auto",
        max_memory={0: "20GiB", "cpu": "64GiB"},
        offload_folder="./offload_validate",
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


def validate_generated_texts(generated_texts, generated_labels, original_texts, label_mapping, log_file=None):
    if len(generated_texts) == 0:
        return {
            "valid_texts": [],
            "valid_labels": [],
            "accept_rate": 0.0,
            "decision_counts": {
                "ACCEPT": 0,
                "REVIEW": 0,
                "REJECT": 0,
                "UNKNOWN": 0,
            },
        }

    tokenizer, model = load_model()

    valid_texts = []
    valid_labels = []
    accepted = 0

    decision_counts = {
        "ACCEPT": 0,
        "REVIEW": 0,
        "REJECT": 0,
        "UNKNOWN": 0,
    }

    for text, label, orig_text in zip(generated_texts, generated_labels, original_texts):
        text = "" if text is None else str(text)
        orig_text = "" if orig_text is None else str(orig_text)
        label = int(label)
        label_name = label_mapping.get(str(label), label_mapping.get(label, f"Label {label}"))

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a banking intent classification data quality evaluator. "
                    "Your task is to decide whether the augmented text can still be correctly labeled "
                    f"as the target intent: '{label_name}'. "
                    "Allow natural paraphrases and wording changes if the banking intent remains unchanged. "
                    "Reject only if the augmented text changes the intent, becomes ambiguous, "
                    "adds unsupported information, or is unrelated."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target Intent: {label_name}\n\n"
                    f"Original Text: {orig_text}\n"
                    f"Augmented Text: {text}\n\n"
                    "Evaluation criteria:\n"
                    "1. Does the augmented text preserve the same banking intent?\n"
                    "2. Is the text not contradictory to the original intent?\n"
                    "3. Is the text specific enough to belong to the target intent rather than another intent?\n\n"
                    "Decision policy:\n"
                    "- ACCEPT: The augmented text clearly preserves the target intent.\n"
                    "- REVIEW: The augmented text may preserve the target intent, but it is ambiguous or not specific enough.\n"
                    "- REJECT: The augmented text changes the intent, adds unsupported information, or is unrelated.\n\n"
                    "Important: Do not reject only because the wording is different. "
                    "Reject only when the intent changes, becomes ambiguous, or contains irrelevant/fabricated details.\n\n"
                    "Output strictly in the following format:\n"
                    "<reasoning>...</reasoning>\n"
                    "<decision>ACCEPT</decision> or <decision>REVIEW</decision> or <decision>REJECT</decision>"
                ),
            },
        ]

        resp = generate_chat_response(tokenizer, model, messages)

        match_reason = re.search(
            r"<reasoning>(.*?)</reasoning>",
            resp,
            re.IGNORECASE | re.DOTALL,
        )
        reasoning = match_reason.group(1).strip() if match_reason else "N/A"

        match_decision = re.search(
            r"<decision>(ACCEPT|REVIEW|REJECT)</decision>",
            resp,
            re.IGNORECASE,
        )
        decision = match_decision.group(1).upper() if match_decision else "UNKNOWN"

        if decision not in decision_counts:
            decision = "UNKNOWN"

        decision_counts[decision] += 1

        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"VAL | [{label_name}] | {decision} | "
                    f"Reason: {reasoning} | "
                    f"Original: {orig_text} | "
                    f"Augmented: {text}\n"
                )

        if decision == "ACCEPT":
            valid_texts.append(text)
            valid_labels.append(label)
            accepted += 1

    accept_rate = accepted / max(len(generated_texts), 1)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                "VAL_SUMMARY | "
                f"ACCEPT: {decision_counts['ACCEPT']} | "
                f"REVIEW: {decision_counts['REVIEW']} | "
                f"REJECT: {decision_counts['REJECT']} | "
                f"UNKNOWN: {decision_counts['UNKNOWN']} | "
                f"Accept Rate: {accept_rate:.4f}\n"
            )

    print(
        "Validator Decision Summary | "
        f"ACCEPT: {decision_counts['ACCEPT']} | "
        f"REVIEW: {decision_counts['REVIEW']} | "
        f"REJECT: {decision_counts['REJECT']} | "
        f"UNKNOWN: {decision_counts['UNKNOWN']} | "
        f"Accept Rate: {accept_rate:.2f}"
    )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return {
        "valid_texts": valid_texts,
        "valid_labels": valid_labels,
        "accept_rate": accept_rate,
        "decision_counts": decision_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result = validate_generated_texts(
        generated_texts=payload["generated_texts"],
        generated_labels=payload["generated_labels"],
        original_texts=payload["original_texts"],
        label_mapping=payload["label_mapping"],
        log_file=args.log_file,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()