import os
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKEN_NUMS = 3000000
RATIO = 1.0
EN_DATA_DIR = "raw_data_en.jsonl"
ZH_DATA_DIR = ""
OUTPUT_FILE = "test_1b.jsonl"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n\n### Input:\n\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n\n### Response:"
    ),
}


def get_token_count(sample):
    instruction_len = len(enc.tokenize(sample["instruction"]))
    input_len = len(enc.tokenize(sample["input"]))
    output_len = len(enc.tokenize(sample["output"]))
    total_prompt_input_len = instruction_len + input_len + prompt_input_len
    return total_prompt_input_len


def findAllFile(base):
    files = []
    if os.path.isfile(base):
        return [base]
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith("jsonl"):
                files.append(os.path.join(root, f))
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_data_dir", type=str, default=EN_DATA_DIR)
    parser.add_argument("--zh_data_dir", type=str, default=ZH_DATA_DIR)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--en_token_ratio", type=float, default=RATIO)
    args = parser.parse_args()

    en_data_dir = args.en_data_dir
    zh_data_dir = args.zh_data_dir
    output_file = args.output_file
    ratio = args.en_token_ratio

    enc = AutoTokenizer.from_pretrained("/workspace/data/models/falcon-rw-1b")
    enc.model_max_length = 1000000000000000019884624838656
    prompt_input_len = len(enc.tokenize(PROMPT_DICT["prompt_input"]))

    en_files = findAllFile(en_data_dir)
    ds_en = load_dataset("json", data_files=en_files, split="train").shuffle(seed=123)
    en_token_nums = TOKEN_NUMS * ratio if zh_data_dir else TOKEN_NUMS
    zh_token_nums = TOKEN_NUMS - en_token_nums

    count = 0
    for i in range(len(ds_en)):
        count += get_token_count(ds_en[i])
        if count >= en_token_nums:
            break
    print(f"Select {i + 1} samples from {len(ds_en)} samples in en dataset - {(i + 1) / len(ds_en) * 100:.2f}%")
    print(f"Total token nums: {count}, Avg token nums: {count / (i + 1):.2f}")
    ds_en = ds_en.select(range(i + 1)).select_columns(["instruction", "input", "output"])

    if zh_data_dir:
        zh_files = findAllFile(zh_data_dir)
        ds_zh = load_dataset("json", data_files=zh_files, split="train").shuffle(seed=123)
        count = 0
        for i in range(len(ds_zh)):
            count += get_token_count(ds_zh[i])
            if count >= zh_token_nums:
                break
        print(f"Select {i + 1} samples from {len(ds_zh)} samples in zh dataset - {(i + 1) / len(ds_zh) * 100:.2f}%")
        print(f"Total token nums: {count}, Avg token nums: {count / (i + 1):.2f}")
        ds_zh = ds_zh.select(range(i + 1)).select_columns(["instruction", "input", "output"])
        ds = concatenate_datasets([ds_en, ds_zh]).shuffle(seed=123)
        ds.to_json(output_file, force_ascii=False)
    else:
        ds_en.to_json(output_file, force_ascii=False)
