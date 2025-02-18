#!/usr/bin/env python3

"""
gloss_preprocessing.py

This script reads multiple JSON files with segments that have a "gloss" field, 
accumulates all the gloss tokens, builds a dictionary <token -> ID>, 
and saves it to a pickle for PSP usage.
"""

import os
import json
import pickle
import fasttext.util
import argparse
from collections import Counter
from pathlib import Path


def merge_json_files(json_files):
    """
    Load multiple JSON files and merge them into one dictionary.
    If keys overlap, it will raise a warning.
    """
    merged_data = {}

    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"⚠ Warning: {json_file} not found. Skipping.")
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, value in data.items():
                    if key in merged_data:
                        print(f"⚠ Warning: Duplicate key found: {key}. Keeping the first occurrence.")
                    else:
                        merged_data[key] = value
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")

    return merged_data


def main(args):
    # 1. Load multiple JSON files and merge
    data = merge_json_files(args.json_files)

    # 2. Gather all tokens from the "gloss" field
    all_tokens = []
    dict_sentence = {}  # Map entire gloss string -> list of tokens

    for key, seg_info in data.items():
        gloss_str = seg_info.get("gloss", "").strip()
        tokens = gloss_str.split() if gloss_str else []
        print(tokens)
        dict_sentence[gloss_str] = tokens
        all_tokens.extend(tokens)

    # 3. Count occurrences
    token_counter = Counter(all_tokens)

    # 4. Build token->ID dictionary
    unique_tokens = sorted(list(set(all_tokens)))
    dict_gloss_to_id = {token: idx for idx, token in enumerate(unique_tokens)}

    print(f"[INFO] Found {len(unique_tokens)} unique gloss tokens among {len(data)} segments.")
    print(f"[INFO] Downloading/loading fastText model for language '{args.lang}'...")

    # 5. fastText model (optional; if your PSP head only needs dict, skip embedding)
    fasttext.util.download_model(args.lang, if_exists="ignore")  # only if not present
    ft = fasttext.load_model(f"cc.{args.lang}.300.bin")

    # 6. Build final dictionary for pickle
    dict_processed_words = {
        "dict_lem_counter": dict(token_counter),   
        "dict_sentence": dict_sentence,            
        "dict_lem_to_id": dict_gloss_to_id         
    }

    # Ensure parent directory exists
    output_dir = os.path.dirname(args.output_pkl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 7. Save to pickle
    with open(args.output_pkl, 'wb') as pf:
        pickle.dump(dict_processed_words, pf)

    print(f"[DONE] Wrote {len(unique_tokens)} tokens to '{args.output_pkl}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess multiple JSON gloss data -> processed_words.pkl for PSP.")
    parser.add_argument("--json_files", nargs="+", required=True, help="List of JSON files with 'gloss' fields.")
    parser.add_argument("--output_pkl", default="data/processed_words.pkl", help="Output pickle path.")
    parser.add_argument("--lang", default="en", help="FastText language code (e.g. en, de).")
    args = parser.parse_args()

    main(args)
