#!/usr/bin/env python3

"""
gloss_preprocessing.py

This script reads a final JSON file with segments that have a "gloss" field, 
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

def main(args):
    # 1. Load JSON
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # 2. Gather all tokens from the "gloss" field
    all_tokens = []
    dict_sentence = {}  # Map entire gloss string -> list of tokens
    
    # Count how many total segments have gloss
    for key, seg_info in data.items():
        gloss_str = seg_info.get("gloss", "").strip()
        tokens = gloss_str.split() if gloss_str else []
        dict_sentence[gloss_str] = tokens
        all_tokens.extend(tokens)

    # 3. Count occurrences
    token_counter = Counter(all_tokens)

    # 4. Build token->ID dictionary
    unique_tokens = sorted(list(set(all_tokens)))
    print(len(unique_tokens))
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
    parser = argparse.ArgumentParser(description="Preprocess final JSON gloss data -> processed_words.pkl for PSP.")
    parser.add_argument("--json_file", required=True, help="Path to final JSON with 'gloss' fields.")
    parser.add_argument("--output_pkl", default="data/processed_words.pkl", help="Output pickle path.")
    parser.add_argument("--lang", default="en", help="FastText language code (e.g. en, de).")
    args = parser.parse_args()
    main(args)
