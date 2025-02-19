#!/usr/bin/env python3

import os
import csv
import json
import argparse
from google.generativeai import GenerativeModel, configure
import google.ai.generativelanguage as glm
from google.generativeai.discuss import ChatSession
import functools

# -------------------------
# Helper Functions
# -------------------------

def load_transcripts(folder_path):
    """
    Searches for supported subtitle files in the given folder and loads their contents.
    Looks for .vtt, .srt, .ssa, .stl, .ass, returning the first found.
    """
    supported_extensions = (".vtt", ".srt", ".ssa", ".stl", ".ass")
    file_path = None

    for file in os.listdir(folder_path):
        if file.endswith(supported_extensions):
            file_path = os.path.join(folder_path, file)
            break  # just load the first matching file

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(
            f"No subtitle file found in {folder_path} with extensions {supported_extensions}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt(prompt_file):
    """
    Loads the text from the given prompt file.
    This becomes the base prompt.
    """
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def save_response(folder_path, response_json):
    """
    Appends the response to a JSON file, keyed by the folder name.
    By default, it writes to a file named `transcripts_output.json` in the current directory.
    """
    output_file = "transcripts_output.json"

    # Load existing data if the JSON file exists, otherwise start empty.
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # Use folder name as the key in the JSON structure.
    folder_name = os.path.basename(os.path.normpath(folder_path))

    # If this folder doesn't have an entry yet, create one as a list.
    if folder_name not in data:
        data[folder_name] = {}

    # Append the new response to the folder's entry.
    data[folder_name] = response_json  # Overwrite, not append

    # Write the updated data back to the output JSON.
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

@functools.lru_cache(maxsize=128)  # Cache chat sessions
def create_chat_session(gemini_model, system_prompt_text, user_prompt_text):
    """Creates and caches a Gemini chat session with system and user prompts."""
    chat = gemini_model.start_chat(
        system_instruction=system_prompt_text
    )

    #Prime with the user prompt for the base context
    chat.send_message(user_prompt_text)  #Initializes chat object.

    return chat



def get_gemini_data(chat, request_prompt):
    """Sends request to Gemini and returns the response."""
    try:
        response = chat.send_message(request_prompt)
        content = response.text
        return content
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

# -------------------------
# Core Function
# -------------------------

def process_transcripts(folder_path, prompt_file, api_key):
    """
    Processes transcripts in `folder_path` using Gemini:
      - Uses a base prompt from prompt_file
      - The transcript is sent as the first user message (to establish context)
      - Makes follow-up requests for specific JSON fields
      - Saves the assembled JSON result
    """

    # 1. Configure the Gemini API
    configure(api_key=api_key)
    gemini_model = GenerativeModel("gemini-1.5-pro-latest")  # Adjust model name

    # 2. Load base prompt & transcript
    system_prompt_text = load_prompt(prompt_file)
    user_prompt_text = load_transcripts(folder_path)

    # 3. Create and cache chat session
    chat = create_chat_session(gemini_model, system_prompt_text, user_prompt_text)  #prime the chat object here

    # 4.  Request each JSON field piece by piece
    # Example request prompts: adapt to match your prompt file's structure
    request_prompts = {
        "metadata": "Extract the metadata JSON.",
        "transcript_extracted_data": "Extract the transcript_extracted_data JSON.  Include transcript, entities, topics, relations, temporal_data, sentiment analysis, key phrases.",
        "summary_100_words": "Provide a 100 word summary of the video.",
        "summary_50_words": "Provide a 50 word summary of the video.",
        "summary_10_words": "Provide a 10 word summary of the video.",
        "qa_pairs": "Create question and answer pairs for the transcript. Provide the result in JSON format."
    }

    extracted_data = {}
    for field, request_prompt in request_prompts.items():
        print(f"Requesting {field}...")
        gemini_response = get_gemini_data(chat, request_prompt)

        if gemini_response:
            try:
                extracted_data[field] = json.loads(gemini_response) #Assumes all fields are JSON parseable
            except json.JSONDecodeError:
                extracted_data[field] = gemini_response #Stores the unparsed response (probably a summary.)
                print(f"Warning:  {field} was not valid JSON.")
                #print(f"Raw response for {field}:\n{gemini_response}")
        else:
            extracted_data[field] = None
            print(f"Warning: Gemini could not generate {field}")



    # 5. Save the assembled JSON
    save_response(folder_path, extracted_data)

    return extracted_data  # Return the extracted data, not raw content

# -------------------------
# CSV Processor
# -------------------------

def process_all_ase(root_folder, csv_file, prompt_file):
    """
    Reads a CSV (video_id, label), checks if label == "ase",
    and for each matching row calls process_transcripts(...) on <root_folder>/<video_id>.
    """

    # 1. Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    # 2. Read the CSV file
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            # Only proceed if we have at least two columns
            if len(row) < 2:
                continue

            video_id = row[0].strip()
            label = row[1].strip().lower()

            # 3. Only process if label == 'ase'
            if label == "ase":
                folder_path = os.path.join(root_folder, video_id)
                print(f"\nProcessing video folder: {folder_path} (label: {label})")

                try:
                    result = process_transcripts(folder_path, prompt_file, api_key)
                    #print(f"\nExtracted Data for {video_id}:\n{json.dumps(result, indent=4)}\n{'='*50}\n") #Print json
                except FileNotFoundError as fnf_err:
                    print(f"Skipping {video_id}: {fnf_err}")
                except Exception as e:
                    print(f"Error processing {video_id}: {e}")
                    continue

# -------------------------
# CLI Entry Point
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Process transcripts for CSV rows where second column is 'ase'.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder containing subfolders named by video IDs (each containing transcripts)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file containing rows of the form: video_id,label"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Path to the base prompt text file."
    )

    args = parser.parse_args()
    process_all_ase(args.root, args.csv, args.prompt)


if __name__ == "__main__":
    main()