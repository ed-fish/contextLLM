import os
import json
import argparse
import concurrent.futures
import time
from google import genai  # Import google-generativeai, not google
import threading
import queue

# Global definitions for the dedicated file writer
file_update_queue = queue.Queue()
accumulated_data = {}
OUTPUT_FILE = "transcripts_output.json"


def file_writer_worker():
    """
    Dedicated file writer worker.
    Processes update messages from the file_update_queue.
    For each update (a tuple of folder_name and response JSON),
    it updates the in-memory dictionary and writes the combined data
    to the output file atomically.
    """
    while True:
        item = file_update_queue.get()
        if item is None:
            file_update_queue.task_done()
            break
        folder_name, response_json = item
        accumulated_data[folder_name] = response_json

        # Write to a temporary file and then atomically replace the output file.
        try:
            temp_file = OUTPUT_FILE + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(accumulated_data, f, indent=4, ensure_ascii=False)
            os.replace(temp_file, OUTPUT_FILE)
        except Exception as e:
            print(f"Error writing to JSON file: {e}")
        file_update_queue.task_done()


def save_response(folder_path, response_json):
    """
    Instead of immediately reading/writing the JSON file,
    this function enqueues the update (folder name and response JSON)
    to be processed by the dedicated file writer thread.
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    file_update_queue.put((folder_name, response_json))


def load_transcripts(folder_path):
    """Searches for .vtt files in the given folder and loads their transcripts."""
    transcript_text = None
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)  # Create full path
        if file.endswith((".vtt", ".srt", ".ssa", ".stl", ".ass")):  # Check multiple extensions
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    transcript_text = f.read()
                break  # Stop after first transcript file is found
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return None  # Return None if transcript couldn't be loaded.
    return transcript_text


def load_prompt(prompt_file):
    """Loads the text from the given prompt file."""
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found: {prompt_file}")
        return None  # Exit if prompt file is missing
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None


def process_transcripts(folder_path, prompt_file, api_key):
    """Processes transcripts using Google's Generative AI."""
    folder_name = os.path.basename(os.path.normpath(folder_path))
    print(f"Processing folder: {folder_name}")  # Helpful print statement

    transcript_text = load_transcripts(folder_path)
    if not transcript_text:  # Handle missing transcript
        print(f"Skipping {folder_name} due to missing/invalid transcript.")
        return None

    prompt_text = load_prompt(prompt_file)
    if not prompt_text:  # Handle missing prompt
        print(f"Skipping {folder_name} due to missing/invalid prompt file.")
        return None

    try:
        client = genai.Client(api_key=api_key)  # Initialize client inside the function

        # Generate response
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Folder id:{folder_name}\n\n{prompt_text}\n\nTranscript:\n{transcript_text}",
            config={'response_mime_type': 'application/json'}
        )

        try:
            response_json = json.loads(response.text)
            save_response(folder_path, response_json)
            print(f"Successfully processed {folder_name}")
            return response.text  # or response_json, depending on what you need
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for {folder_name}: {e}")
            print(f"Response text: {response.text}")  # Print the raw response
            return None

    except Exception as e:
        print(f"Error processing {folder_name}: {e}")
        return None


def main(base_folder, prompt_file, api_key, num_threads=15):
    """Main function to process all subfolders in parallel."""
    start_time = time.time()
    subfolders = [os.path.join(base_folder, d)
                  for d in os.listdir(base_folder)
                  if os.path.isdir(os.path.join(base_folder, d))]
    print(f"Found {len(subfolders)} subfolders.")

    # Start the dedicated file writer thread.
    file_writer_thread = threading.Thread(target=file_writer_worker)
    file_writer_thread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_transcripts, folder, prompt_file, api_key)
                   for folder in subfolders]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Get the result (or exception) from the thread
            except Exception as e:
                print(f"A thread raised an exception: {e}")

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")

    # Wait for all update messages to be processed.
    file_update_queue.join()
    # Signal the file writer thread to exit.
    file_update_queue.put(None)
    file_update_queue.join()
    file_writer_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process transcripts in subfolders using Google Generative AI.")
    parser.add_argument("base_folder", type=str, help="Path to the folder containing subfolders.")
    parser.add_argument("prompt", type=str, help="Path to the prompt .txt file.")
    parser.add_argument("api_key", type=str, help="Google Generative AI API Key.")
    parser.add_argument("--threads", type=int, default=15, help="Number of threads to use (default: 15)")
    args = parser.parse_args()

    main(args.base_folder, args.prompt, args.api_key, args.threads)
