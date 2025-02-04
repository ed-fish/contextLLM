import os
import google.generativeai as genai
import json

def load_transcripts(folder_path):
    """Searches for .vtt files in the given folder and loads their transcripts."""
    for file in os.listdir(folder_path):
        if file.endswith(".vtt"):
            file_path = os.path.join(folder_path, file)
        elif file.endswith(".srt"):
            file_path = os.path.join(folder_path, file)
        elif file.endswith(".ssa"):
            file_path = os.path.join(folder_path, file)
        elif file.endswith(".stl"):
            file_path = os.path.join(folder_path, file)
        elif file.endswith(".ass"):
            file_path = os.path.join(folder_path, file)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt(prompt_file):
    """Loads the text from the given prompt file."""
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()

def process_transcripts(folder_path, prompt_file, api_key):
    """Processes transcripts using Google's Generative AI."""
    # Configure the generative AI
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Load transcripts and prompt
    transcript_text = load_transcripts(folder_path)
    prompt_text = load_prompt(prompt_file)

    print(prompt_text)
    print(transcript_text)
    
    # Generate response
    response = model.generate_content(f"{prompt_text}\n\nTranscript:\n{transcript_text}")
    # Save the response
    try:
        response_json = json.loads(response.text)
        save_response(folder_path, response_json)
    except json.JSONDecodeError:
        print("Error: Response is not valid JSON.")
    
    return response.text

    
    return response.text

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process .vtt transcripts with Google Generative AI.")
    parser.add_argument("folder", type=str, help="Path to the folder containing .vtt files.")
    parser.add_argument("prompt", type=str, help="Path to the prompt .txt file.")
    parser.add_argument("api_key", type=str, help="Google Generative AI API Key.")
    
    args = parser.parse_args()
    
    output = process_transcripts(args.folder, args.prompt, args.api_key)
    print(output)

