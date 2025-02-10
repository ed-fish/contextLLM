import os
import json
import subprocess
import argparse

def get_video_metadata_ffprobe(video_path):
    """
    Runs ffprobe on the given video_path and returns a dictionary with:
      - fps
      - width
      - height
      - duration_seconds
      - nb_frames (if available)
    
    Returns None if the video doesn't exist or if ffprobe fails.
    """
    if not os.path.exists(video_path):
        return None
    
    cmd = [
        "ffprobe", 
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams", 
        "-show_format",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        fps = None
        width = None
        height = None
        duration_seconds = None
        nb_frames = None
        
        # Search among streams for a video stream
        if "streams" in probe_data:
            for stream in probe_data["streams"]:
                if stream.get("codec_type") == "video":
                    # Attempt to parse FPS from "r_frame_rate" or "avg_frame_rate"
                    r_frame_rate = stream.get("r_frame_rate", "0/0")
                    if r_frame_rate != "0/0":
                        num, den = r_frame_rate.split('/')
                        if float(den) != 0:
                            fps = float(num) / float(den)
                    
                    width = stream.get("width", None)
                    height = stream.get("height", None)
                    
                    raw_nb_frames = stream.get("nb_frames", None)
                    if raw_nb_frames and raw_nb_frames.isdigit():
                        nb_frames = int(raw_nb_frames)
        
        # Look in "format" for duration
        if "format" in probe_data:
            raw_duration = probe_data["format"].get("duration", None)
            if raw_duration:
                try:
                    duration_seconds = float(raw_duration)
                except ValueError:
                    duration_seconds = None
        
        metadata = {
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration_seconds,
            "nb_frames": nb_frames
        }
        return metadata
    except subprocess.CalledProcessError as e:
        print(f"ffprobe failed for {video_path}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error for {video_path}: {e}")
        return None

def time_to_seconds(time_str):
    """
    Convert a timestamp string 'HH:MM:SS.xxx' into seconds as a float.
    """
    if time_str.count(':') == 1:
        time_str = "00:" + time_str

    try:
        h, m, s = time_str.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception as e:
        print(f"Error converting time '{time_str}' to seconds: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(
        description="Update JSON with ffprobe metadata (fps, resolution, etc.) and add start/end frames for each segment."
    )
    parser.add_argument("--input_json", required=True, help="Path to original JSON file.")
    parser.add_argument("--video_dir", required=True, help="Directory with <video_id>/<video_id>.mp4 structure.")
    parser.add_argument("--output_json", required=True, help="Path to write updated JSON.")
    args = parser.parse_args()
    
    # 1) Load the existing JSON
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    updated_count = 0
    total_count = len(data)
    
    # 2) For each video, gather metadata & compute frame ranges for segments
    for video_id, meta in data.items():

        if not isinstance(meta, dict):
            print(f"Skipping {video_id} because meta is type {type(meta)}")
            continue
        # Build the expected video file path
        video_file = os.path.join(args.video_dir, video_id, f"{video_id}.mp4")
        
        video_metadata = get_video_metadata_ffprobe(video_file)
        
        if video_metadata is not None:
            try:
                data[video_id]["video_metadata"] = video_metadata
            except:
                print(video_metadata)
            fps = video_metadata.get("fps", None)
            
            # If FPS is valid, convert each segment's time to frame indices
            if fps and fps > 0:
                # Check if transcript_extracted_data is a list or dict
                segments = meta.get("transcript_extracted_data", None)
                if isinstance(segments, dict):
                    segments = [segments]
                elif not isinstance(segments, list):
                    segments = []
                
                # For each segment, add "start_frame" and "end_frame"
                for seg in segments:
                    start_time_str = seg.get("start_time", "00:00:00.000")
                    end_time_str   = seg.get("end_time", "00:00:00.000")
                    start_sec = time_to_seconds(start_time_str)
                    end_sec   = time_to_seconds(end_time_str)
                    
                    # Round or floor? Commonly int(round(...)) or just int(...)
                    start_frame = int(round(start_sec * fps))
                    end_frame   = int(round(end_sec   * fps))
                    
                    seg["start_frame"] = start_frame
                    seg["end_frame"]   = end_frame
                
                # If transcript_extracted_data was a dict, we replaced it with a list
                # so we reassign properly if needed
                if isinstance(meta.get("transcript_extracted_data"), dict):
                    # Only one segment
                    data[video_id]["transcript_extracted_data"] = segments[0] if segments else {}
                else:
                    data[video_id]["transcript_extracted_data"] = segments
            
            updated_count += 1
        else:
            # ffprobe failed or file not found
            data[video_id]["video_metadata"] = None
    
    # 3) Write the updated JSON
    with open(args.output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Metadata updated for {updated_count}/{total_count} videos.")
    print(f"Result saved to {args.output_json}.")

if __name__ == "__main__":
    main()
