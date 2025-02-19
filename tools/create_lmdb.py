import os
import json
import lmdb
import pickle
import argparse
from decord import VideoReader, gpu, cpu
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def time_to_seconds(time_str):
    """
    Convert a timestamp string "HH:MM:SS.mmm" into seconds.
    """
    try:
        parts = time_str.split(':')
        if len(parts) != 3:
            raise ValueError("Invalid time format: " + time_str)
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    except Exception as e:
        print(f"Error converting time '{time_str}' to seconds: {e}")
        return 0.0

def process_single_video(video_meta, video_dir, use_gpu):
    """
    Process one video (to be run in a worker process) and return a list of (key, value)
    tuples representing processed segments.
    
    Each video file is assumed to be stored as:
      <video_dir>/<video_id>/<video_id>.mp4
    """
    video_id = video_meta.get("video_id")
    video_file = os.path.join(video_dir, video_id, f"{video_id}.mp4")
    if not os.path.exists(video_file):
        print(f"Video file {video_file} not found.")
        return []
    try:
        ctx = gpu(0) if use_gpu else cpu(0)
        vr = VideoReader(video_file, ctx=ctx)
        fps = vr.get_avg_fps()
    except Exception as e:
        print(f"Error reading video {video_file} or obtaining FPS: {e}")
        return []
    
    # Retrieve transcript data; if a dict is provided, wrap it in a list.
    segments = video_meta.get("transcript_extracted_data")
    if isinstance(segments, dict):
        segments = [segments]
    elif not isinstance(segments, list):
        segments = []
    
    outputs = []
    errors_count = 0
    max_errors = 5  # if more than 5 errors occur for this video, skip remaining segments
    for seg_idx, segment in enumerate(segments):
        try:
            start_time_str = segment.get("start_time", "00:00:00.000")
            end_time_str = segment.get("end_time", "00:00:00.000")
            start_sec = time_to_seconds(start_time_str)
            end_sec = time_to_seconds(end_time_str)
            start_idx = int(start_sec * fps)
            end_idx = int(end_sec * fps)
            num_frames = len(vr)
            if start_idx >= num_frames or start_idx > end_idx:
                continue
            if end_idx >= num_frames:
                end_idx = num_frames - 1
            indices = list(range(start_idx, end_idx + 1))
            # Attempt to extract frames; if decoding fails, skip this segment.
            frames = vr.get_batch(indices).asnumpy()
            
            sample = {
                "video_id": video_id,
                "segment_index": seg_idx,
                "frames": frames,
                "caption": segment.get("caption", ""),
                "gloss": segment.get("gloss", ""),
                "topics": video_meta.get("topics", []),
                "start_time": start_time_str,
                "end_time": end_time_str,
                "frame_rate": fps,
                "title": video_meta.get("title", ""),
                "channel_name": video_meta.get("channel_name", ""),
                "publish_date": video_meta.get("publish_date", ""),
                "duration": video_meta.get("duration", ""),
                "summary": video_meta.get("summary", "")
            }
            key = f"{video_id}_{seg_idx}".encode('ascii')
            value = pickle.dumps(sample)
            outputs.append((key, value))
        except Exception as e:
            errors_count += 1
            print(f"Error processing video {video_id}, segment {seg_idx}: {e}")
            if errors_count > max_errors:
                print(f"Too many errors processing video {video_id}, skipping remaining segments.")
                break
            continue
    return outputs

def create_lmdb_dataset_parallel(json_path, video_dir, lmdb_path, map_size=1<<40, use_gpu=True, max_workers=4, commit_every=500):
    """
    Create an LMDB dataset from the JSON metadata and video files using parallel processing.
    
    - Each video is processed in parallel (via ProcessPoolExecutor) to extract its segments.
    - The main process collects the processed segments and writes them into LMDB.
    
    Parameters:
      json_path   : Path to the JSON metadata file.
      video_dir   : Base directory containing video folders (<video_id>/<video_id>.mp4).
      lmdb_path   : Output path for the LMDB database.
      map_size    : Maximum LMDB size (default ~1TB).
      use_gpu     : Whether to use GPU decoding (if available).
      max_workers : Maximum number of worker processes.
      commit_every: Commit LMDB transaction after this many segments.
    """
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    total_segments = 0
    processed_videos = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, video_meta, video_dir, use_gpu): video_id 
                   for video_id, video_meta in metadata.items()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            vid = futures[future]
            try:
                segments = future.result()
                for key, value in segments:
                    txn.put(key, value)
                    total_segments += 1
                    if total_segments % commit_every == 0:
                        txn.commit()
                        txn = env.begin(write=True)
            except Exception as e:
                print(f"Error processing video {vid}: {e}")
            processed_videos += 1
    
    txn.commit()
    env.close()
    print(f"Finished processing {processed_videos} videos. Total segments stored: {total_segments}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LMDB dataset from video JSON metadata in parallel."
    )
    parser.add_argument("--json_file", type=str, required=True, help="Path to JSON metadata file.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video folders.")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Output LMDB database path.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker processes.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for decoding (if available)")
    parser.add_argument("--commit_every", type=int, default=500, help="Commit transaction after this many segments")
    args = parser.parse_args()
    
    create_lmdb_dataset_parallel(
        json_path=args.json_file,
        video_dir=args.video_dir,
        lmdb_path=args.lmdb_path,
        map_size=1<<40,
        use_gpu=args.use_gpu,
        max_workers=args.max_workers,
        commit_every=args.commit_every
    )
