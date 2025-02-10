#!/usr/bin/env python3

import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

###############################################################################
# 1) DETECTION + BOUNDING BOX UTILS
###############################################################################

def detect_largest_person_per_frame(model, frame_np):
    """
    Runs YOLOv5 on a single frame (RGB, shape: HxWx3).
    Returns largest 'person' bounding box (x1,y1,x2,y2) or None if no person found.
    """
    img_pil = Image.fromarray(frame_np)
    results = model(img_pil, size=640)
    det = results.xyxy[0].cpu().numpy()

    largest_area = 0
    best_box = None
    for x1, y1, x2, y2, conf, cls_id in det:
        if int(cls_id) == 0:  # class 0 => 'person'
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if area > largest_area:
                largest_area = area
                best_box = (x1, y1, x2, y2)
    return best_box

def accumulate_union_box(union_box, new_box):
    """
    Merges 'new_box' into 'union_box' by taking the union bounding coords.
    Both boxes are (x1,y1,x2,y2). If union_box is None, return new_box directly.
    """
    if new_box is None:
        return union_box
    if union_box is None:
        return new_box
    (ux1, uy1, ux2, uy2) = union_box
    (nx1, ny1, nx2, ny2) = new_box
    return (
        min(ux1, nx1),
        min(uy1, ny1),
        max(ux2, nx2),
        max(uy2, ny2),
    )

def expand_box(box, margin_ratio, W, H):
    """
    Expands bounding box by 'margin_ratio' on each side, clamps to (W,H).
    Returns (nx1, ny1, nx2, ny2) or None if invalid.
    """
    if not box:
        return None
    (x1, y1, x2, y2) = box
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1 or bh < 1:
        return None
    
    mx = margin_ratio * bw
    my = margin_ratio * bh
    nx1 = max(0, int(x1 - mx))
    ny1 = max(0, int(y1 - my))
    nx2 = min(W, int(x2 + mx))
    ny2 = min(H, int(y2 + my))
    
    if nx1 >= nx2 or ny1 >= ny2:
        return None
    return (nx1, ny1, nx2, ny2)

###############################################################################
# 2) PER-SEGMENT FUNCTION (TOP-LEVEL) FOR PARALLEL EXECUTION
###############################################################################

def run_job(
    job,
    video_dir,
    output_dir,
    margin,
    detect_skip,
    model_conf,
    device
):
    """
    This is the top-level function that the ProcessPoolExecutor calls.
    It processes exactly one segment. 
    'job' is a tuple: (video_id, seg, seg_idx, use_fps, summary, prev_caption, next_caption).
    """
    (
        video_id,
        seg,
        seg_idx,
        use_fps,
        summary,
        prev_caption,
        next_caption
    ) = job
    
    # 2a) Load YOLO at the start (each worker has its own instance)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=device)
    model.conf = model_conf

    # 2b) Prepare paths
    video_file = os.path.join(video_dir, video_id, f"{video_id}.mp4")
    if not os.path.exists(video_file):
        return None

    start_frame = seg.get("start_frame", None)
    end_frame   = seg.get("end_frame", None)
    caption     = seg.get("caption", "")
    gloss       = seg.get("gloss", "")
    
    if start_frame is None or end_frame is None or start_frame >= end_frame:
        return None
    
    # 2c) Decode frames from decord
    try:
        vr = VideoReader(video_file, ctx=cpu(0))
    except:
        return None
    max_idx = min(end_frame, len(vr)-1)
    frame_indices = list(range(start_frame, max_idx+1))
    if not frame_indices:
        return None
    
    try:
        frames = vr.get_batch(frame_indices).asnumpy()  # (#frames,H,W,3)
    except:
        return None
    
    orig_h, orig_w, _ = frames[0].shape
    # downscale to width=640
    new_w = 640
    ratio = new_w / orig_w
    new_h = int(round(orig_h * ratio))
    
    # 2d) Gather union bounding box from sampled frames
    union_box = None
    num_frames = len(frames)
    
    detect_indices = list(range(0, num_frames, detect_skip))
    for i2 in detect_indices:
        f_np = frames[i2]
        # downscale
        small = cv2.resize(f_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        box = detect_largest_person_per_frame(model, small)
        if box is not None:
            union_box = accumulate_union_box(union_box, box)
    
    if union_box is None:
        # no person in sampled frames
        return None
    
    union_box = expand_box(union_box, margin, new_w, new_h)
    if union_box is None:
        return None
    
    (ux1, uy1, ux2, uy2) = union_box
    crop_w = ux2 - ux1
    crop_h = uy2 - uy1
    if crop_w < 2 or crop_h < 2:
        return None
    
    # 2e) Output MP4
    out_name = f"{video_id}_{seg_idx}.mp4"
    out_path = os.path.join(output_dir, out_name)
    
    # use original fps or fallback
    out_fps = use_fps if use_fps > 0 else 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (crop_w, crop_h))
    
    # encode all frames in the segment
    for f_np in frames:
        small = cv2.resize(f_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        crop  = small[uy1:uy2, ux1:ux2, :]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        writer.write(crop_bgr)
    
    writer.release()
    
    # 2f) Return final metadata (including summary + prev/next caption)
    return {
        "video_id": video_id,
        "segment_index": seg_idx,
        "output_file": out_path,
        "caption": caption,
        "gloss": gloss,
        "summary": summary,
        "prev_caption": prev_caption,
        "next_caption": next_caption,
        "width": crop_w,
        "height": crop_h,
        "fps": out_fps
    }

###############################################################################
# 3) MAIN LOGIC
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Parallel sign-language signer extraction with skip & downscale, plus summary & prev/next caption.")
    parser.add_argument("--json_file", required=True, help="JSON with video_metadata, start_frame, end_frame, etc.")
    parser.add_argument("--video_dir", required=True, help="Dir structure: <video_id>/<video_id>.mp4.")
    parser.add_argument("--output_dir", required=True, help="Where to store final .mp4.")
    parser.add_argument("--output_json", required=True, help="JSON to write with final segment metadata.")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin ratio for bounding box.")
    parser.add_argument("--skip", type=int, default=30, help="Detect bounding box every N frames.")
    parser.add_argument("--model_conf", type=float, default=0.3, help="YOLO confidence threshold.")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Write partial progress after this many jobs.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # We'll write partial results to partial_<output_json>
    partial_output_path = os.path.join(
        os.path.dirname(args.output_json),
        f"partial_{os.path.basename(args.output_json)}"
    )
    
    # 3a) Load JSON
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # 3b) Build jobs list
    jobs = []
    for video_id, vid_info in data.items():
        # ensure vid_info is a dict
        if not isinstance(vid_info, dict):
            print(f"Skipping {video_id}, not a dict.")
            continue
        
        vid_meta = vid_info.get("video_metadata", None)
        # skip if no metadata or not a dict
        if vid_meta is None or not isinstance(vid_meta, dict):
            print(f"Skipping {video_id}, invalid video_metadata.")
            continue
        
        # original fps
        use_fps = vid_meta.get("fps", 30.0)
        summary = vid_info.get("summary", "")  # store the summary for each video
        
        seg_data = vid_info.get("transcript_extracted_data", [])
        # unify to list
        if isinstance(seg_data, dict):
            seg_data = [seg_data]
        elif not isinstance(seg_data, list):
            continue
        
        for idx, seg in enumerate(seg_data):
            start_frame = seg.get("start_frame", None)
            end_frame   = seg.get("end_frame", None)
            if start_frame is None or end_frame is None or start_frame >= end_frame:
                continue
            
            # Determine prev/next caption
            prev_caption = ""
            next_caption = ""
            if idx - 1 >= 0:
                prev_caption = seg_data[idx - 1].get("caption", "")
            if idx + 1 < len(seg_data):
                next_caption = seg_data[idx + 1].get("caption", "")
            
            # job = (video_id, seg, seg_idx, use_fps, summary, prev, next)
            jobs.append((video_id, seg, idx, use_fps, summary, prev_caption, next_caption))
    
    print(f"Collected {len(jobs)} segments. Now processing in parallel...")
    
    results_dict = {}
    completed_count = 0
    CHECKPOINT_EVERY = args.checkpoint_interval
    
    def write_partial_results():
        """Write 'results_dict' to a partial file so we don't lose progress."""
        try:
            with open(partial_output_path, 'w') as pf:
                json.dump(results_dict, pf, indent=2)
            print(f"[Checkpoint] Wrote partial results for {len(results_dict)} segments to {partial_output_path}")
        except Exception as e:
            print(f"[Checkpoint Error] Could not write partial results: {e}")
    
    # 3c) Parallel
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_job = {}
        for j in jobs:
            future = executor.submit(
                run_job,
                j,
                args.video_dir,
                args.output_dir,
                args.margin,
                args.skip,
                args.model_conf,
                args.device
            )
            future_to_job[future] = j
        
        for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Processing"):
            job_data = future_to_job[future]
            (
                video_id,
                seg,
                seg_idx,
                use_fps,
                summary,
                prev_caption,
                next_caption
            ) = job_data
            key = f"{video_id}_{seg_idx}"
            try:
                res = future.result()
                if res:
                    results_dict[key] = res
            except Exception as e:
                print(f"Error in job {key}: {e}")
            completed_count += 1
            
            # Periodic checkpoint
            if completed_count % CHECKPOINT_EVERY == 0:
                write_partial_results()
    
    # 3d) Final write
    with open(args.output_json, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Done. Wrote {len(results_dict)} segments to {args.output_json}")
    # remove partial if it exists
    if os.path.exists(partial_output_path):
        os.remove(partial_output_path)
        print(f"Removed partial file {partial_output_path}")

if __name__ == "__main__":
    main()
