#!/usr/bin/env python3

import os
import csv
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu

# --------------------------------------------------------------
# YOLO detection utilities
# --------------------------------------------------------------

def detect_largest_person_per_frame(model, frame_rgb):
    """
    Run YOLOv5 on a single RGB frame (H x W x 3, NumPy).
    Returns (x1, y1, x2, y2) for the largest person, or None.
    """
    img_pil = Image.fromarray(frame_rgb)  # convert NumPy -> PIL
    results = model(img_pil, size=640)    # run inference
    det = results.xyxy[0].cpu().numpy()   # [N, 6] => x1, y1, x2, y2, conf, cls
    
    largest_area = 0
    best_box = None
    for x1, y1, x2, y2, conf, cls_id in det:
        if int(cls_id) == 0:  # 0 => 'person'
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if area > largest_area:
                largest_area = area
                best_box = (x1, y1, x2, y2)
    return best_box

def accumulate_union_box(union_box, new_box):
    """
    Union of bounding boxes: (x1,y1,x2,y2).
    If union_box is None, return new_box directly.
    """
    if new_box is None:
        return union_box
    if union_box is None:
        return new_box
    ux1, uy1, ux2, uy2 = union_box
    nx1, ny1, nx2, ny2 = new_box
    return (min(ux1, nx1),
            min(uy1, ny1),
            max(ux2, nx2),
            max(uy2, ny2))

def expand_box(box, margin_ratio, W, H):
    """
    Expands bounding box by 'margin_ratio' on each side; clamps to (W,H).
    box = (x1, y1, x2, y2). Returns expanded box or None if invalid.
    """
    if not box:
        return None
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bw < 2 or bh < 2:
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

# --------------------------------------------------------------
# Main script logic
# --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment & crop signers with YOLOv5 + decord, output letterboxed 640x640 videos without stretching."
    )
    parser.add_argument("--csv", required=True,
                        help="Path to tab-delimited CSV with VIDEO_NAME, SENTENCE_NAME, START_REALIGNED, END_REALIGNED, etc.")
    parser.add_argument("--video_folder", default=".",
                        help="Folder containing the source .mp4 videos (named <VIDEO_NAME>.mp4).")
    parser.add_argument("--output_folder", default="clips",
                        help="Folder to write the cropped clip segments.")
    
    # YOLO-related params
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Margin ratio around bounding box.")
    parser.add_argument("--skip", type=int, default=30,
                        help="Analyze bounding box every N frames for speed.")
    parser.add_argument("--model_conf", type=float, default=0.3,
                        help="YOLO confidence threshold.")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device: e.g., 'cuda:0' or 'cpu'.")
    
    # Final letterbox dimension
    parser.add_argument("--final_dim", type=int, default=640,
                        help="Output dimension for letterboxed video (e.g. 640 for 640x640).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 1) Load YOLOv5
    print(f"Loading YOLOv5 on device: {args.device}")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=args.device)
    model.conf = args.model_conf
    
    # 2) Read CSV
    rows = []
    with open(args.csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    
    print(f"Loaded {len(rows)} rows from {args.csv}")
    
    # 3) Process each row
    for row in tqdm(rows, desc="Processing rows"):
        video_name = row["VIDEO_NAME"]       # e.g. --7E2sU6zP4-5-rgb_front
        sentence_name = row["SENTENCE_NAME"] # e.g. --7E2sU6zP4_10-5-rgb_front
        start_sec = float(row["START_REALIGNED"])
        end_sec   = float(row["END_REALIGNED"])
        
        input_video = os.path.join(args.video_folder, f"{video_name}.mp4")
        if not os.path.isfile(input_video):
            print(f"Warning: video file not found: {input_video}")
            continue
        
        # Output path
        output_clip = os.path.join(args.output_folder, f"{sentence_name}.mp4")
        
        # 3a) Read video with decord
        try:
            vr = VideoReader(input_video, ctx=cpu(0))
        except Exception as e:
            print(f"Error loading video {input_video}: {e}")
            continue
        
        fps = vr.get_avg_fps()
        if not fps or fps < 1e-6:
            fps = 30.0
        
        start_frame = int(start_sec * fps)
        end_frame   = int(end_sec   * fps)
        
        if end_frame >= len(vr):
            end_frame = len(vr) - 1
        if start_frame < 0 or end_frame <= start_frame:
            continue
        
        frame_indices = range(start_frame, end_frame + 1)
        
        # 3b) Load frames
        try:
            frames = vr.get_batch(list(frame_indices)).asnumpy()  # (num_frames, H, W, 3)
        except Exception as e:
            print(f"Error reading frames for {sentence_name}: {e}")
            continue
        
        if len(frames) == 0:
            continue
        
        orig_h, orig_w, _ = frames[0].shape
        
        # We'll do YOLO detection on a downsized version => width=640
        # to speed up detection. Then map back to original size.
        detect_width = 640
        scale_ratio = detect_width / orig_w
        detect_height = int(round(orig_h * scale_ratio))
        
        union_box = None
        frame_count = len(frames)
        
        # skip frames for detection
        detect_indices = range(0, frame_count, args.skip)
        
        for di in detect_indices:
            frame_rgb = frames[di]
            small = cv2.resize(frame_rgb, (detect_width, detect_height), interpolation=cv2.INTER_AREA)
            box = detect_largest_person_per_frame(model, small)
            if box is not None:
                x1, y1, x2, y2 = box
                # upscale to original resolution
                x1 /= scale_ratio
                x2 /= scale_ratio
                y1 /= scale_ratio
                y2 /= scale_ratio
                union_box = accumulate_union_box(union_box, (x1, y1, x2, y2))
        
        if union_box is None:
            print(f"No person detected for segment {sentence_name}. Skipping.")
            continue
        
        # 3c) Expand bounding box
        expanded_box = expand_box(union_box, args.margin, orig_w, orig_h)
        if expanded_box is None:
            print(f"Expanded box invalid for segment {sentence_name}. Skipping.")
            continue
        
        x1, y1, x2, y2 = expanded_box
        bb_w = x2 - x1
        bb_h = y2 - y1
        
        if bb_w < 2 or bb_h < 2:
            print(f"Bounding box too small for {sentence_name}. Skipping.")
            continue
        
        # 3d) Prepare the writer => fixed dimension
        out_fps = fps
        out_dim = args.final_dim
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_clip, fourcc, out_fps, (out_dim, out_dim))
        
        # For each frame, we:
        #   1) Crop the bounding box region
        #   2) Resize it to fit into a 640x640 canvas preserving aspect ratio
        #   3) Letterbox or pillarbox in the center of a black 640x640 image
        for frame_rgb in frames:
            crop = frame_rgb[y1:y2, x1:x2, :]  # shape: (bb_h, bb_w, 3)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            
            # Maintain aspect ratio within out_dim x out_dim
            # step A: compute ratio
            ratio = min(out_dim / bb_w, out_dim / bb_h)
            new_w = int(bb_w * ratio)
            new_h = int(bb_h * ratio)
            
            # step B: resize the cropped region
            resized_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # step C: letterbox onto a black canvas of size out_dim x out_dim
            canvas = np.zeros((out_dim, out_dim, 3), dtype=np.uint8)
            # center the resized frame
            offset_x = (out_dim - new_w) // 2
            offset_y = (out_dim - new_h) // 2
            
            canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_bgr
            
            # step D: write the letterboxed frame
            writer.write(canvas)
        
        writer.release()
    
    print("All segments have been processed!")

if __name__ == "__main__":
    main()
