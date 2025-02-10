import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

class SignSegmentDataset(Dataset):
    """
    A simple Dataset that:
      - Expects a JSON file with keys like "7Yq3PG6vhWM_0" 
        and fields including "output_file".
      - Uses decord to load each `.mp4` segment from 'output_file'.
      - Applies a basic transform (e.g. resize).
      - Returns frames + metadata.
    """

    def __init__(self, json_path, resize=(224, 224), phase='train'):
        """
        Args:
          json_path: path to your final JSON with each segment's metadata.
          resize: (width, height) to resize frames.
          phase: for optional augmentations or train/test split logic.
        """
        super().__init__()
        self.phase = phase
        self.resize = resize

        # 1) Load the JSON
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # We'll store keys (e.g. "7Yq3PG6vhWM_0") in a list for indexing
        self.keys = list(self.data.keys())
        
        # 2) Example transform pipeline (you can expand with augmentations)
        # We'll do a basic ToTensor + resize + normalization
        self.transform = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            # Example normalization for ImageNet
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Returns:
          frames_tensor:  shape = (num_frames, 3, H, W) [float tensor]
          metadata: dict with keys like "video_id", "segment_index", "caption", ...
        """
        key = self.keys[idx]
        seg_info = self.data[key]  # the dictionary of metadata
        
        # 1) Retrieve output_file for the cropped mp4
        mp4_path = seg_info.get("output_file", None)
        if not mp4_path or not os.path.exists(mp4_path):
            # If missing or invalid, return empty or raise an error
            raise FileNotFoundError(f"No valid .mp4 for segment key={key} => {mp4_path}")
        
        # 2) Use decord to read frames
        vr = VideoReader(mp4_path, ctx=cpu(0))
        num_frames = len(vr)

        frames_list = []
        for i in range(num_frames):
            frame = vr[i].asnumpy()  # shape (H, W, 3) in RGB
            # Convert to PIL so we can use torchvision transforms
            pil_frame = Image.fromarray(frame)  
            # Apply your transform
            frame_tensor = self.transform(pil_frame)  # shape (3, H, W)
            frames_list.append(frame_tensor)
        
        # shape => (num_frames, 3, H, W)
        if len(frames_list) == 0:
            # If no frames found for some reason
            frames_tensor = torch.empty(0, 3, self.resize[1], self.resize[0])
        else:
            frames_tensor = torch.stack(frames_list, dim=0)
        
        # 3) Create a metadata dict (or directly embed in the returned object)
        metadata = {
            "key": key,
            "video_id": seg_info.get("video_id", ""),
            "segment_index": seg_info.get("segment_index", -1),
            "caption": seg_info.get("caption", ""),
            "gloss": seg_info.get("gloss", ""),
            "summary": seg_info.get("summary", ""),
            "prev_caption": seg_info.get("prev_caption", ""),
            "next_caption": seg_info.get("next_caption", ""),
            "fps": seg_info.get("fps", 30.0),
            # add more fields if needed
        }
        
        return frames_tensor, metadata

def collate_fn(batch):
    """
    A simple collate_fn that:
    - puts frames of different lengths into a list
    - returns metadata as a list
    """
    # batch is list of (frames_tensor, metadata)
    frames_list = []
    metas_list = []
    
    for frames, meta in batch:
        frames_list.append(frames)  # shape => (F,3,H,W) each can differ in F
        metas_list.append(meta)
    
    # If you want to pad frames to the same length:
    # find max length
    max_len = max(f.shape[0] for f in frames_list)
    # We'll create a batch_frames of shape => (B, max_len, 3, H, W)
    # and a mask
    b = len(frames_list)
    if b == 0:
        return torch.empty(0), [], torch.empty(0)
    
    c, h, w = frames_list[0].shape[1:]  # 3,H,W
    batch_frames = torch.zeros(b, max_len, c, h, w)
    mask = torch.zeros(b, max_len, dtype=torch.bool)
    
    for i, f in enumerate(frames_list):
        length = f.shape[0]
        batch_frames[i, :length] = f
        mask[i, :length] = True
    
    # Return (frames, mask, metas)
    return batch_frames, mask, metas_list

# ----------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_decord.py <json_file> <output_dir>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    # out_dir = sys.argv[2]

    dataset = SignSegmentDataset(
        json_path=json_path,
        resize=(224,224),
        phase='train'
    )

    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    for batch_frames, mask, metas in loader:
        print("[Batch] frames shape:", batch_frames.shape)  # => (B, T, C, H, W)
        print("mask shape:", mask.shape)
        print("metadata example:", metas[0])
        print("metadata example:", metas[1])
        # do your training step
        break
