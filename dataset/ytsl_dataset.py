import os
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl
from decord import VideoReader, cpu
from PIL import Image
import argparse
import yaml
from transformers import MBartTokenizer
import numpy as np


# ----------------------------------------------------------------------------
# 1) DATASET
# ----------------------------------------------------------------------------

SI_IDX,PAD_IDX, UNK_IDX,BOS_IDX, EOS_IDX = 0 ,1 ,2 ,3 ,4
class SignSegmentS2TDataset(Dataset):
    """
    Loads a final JSON with:
    - "output_file": path to .mp4
    - "caption" or "text": the spoken text to tokenize
    - "gloss": entire pseudo-gloss string
    Also loads a processed_words.*.pkl to map each token in 'gloss' -> ID list.
    Returns (name_sample, frames_tensor, text_str, pg_ids).
    """

    def __init__(
        self,
        json_path,
        tokenizer,        # MBart or other HF tokenizer
        config: dict,     # your YAML config
        phase="train",
        max_words=128,
        resize=256,       # resized dimension
        input_size=224,   # final dimension
    ):
        super().__init__()
        self.phase = phase
        self.max_words = max_words
        self.resize = resize
        self.input_size = input_size
        self.tokenizer = tokenizer
        self.config = config

        # # 1) Load the JSON
        # if self.phase == "train":
        #     self.raw_data = merge_json_files(json_paths)
        # else:
        with open(json_path, 'rb') as f:
            self.raw_data = json.load(f)

        self.keys = list(self.raw_data.keys())


        # 2) Load the pseudo-gloss dictionary (like "data/processed_words.gloss_pkl")
        # emb_pkl = self.config["data"].get("pg_pickle", "data/processed_words.gloss_pkl")
        emb_pkl = config["data"].get("vocab_file", "")
        if not os.path.exists(emb_pkl):
            raise FileNotFoundError(f"Pseudo-gloss pickle not found => {emb_pkl}")

        with open(emb_pkl, "rb") as pf:
            self.dict_processed_words = pickle.load(pf)
        #  { "dict_sentence": { entireGlossStr => [tokens...] },
        #    "dict_lem_to_id": { token => tokenID }, ... }

        # 3) Basic transform pipeline
        #    If you want vidaug, random transforms, etc. you can insert them here
        #    For now, we do a simple "Resize => ToTensor => Normalize"

        self.data_transform = T.Compose([
            T.Lambda(lambda x: x.float() / 255.0),  # Replace ToTensor()
            T.Resize((self.input_size, self.input_size), 
                     antialias=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # self.data_transform = T.Compose([
        #     T.Resize((self.input_size, self.input_size), interpolation=T.InterpolationMode.BILINEAR),
        #     T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406],
        #                 [0.229, 0.224, 0.225]),
        # ])

        # 4) Max length for frames
        self.max_length = self.config["data"].get("max_length", 300)
        
        # 5) Set downsampling rate
        self.k = self.config["data"].get("downsample_rate", 0.25)
    
    def get_downsampled_indices(self, num_frames: int, train: bool, k: float = 0.25) -> list:
        """
        Get indices of selected frames after downsampling.
        
        Args:
            num_frames (int): Total number of frames in the original video.
            phase (str): "train" for random sampling, "inference" for first frame selection.
            k (float, optional): Downsampling rate. Defaults to 0.25.
        
        Returns:
            list: Indices of selected frames.
        """
        num_clips = int(num_frames * k)  # Total number of clips
        clip_size = num_frames // num_clips  # Frames per clip
        indices = []
        
        for i in range(num_clips):
            start_idx = i * clip_size
            if train:
                indices.append(np.random.randint(start_idx, start_idx + clip_size))
            else:
                indices.append(start_idx)
            
        return indices

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        seg_info = self.raw_data[key]

        # A) name_sample can just be the JSON key
        name_sample = key

        # B) text_str from 'caption' or some other field
        text_str = seg_info.get("caption", "")

        # C) Build pseudo-gloss ID list from seg_info["gloss"]
        gloss_str = seg_info.get("gloss", "")
        tokens = self.dict_processed_words["dict_sentence"].get(gloss_str, [])
        pg_id = []
        for tok in tokens:
            if tok in self.dict_processed_words["dict_lem_to_id"]:
                pg_id.append(self.dict_processed_words["dict_lem_to_id"][tok])
            else:
                pg_id.append(-1)  # unknown token

        # D) Use decord to load frames from .mp4
        filename = seg_info.get("output_file")
        root_path = self.config["data"].get("root_dir")
        mp4_path = os.path.join(root_path, filename)

        if not os.path.exists(mp4_path):
            print(f"Warning: Video path not found => {mp4_path}. Skipping sample.")
            return None


        try:
            vr = VideoReader(mp4_path, ctx=cpu(0))
            num_frames = len(vr)
            stride = self.config["data"].get("frame_stride", 10)
            # indices = list(range(0, num_frames, stride))
            indices = self.get_downsampled_indices(num_frames, self.phase == "train", k=self.k)
            
            # Batch decode frames
            frames = vr.get_batch(indices)  # decord.NDArray (T,H,W,3)
            frames = frames.asnumpy()  # numpy array
            frames = torch.from_numpy(frames)  # (T,H,W,3), uint8
            
            # Permute to (T,C,H,W) and normalize
            frames = frames.permute(0, 3, 1, 2)  # (T,3,H,W)
            frames = self.data_transform(frames)  # (T,3,input_size,input_size)
            
            # Truncate if needed
            if frames.shape[0] > self.max_length:
                frames = frames[:self.max_length]
                
        except Exception as e:
            print(f"Error loading {mp4_path}: {e}")
            return None

        return name_sample, frames, text_str, pg_id

def signsegment_s2t_collate_fn(batch, tokenizer, max_words=128):
    """
    Expects a list of (name_sample, frames_tensor, text_str, pg_id).
    We'll filter out any samples that failed (i.e. are None). Then, we pad frames => (B, T, 3, H, W), create a mask => (B, T).
    We'll tokenize 'text_str' => 'tgt_input'.
    We'll collect 'pg_id' into 'pgs'.
    Output => (src_input, tgt_input, pgs).
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    name_batch = []
    frames_list = []
    text_list = []
    pgs_list = []

    # If no valid samples remain, return empty outputs.
    if len(batch) == 0:
        return {}, {}, []

    # 1) Unpack the batch
    for (name_sample, frames_tensor, text_str, pg_id) in batch:
        name_batch.append(name_sample)
        frames_list.append(frames_tensor)
        text_list.append(text_str)
        pgs_list.append(pg_id)

    b = len(frames_list)
    # 2) Find max T for frames
    max_len = max(f.shape[0] for f in frames_list)
    c = 3
    h, w = frames_list[0].shape[-2], frames_list[0].shape[-1]
    batch_frames = torch.zeros(b, max_len, c, h, w)
    mask = torch.zeros(b, max_len, dtype=torch.long)
    src_length_batch = []

    for i in range(b):
        length = frames_list[i].shape[0]
        batch_frames[i, :length] = frames_list[i]
        mask[i, :length] = 1
        src_length_batch.append(length)

    src_length_batch = torch.tensor(src_length_batch, dtype=torch.long)

    # 3) Tokenize text => 'tgt_input'
    with tokenizer.as_target_tokenizer():
        tgt_input = tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            max_length=max_words,
            truncation=True
        )

    # 4) Build src_input
    src_input = {
        "input_ids": batch_frames,   # shape => (B, T, 3, H, W)
        "attention_mask": mask,        # shape => (B, T)
        "name_batch": name_batch,
        "src_length_batch": src_length_batch
    }

    return src_input, tgt_input, pgs_list

# ----------------------------------------------------------------------------
# 3) DATA MODULE
# ----------------------------------------------------------------------------

class SignSegmentS2TDataModule(pl.LightningDataModule):
    """
    A single DataModule that uses the above dataset + collate 
    to produce (src_input, tgt_input, pgs).
    """

    def __init__(
        self,
        train_json,
        val_json,
        test_json,
        tokenizer,
        data_config,
        batch_size=2,
        num_workers=2,
        max_words=128,
        resize=256,
        input_size=224
    ):
        super().__init__()
        self.train_json = train_json
        self.val_json = val_json
        self.test_json = test_json
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_words = max_words
        self.resize = resize
        self.input_size = input_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = SignSegmentS2TDataset(
                json_path=self.train_json,
                tokenizer=self.tokenizer,
                config=self.data_config,
                phase="train",
                max_words=self.max_words,
                resize=self.resize,
                input_size=self.input_size
            )
            self.val_dataset = SignSegmentS2TDataset(
                json_path=self.val_json,
                tokenizer=self.tokenizer,
                config=self.data_config,
                phase="dev",
                max_words=self.max_words,
                resize=self.resize,
                input_size=self.input_size
            )
        if stage in ("validate", None):
            self.val_dataset = SignSegmentS2TDataset(
                json_path=self.val_json,
                tokenizer=self.tokenizer,
                config=self.data_config,
                phase="dev",
                max_words=self.max_words,
                resize=self.resize,
                input_size=self.input_size
            )
        if stage in ("test", None):
            self.test_dataset = SignSegmentS2TDataset(
                json_path=self.test_json,
                tokenizer=self.tokenizer,
                config=self.data_config,
                phase="test",
                max_words=self.max_words,
                resize=self.resize,
                input_size=self.input_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: signsegment_s2t_collate_fn(
                batch, self.tokenizer, self.max_words
            )
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=lambda batch: signsegment_s2t_collate_fn(
                batch, self.tokenizer, self.max_words
            )
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: signsegment_s2t_collate_fn(
                batch, self.tokenizer, self.max_words
            )
        )

# ----------------------------------------------------------------------------
# 4) MAIN TEST
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignSegmentS2TDataModule test script.")
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--data_config", type=str, default="/home/ef0036/Projects/contextLLM/configs/config.yaml")
    parser.add_argument("--tokenizer_path", type=str, default="/home/ef0036/Projects/contextLLM/pretrain_models/MBart_trimmed", help="Path to MBart tokenizer.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_words", type=int, default=128)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--max_batches", type=int, default=1)
    args = parser.parse_args()

    # Load config
    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)

    # If no tokenizer path is given, fallback or error
    if not args.tokenizer_path:
        print("[Warning] No tokenizer_path provided, using a random MBart path.")
        args.tokenizer_path = "facebook/mbart-large-50"

    # Create the tokenizer
    tokenizer = MBartTokenizer.from_pretrained(args.tokenizer_path)

    # Create DataModule
    dm = SignSegmentS2TDataModule(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        tokenizer=tokenizer,
        data_config=data_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_words=args.max_words,
        resize=args.resize,
        input_size=args.input_size
    )

    # Setup
    dm.setup(stage="fit")

    # Quick test on train_dataloader
    train_loader = dm.train_dataloader()
    if not train_loader:
        print("[Main] No train loader found.")
    else:
        print(f"[Main] Checking train loader with up to {args.max_batches} batches.")
        for i, batch in enumerate(train_loader):
            src_input, tgt_input, pgs = batch
            if not src_input:
                print(f"[Main] Skipped an empty train batch due to corrupted samples.")
                continue
            print(f"  [Train Batch {i}]")
            print("   src_input['input_ids'].shape:", src_input["input_ids"].shape)
            print("   src_input['attention_mask'].shape:", src_input["attention_mask"].shape)
            print("   pgs[0]:", pgs[0], " (some pseudo-gloss IDs)")
            if i+1 >= args.max_batches:
                break

    # Quick test on val_dataloader
    val_loader = dm.val_dataloader()
    if not val_loader:
        print("[Main] No val loader found.")
    else:
        print(f"[Main] Checking val loader with up to {args.max_batches} batches.")
        for i, batch in enumerate(val_loader):
            src_input, tgt_input, pgs = batch
            if not src_input:
                print(f"[Main] Skipped an empty val batch due to corrupted samples.")
                continue
            print(f"  [Val Batch {i}]")
            print("   src_input['input_ids'].shape:", src_input["input_ids"].shape)
            print("   src_input['attention_mask'].shape:", src_input["attention_mask"].shape)
            print("   pgs[0]:", pgs[0])
            if i+1 >= args.max_batches:
                break

    # Quick test on test_dataloader
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    if not test_loader:
        print("[Main] No test loader found.")
    else:
        print(f"[Main] Checking test loader with up to {args.max_batches} batches.")
        for i, batch in enumerate(test_loader):
            src_input, tgt_input, pgs = batch
            if not src_input:
                print(f"[Main] Skipped an empty test batch due to corrupted samples.")
                continue
            print(f"  [Test Batch {i}]")
            print("   src_input['input_ids'].shape:", src_input["input_ids"].shape)
            print("   src_input['attention_mask'].shape:", src_input["attention_mask"].shape)
            print("   pgs[0]:", pgs[0])
            if i+1 >= args.max_batches:
                break
    print("[Main] Done testing data module.")