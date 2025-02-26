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
import numpy as np
from transformers import MBartTokenizer

SI_IDX, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3, 4

def merge_json_files(json_paths):
    """
    If you want to optionally merge multiple JSONs for training, you could do so here.
    Not currently used, but left for reference.
    """
    merged_data = {}
    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"[WARN] {json_path} not found, skipping.")
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
            for k,v in data.items():
                if k not in merged_data:
                    merged_data[k] = v
                else:
                    print(f"[WARN] Duplicate key '{k}' encountered. Keeping first occurrence.")
    return merged_data

class SignSegmentS2TDataset(Dataset):
    """
    Loads a final JSON with:
      - "output_file": path to .mp4
      - "caption": the spoken text to tokenize
      - "gloss": entire pseudo-gloss string
      - "topics": list of topic strings (only if keywords_proj = True)
    Also loads:
      - processed_words.pkl for gloss tokens
      - processed_topics.pkl for topic tokens, only if keywords_proj = True

    Returns (name_sample, frames_tensor, text_str, pg_id, topic_ids OR None).
    """

    def __init__(
        self,
        json_path: str,
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

        # 1) Load the JSON
        with open(json_path, "rb") as f:
            self.raw_data = json.load(f)
        self.keys = list(self.raw_data.keys())[:500]

        # 2) Load the pseudo-gloss dictionary (like "processed_words.pkl")
        emb_pkl = config["data"].get("vocab_file", "")
        if not os.path.exists(emb_pkl):
            raise FileNotFoundError(f"Pseudo-gloss pickle not found => {emb_pkl}")

        with open(emb_pkl, "rb") as pf:
            self.dict_processed_words = pickle.load(pf)
        # e.g. {
        #    "dict_sentence": { "ACANTHUS LEAF ..." => [12, 49, ...] },
        #    "dict_lem_to_id": { "ACANTHUS":12, "LEAF":49, ... }
        # }

        # 2b) If 'keywords_proj' is True, load topic dictionary
        self.use_topics = bool(self.config["model"].get("keywords_proj", False))
        if self.use_topics:
            topic_pkl = self.config["data"].get("topic_file", "")
            if not os.path.exists(topic_pkl):
                raise FileNotFoundError(f"Topic pickle not found => {topic_pkl}")
            with open(topic_pkl, "rb") as pf:
                self.dict_processed_topics = pickle.load(pf)
            # e.g. {
            #    "dict_lem_counter": {...},
            #    "dict_sentence": { (topicA, topicB,...): [topicA, topicB,...] },
            #    "dict_lem_to_id": { "Health":0, "Dust Mask":1, ... }
            # }
        else:
            self.dict_processed_topics = None

        # 3) Data transform pipeline
        self.data_transform = T.Compose([
            T.Lambda(lambda x: x.float() / 255.0),
            T.Resize((self.input_size, self.input_size), antialias=True),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

        # 4) Max length frames
        self.max_length = self.config["data"].get("max_length", 300)
        # 5) Downsample rate
        self.k = self.config["data"].get("downsample_rate", 0.25)

    def get_downsampled_indices(self, num_frames: int, train: bool, k: float = 0.25) -> list:
        """
        Get indices of selected frames after downsampling.
        """
        num_clips = max(1, int(num_frames * k))
        clip_size = max(1, num_frames // num_clips)
        indices = []
        
        for i in range(num_clips):
            start_idx = i * clip_size
            if train:
                idx_ = np.random.randint(start_idx, start_idx + clip_size)
            else:
                idx_ = start_idx
            idx_ = min(idx_, num_frames - 1)  # safety clamp
            indices.append(idx_)
        return indices

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        seg_info = self.raw_data[key]

        # A) name_sample
        name_sample = key

        # B) text_str
        text_str = seg_info.get("caption", "")

        # C) Pseudo-gloss ID list
        gloss_str = seg_info.get("gloss", "")
        tokens = self.dict_processed_words["dict_sentence"].get(gloss_str, [])
        pg_id = [self.dict_processed_words["dict_lem_to_id"].get(tok, -1) for tok in tokens]

        # D) If topics are enabled, parse them
        topic_ids = None
        if self.use_topics and "topics" in seg_info:
            topic_list = seg_info["topics"]
            # Map each topic to ID from dict_processed_topics
            topic_ids = [self.dict_processed_topics["dict_lem_to_id"].get(t, -1) for t in topic_list]

        # E) Video frames
        filename = seg_info.get("output_file", "")
        root_path = self.config["data"].get("root_dir", "")
        mp4_path = os.path.join(root_path, filename)

        if not os.path.exists(mp4_path):
            print(f"[WARN] Video path not found => {mp4_path}. Skipping sample.")
            return None

        try:
            vr = VideoReader(mp4_path, ctx=cpu(0))
            num_frames = len(vr)
            indices = self.get_downsampled_indices(num_frames, (self.phase=="train"), k=self.k)
            frames = vr.get_batch(indices).asnumpy()  # (T,H,W,3)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # => (T,3,H,W)
            frames = self.data_transform(frames)
            if frames.shape[0] > self.max_length:
                frames = frames[:self.max_length]
        except Exception as e:
            print(f"[ERROR] loading {mp4_path}: {e}")
            return None

        # Return everything
        return (name_sample, frames, text_str, pg_id, topic_ids)

def signsegment_s2t_collate_fn(batch, tokenizer, max_words=128):
    """
    Expects a list of (name_sample, frames_tensor, text_str, pg_id, topic_ids or None).
    We'll filter out any samples that failed => None.

    Returns => 
      src_input = {
        "input_ids": (B,T,3,H,W),
        "attention_mask": (B,T),
        "name_batch": [...],
        "src_length_batch": (B,)
      },
      tgt_input = { "input_ids", "attention_mask" }, # from text
      pgs_list => list of gloss ID lists
      topics_list => list of topic ID lists or None
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    name_batch = []
    frames_list = []
    text_list = []
    pgs_list = []
    topics_list = []

    if len(batch) == 0:
        return {}, {}, [], []

    # 1) Unpack
    for sample in batch:
        name_sample, frames_tensor, text_str, pg_id, topic_ids = sample
        name_batch.append(name_sample)
        frames_list.append(frames_tensor)
        text_list.append(text_str)
        pgs_list.append(pg_id)
        if topic_ids is None:
            topics_list.append([])
        else:
            topics_list.append(topic_ids)

    b = len(frames_list)
    # 2) Pad frames => (B, max_len, 3, H, W)
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

    # 4) build src_input
    src_input = {
        "input_ids": batch_frames,  # (B, T, 3, H, W)
        "attention_mask": mask,     # (B, T)
        "name_batch": name_batch,
        "src_length_batch": src_length_batch
    }

    return src_input, tgt_input, pgs_list, topics_list

class SignSegmentS2TDataModule(pl.LightningDataModule):
    """
    YTSL DataModule that yields => (src_input, tgt_input, pgs_list, topics_list).
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
        input_size=224,
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
            persistent_workers=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=lambda batch: signsegment_s2t_collate_fn(
                batch, self.tokenizer, self.max_words
            )
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
            collate_fn=lambda batch: signsegment_s2t_collate_fn(
                batch, self.tokenizer, self.max_words
            )
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignSegmentS2TDataModule test script.")
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--data_config", type=str, default="configs/config.yaml")
    parser.add_argument("--tokenizer_path", type=str, default="pretrain_models/MBart_trimmed", help="Path to MBart tokenizer.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_words", type=int, default=128)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--max_batches", type=int, default=1)
    args = parser.parse_args()

    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)

    # Create tokenizer
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

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    if not train_loader:
        print("[Main] No train loader found.")
    else:
        print(f"[Main] Checking train loader with up to {args.max_batches} batches.")
        for i, batch in enumerate(train_loader):
            src_input, tgt_input, pgs_list, topics_list = batch
            if not src_input:
                print(f"[Main] Skipped an empty train batch due to corrupted samples.")
                continue
            print(f"  [Train Batch {i}]")
            print("   src_input['input_ids'].shape:", src_input["input_ids"].shape)
            print("   src_input['attention_mask'].shape:", src_input["attention_mask"].shape)
            print("   pgs_list[0]:", pgs_list[0], " (some pseudo-gloss IDs)")
            print("   topics_list[0]:", topics_list[0], " (some topic IDs or empty if keywords_proj=False)")
            if i+1 >= args.max_batches:
                break

    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    if not test_loader:
        print("[Main] No test loader found.")
    else:
        print(f"[Main] Checking test loader with up to {args.max_batches} batches.")
        for i, batch in enumerate(test_loader):
            src_input, tgt_input, pgs_list, topics_list = batch
            if not src_input:
                print(f"[Main] Skipped an empty test batch.")
                continue
            print(f"  [Test Batch {i}]")
            print("   src_input['input_ids'].shape:", src_input["input_ids"].shape)
            print("   src_input['attention_mask'].shape:", src_input["attention_mask"].shape)
            print("   pgs_list[0]:", pgs_list[0])
            print("   topics_list[0]:", topics_list[0])
            if i+1 >= args.max_batches:
                break

    print("[Main] Done testing data module.")
