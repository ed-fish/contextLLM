#!/usr/bin/env python3

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import MBartTokenizer
import yaml
from pathlib import Path
from datetime import datetime

# Your model:
from models.Pretrain_Model import PreTrainModel

# Utility for directory management (if still needed):
from models.utils import manage_directory

# The new decord-based DataModule 
# (Make sure dataset.ytsl_dataset contains the updated SignSegmentDataModule 
# that reads final JSON with "output_file" and uses decord.)
from dataset.ytsl_dataset import SignSegmentS2TDataModule

torch.set_float32_matmul_precision("medium")

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP', add_help=False)
    
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--num_gpus', default=1, type=int, metavar='N', help='number of gpus per node')
    parser.add_argument('--eval_freq', default=10, type=int, metavar='N', 
                        help='The frequency of metric evaluation, e.g. BLEU score')
    
    # Transformer & Encoder
    parser.add_argument('--mbart_path', type=str, default="/home/ef0036/Projects/contextLLM/pretrain_models/MBart_trimmed_yt_h2s",
                        help='Path to the MBart model.')
    parser.add_argument('--tokenizer_path', type=str, default="/home/ef0036/Projects/contextLLM/pretrain_models/MBart_trimmed_yt_h2s",
                        help='Path to the MBart tokenizer.')
    parser.add_argument('--encoder_ckpt', type=str, default=None, help='Path to the encoder checkpoint.')
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to the model checkpoint.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    
    # Data / JSON paths
    parser.add_argument('--train_json', type=str, default="/home/ef0036/Projects/contextLLM/data/combined_files/train_combine.json", help='Path to final JSON for train segments.')
    parser.add_argument('--val_json', type=str, default="/home/ef0036/Projects/contextLLM/data/how2sign/val/how2sign_val_path.json", help='Path to final JSON for val segments.')
    parser.add_argument('--test_json', type=str, default="/home/ef0036/Projects/contextLLM/data/how2sign/test/how2sign_test_path.json", help='Path to final JSON for test segments.')
    
    parser.add_argument('--data_config', type=str, default='configs/config.yaml',
                        help='Path to the data config file.')  
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
    parser.add_argument('--data_ver', type=int, default=0, help='Data version (if needed).')
    
    # Logging
    parser.add_argument('--logger', type=str, default='wandb', help='Logger type: wandb or tensorboard.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output_dir', type=str, default="output/pretrain", help='Output directory.')
    parser.add_argument('--log_dir', type=str, default="output/pretrain", help='Output directory.')
    parser.add_argument('--save_csv', type=str, default="csv_outputs/", help='Output directory for CSV logs.')

    parser.add_argument('--accumulate_batches', type=int, default=1, help='Accumulate batches num - 1 default')
    return parser

# W&B environment config
WANDB_CONFIG = {
    "WANDB_API_KEY": "b3a33bb694f0df2dfbd61a052b8e6d5aef47dba6",
    "WANDB_IGNORE_GLOBS": "*.patch",
    "WANDB_DISABLE_CODE": "true",
    "TOKENIZERS_PARALLELISM": "false"
}
def setupWandB(storage=None):
    os.environ.update(WANDB_CONFIG)
    if storage is not None:
        os.environ['WANDB_CACHE_DIR'] = storage+'/wandb/cache'
        os.environ['WANDB_CONFIG_DIR'] = storage+'/wandb/config'

def main(args):
    pl.seed_everything(args.seed)
    cudnn.benchmark = True

    # Load config
    with open(args.data_config, 'r') as file:
        config = yaml.safe_load(file)

    # Overwrite or fill missing arguments from the config if necessary:
    # E.g. if the config has 'data' => 'train_json', 'val_json', 'test_json', 
    # you can fill them if user hasn't set them:
    if 'data' in config:
        if 'train_json' in config['data'] and not args.train_json:
            args.train_json = config['data']['train_json']
        if 'val_json' in config['data'] and not args.val_json:
            args.val_json = config['data']['val_json']
        if 'test_json' in config['data'] and not args.test_json:
            args.test_json = config['data']['test_json']
    
    # Overwrite other fields from config if needed:
    if 'save' in config:
        args.output_dir = config['save'].get('output', args.output_dir)
        args.log_dir    = config['save'].get('output', args.log_dir)
        args.save_csv   = config['save'].get('csv', args.save_csv)
    if 'training' in config:
        args.model_ckpt = config['training'].get('ckpt_path', args.model_ckpt)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.logger == 'wandb':
        save_dir = f'{args.log_dir}/log_{current_time}'
        setupWandB(storage=save_dir)
        logger = WandbLogger(project="New-PSP", config=vars(args))
    else:
        logger = TensorBoardLogger(save_dir=f'{args.log_dir}/log_{current_time}', name="Sign2GPT")
    
    dirpath = f'{args.output_dir}/run_{current_time}'
    print("Current Time = {}".format(current_time))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_total_loss",
        mode="min",
        dirpath=dirpath,
        filename="best-{epoch:03d}-{val_total_loss:.3f}",
    )
    early_stop = EarlyStopping("val_loss", patience=args.epochs, mode="min", verbose=True)
    callbacks = [checkpoint_callback]
    # manage_directory(args.save_csv) # If you still need it

    # Create model
    model = PreTrainModel(
        config=args.data_config,
        lr=args.lr,
    )

    # Create tokenizer
    tokenizer = MBartTokenizer.from_pretrained(args.tokenizer_path)

    # Create the new decord-based DataModule
    # (Make sure your SignSegmentDataModule uses train_json, val_json, test_json 
    #  and returns DataLoaders with signsegment_collate_fn internally.)
    data_module = SignSegmentS2TDataModule(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        resize=(224,224),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
        data_config=config
    )

    trainer = pl.Trainer(
        logger=logger,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        min_epochs=1,
        max_epochs=args.epochs,
        precision=16,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_batches,
    )

    # Train
    trainer.fit(model, data_module)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")

    best_model = PreTrainModel.load_from_checkpoint(best_model_path)
    trainer.validate(best_model, data_module)
    trainer.test(best_model, data_module)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
