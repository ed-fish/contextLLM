import torch
from collections import OrderedDict
from pytorch_lightning.callbacks import Callback
import os

def manage_directory(path):
    # Check if the directory exists
    if not os.path.exists(path):
        # Create the directory if it doesn't exist
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        # Remove all .csv files in the directory
        for filename in os.listdir(path):
            if filename.endswith(".csv"):  # Only target .csv files
                file_path = os.path.join(path, filename)
                try:
                    os.unlink(file_path)  # Remove the .csv file
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        print(f"All .csv files in the directory '{path}' have been removed.")


def extract_layers_by_prefix(checkpoint_path, prefixes):
    """
    Extract layers that start with specified prefixes from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        prefixes (list): List of prefixes to match (e.g., ['feature_1', 'feature_2'])
    Returns:
        dict: Dictionary containing matched layer names and their shapes
        dict: Filtered state dictionary containing only matched layers
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Find matching layers
    matched_layers = {}
    filtered_state_dict = OrderedDict()
    
    for layer_name, tensor in state_dict.items():
        # Check if layer name starts with any of the prefixes
        if any(layer_name.startswith(prefix) for prefix in prefixes):
            matched_layers[layer_name] = tensor.shape
            filtered_state_dict[layer_name] = tensor
            
    # # Print matched layers
    # print("\n=== Matched Layers ===")
    # for name, shape in matched_layers.items():
    #     print(f"Layer: {name}")
    #     print(f"Shape: {shape}")
    #     print("-" * 50)
    # exit(0)
        
    return matched_layers, filtered_state_dict
import torch.nn.functional as F
class KLLoss(torch.nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=torch.nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        # print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

class SaveBestModelOnNEpochs(Callback):
    def __init__(self, save_every_n_epochs, monitor, mode, dirpath, filename_template="best-epoch={epoch:03d}-val_loss={val_loss:.3f}-val_bleu={val_bleu:.3f}.ckpt"):
        """
        Args:
            save_every_n_epochs (int): Save every N epochs.
            monitor (str): Metric to monitor.
            mode (str): One of {"min", "max"}.
            dirpath (str): Directory to save checkpoints.
            filename_template (str): Template for checkpoint filename. Use `{epoch}`, `{val_loss}`, `{val_bleu}` for naming.
        """
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.dirpath = dirpath
        self.filename_template = filename_template
        self.best_score = None
        self.last_best_checkpoint_path = None  # Track the last best checkpoint

        # Ensure the directory exists
        os.makedirs(self.dirpath, exist_ok=True)

        if self.mode not in ["min", "max"]:
            raise ValueError("Mode should be either 'min' or 'max'")

        if self.mode == "min":
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def on_validation_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch  # +1 to make epochs 1-based

        # Check if we are at the right epoch to save a checkpoint
        if (current_epoch+1) % self.save_every_n_epochs == 0 and current_epoch != 0:
            current_score = trainer.callback_metrics.get(self.monitor)
            if current_score is None:
                print(f"Warning: {self.monitor} metric is not available. Skipping checkpoint saving.")
                return

            # Compare current score with the best score based on mode
            is_best = (self.mode == "min" and current_score < self.best_score) or \
                    (self.mode == "max" and current_score > self.best_score)

            if is_best:
                self.best_score = current_score

                # Generate the new checkpoint filename
                filename = self.filename_template.format(
                    epoch=current_epoch,
                    val_loss=trainer.callback_metrics.get("val_loss", 0.0),
                    val_bleu=current_score,
                )
                new_checkpoint_path = os.path.join(self.dirpath, filename)

                # Save the new best checkpoint
                trainer.save_checkpoint(new_checkpoint_path)
                print(f"New best checkpoint saved: {new_checkpoint_path}")

                # Remove the previous best checkpoint if it exists
                if self.last_best_checkpoint_path and os.path.exists(self.last_best_checkpoint_path):
                    os.remove(self.last_best_checkpoint_path)
                    print(f"Removed previous best checkpoint: {self.last_best_checkpoint_path}")

                # Update the last best checkpoint path
                self.last_best_checkpoint_path = new_checkpoint_path

def local_1d_pattern(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Generates a strictly centered local attention mask for a 1D sequence.
    
    Args:
        seq_len (int): Length of the sequence.
        window_size (int): Size of the attention window. Must be odd.
        
    Returns:
        mask (torch.Tensor): Binary mask of shape (seq_len, seq_len).
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    
    # Create a mask of zeros
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    
    # Calculate half-window size
    half_window = window_size // 2
    
    # Fill the mask with 1s within the centered window
    for i in range(seq_len):
        # Calculate the start and end of the window
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        
        # Set the window to 1
        mask[i, start:end] = 1
    
    return mask


import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn


class PG_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # We use BCELoss with reduction='none', so we can sum manually
        # at the end. That helps when you want to accumulate gradients.
        self.bce_loss_fn = nn.BCELoss(reduction="none")

    def forward(self, src, tgt):
        """
        Args:
          src: shape (batch_size, vocab_size), in [0,1].
               e.g. probabilities after some row/col softmax or sigmoid.
          tgt: a list (length=batch_size) of lists-of-indices that should be 1.0 
               for each sample. For example, tgt[i] = [4, 9] => those positions are 1.0,
               others are 0.0 in the i-th row.

        Returns:
          A single scalar (float) summing the losses across the batch. 
          (Hence, it grows w/ batch size.)
        """
        # If using autocast for the rest of the model, disable it here to avoid "unsafe to autocast" in BCELoss
        with torch.cuda.amp.autocast(enabled=False):
            batch_size, vocab_size = src.shape
            # 1) Build the (batch_size, vocab_size) target
            gloss_targets = torch.zeros(batch_size, vocab_size, dtype=torch.float32, device=src.device)
            for i, token_list in enumerate(tgt):
                for token_id in token_list:
                    gloss_targets[i, token_id] = 1.0

            # 2) Clamp preds => avoid exact 0 or 1
            preds = torch.clamp(src.float(), 1e-8, 1.0 - 1e-8)
            
            # 3) BCELoss with "none"
            per_elem_loss = self.bce_loss_fn(preds, gloss_targets)  # shape => (batch_size, vocab_size)

            # 4) Sum over all elements => total scalar
            # If you're accumulating gradients, using sum means the total gradient scale 
            # grows with batch size or accumulation steps, which may be desired.
            return per_elem_loss.sum() 
        


class PG_FocalLossProb(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="sum"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, src, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            batch_size, vocab_size = src.shape
            gloss_targets = torch.zeros(batch_size, vocab_size, dtype=torch.float32, device=src.device)
            for i, token_list in enumerate(tgt):
                for token_id in token_list:
                    gloss_targets[i, token_id] = 1.0

            prob = torch.clamp(src.float(), 1e-8, 1 - 1e-8)
            bce = - (gloss_targets * torch.log(prob) + (1 - gloss_targets) * torch.log(1 - prob))
            p_t = prob * gloss_targets + (1.0 - prob) * (1.0 - gloss_targets)
            alpha_t = self.alpha * gloss_targets + (1.0 - self.alpha) * (1.0 - gloss_targets)
            focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)
            loss = focal_weight * bce

            if self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "mean":
                return loss.mean()
            else:
                return loss
