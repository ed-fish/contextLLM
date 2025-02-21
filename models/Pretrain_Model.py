import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import wandb
from collections import OrderedDict
from models.utils import KLLoss, PG_Loss, PG_FocalLossProb
from models.clip_models import SLRCLIP
import yaml


class PreTrainModel(pl.LightningModule):
    def __init__(self,
                 config="configs/config.yaml",
                 lr=3e-4):
        super().__init__()
        self.save_hyperparameters()

        # 1) Load config
        with open(config, 'r') as file:
            self.config = yaml.safe_load(file)

        # 2) Build SLRCLIP or other model
        from models.clip_models import SLRCLIP  # Make sure it's in your path
        self.model = SLRCLIP(self.config)

        # 3) LR and losses
        self.lr = lr
        self.loss_img = KLLoss()
        self.loss_txt = KLLoss()
        loss_function = self.config["model"].get("model", "pgloss")
        if loss_function == "pgloss":
            self.loss_pg  = PG_Loss()
        elif loss_function == "focal_loss":
            self.loss_pg  = PG_FocalLossProb()
        else:
            raise ValueError(f"Invalid loss function: {loss_function}")

        self.landa = 0.5  # scale for psp loss
        # 4) Check alpha scheduling or static
        self.use_decay = bool(self.config["training"].get("alpha_use_decay", False))
        self.use_topic_overlap = bool(self.config["training"].get("use_topic_overlap", False))

        # If user wants topic overlap, we'll allow alpha scheduling
        self.topic_alpha_start = float(self.config["training"].get("topic_alpha_start", 0.0))
        self.topic_alpha_end   = float(self.config["training"].get("topic_alpha_end",   0.0))
        self.num_epochs        = float(self.config["training"].get("max_epochs", 100))

        # If no decay, set alpha just once
        if not self.use_decay:
            self.model.topic_alpha = self.topic_alpha_start

    def on_train_epoch_start(self):
        """
        If alpha_use_decay = True, do linear schedule from topic_alpha_start to topic_alpha_end
        across self.num_epochs.
        """
        if self.use_decay:
            current_epoch = self.current_epoch
            fraction = min(1.0, float(current_epoch)/(self.num_epochs - 1e-9))
            alpha = self.topic_alpha_start + fraction*(self.topic_alpha_end - self.topic_alpha_start)
            self.model.topic_alpha = alpha
            self.log("topic_alpha", alpha, on_step=False, on_epoch=True, prog_bar=True)

        # Optionally log LR
        if len(self.trainer.optimizers) > 0:
            optimizer = self.trainer.optimizers[0]
            lr = optimizer.param_groups[0]['lr']
            self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def forward(self, batch):
        """
        This isn't typically used in Lightning, we place calls in training_step. 
        But you can keep or remove as you want.
        """
        return self.model(*batch)

    def training_step(self, input_batch, batch_idx):
        # input_batch => (src_input, tgt_input, pgs_list, topics_list)
        (src_input, tgt_input, pgs_list, topics_list) = input_batch

        # forward pass
        logits_per_image, logits_per_text, clip_ground_truth, psp_logits = \
            self.model(src_input, tgt_input, topics_list=topics_list)

        # compute CLIP-like loss
        loss_imgs  = self.loss_img(logits_per_image, clip_ground_truth)
        loss_texts = self.loss_txt(logits_per_text,  clip_ground_truth)
        train_clip_loss = (loss_imgs + loss_texts)/2.0

        # psp loss
        train_psp_loss = self.loss_pg(psp_logits, pgs_list)

        train_total_loss = train_clip_loss + self.landa * train_psp_loss

        self.log("train_clip_loss", train_clip_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_psp_loss",  train_psp_loss,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_total_loss",train_total_loss,on_step=False, on_epoch=True, prog_bar=True)

        return train_total_loss

    def validation_step(self, input_batch, batch_idx):
        (src_input, tgt_input, pgs_list, topics_list) = input_batch

        logits_per_image, logits_per_text, clip_ground_truth, psp_logits = \
            self.model(src_input, tgt_input, topics_list=topics_list)

        loss_imgs  = self.loss_img(logits_per_image, clip_ground_truth)
        loss_texts = self.loss_txt(logits_per_text,  clip_ground_truth)
        val_clip_loss = (loss_imgs + loss_texts)/2.0
        val_psp_loss  = self.loss_pg(psp_logits, pgs_list)
        val_total_loss= val_clip_loss + self.landa*val_psp_loss

        self.log("val_clip_loss",  val_clip_loss,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psp_loss",   val_psp_loss,   on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_total_loss", val_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_total_loss

    def test_step(self, input_batch, batch_idx):
        (src_input, tgt_input, pgs_list, topics_list) = input_batch

        logits_per_image, logits_per_text, clip_ground_truth, psp_logits = \
            self.model(src_input, tgt_input, topics_list=topics_list)

        loss_imgs  = self.loss_img(logits_per_image, clip_ground_truth)
        loss_texts = self.loss_txt(logits_per_text,  clip_ground_truth)
        test_clip_loss = (loss_imgs + loss_texts)/2.0
        test_psp_loss  = self.loss_pg(psp_logits, pgs_list)
        test_total_loss= test_clip_loss + self.landa*test_psp_loss

        self.log("test_clip_loss", test_clip_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_psp_loss",  test_psp_loss,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_total_loss",test_total_loss,on_step=False, on_epoch=True, prog_bar=True)
        return test_total_loss

    def configure_optimizers(self):
        # e.g. AdamW
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.trainer.estimated_stepping_batches
        sched = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=steps,
                pct_start=0.05,
                anneal_strategy='cos'
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [sched]