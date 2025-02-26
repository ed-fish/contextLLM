import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from sacrebleu.metrics import BLEU
import pandas as pd
import os
import wandb
from collections import OrderedDict
from models.utils import extract_layers_by_prefix
from models.clip_models import gloss_free_model
from pathlib import Path
import yaml
import math

SI_IDX, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3, 4

class FineTuneModel(pl.LightningModule):
    def __init__(self,
                 config="configs/config.yaml",
                 args=None,
                 eval_freq=10,
                 csv_dire=None):
        super().__init__()
        self.eval_freq = eval_freq
        self.csv_dire = csv_dire
        self.save_hyperparameters()

        ################# Load the Config file ####################
        with open(config, 'r') as file:
            self.config = yaml.safe_load(file)
        self.args = args

        ################ Set the Sign Encoder ####################
        self.model = gloss_free_model(self.config, self.args)
        print('***********************************')
        print('Load parameters from Pretrained...')
        print('***********************************')
        if args.model_ckpt:
            state_dict = torch.load(args.model_ckpt, map_location='cpu')['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'conv_2d' in k or 'conv_1d' in k:
                    k = 'backbone.' + '.'.join(k.split('.')[3:])
                    new_state_dict[k] = v
                if 'trans_encoder' in k:
                    k = 'mbart.base_model.model.model.encoder.' + '.'.join(k.split('.')[5:])
                    new_state_dict[k] = v

            ret = self.model.load_state_dict(new_state_dict, strict=False)
            print('Missing keys: \n', '\n'.join(ret.missing_keys))
            print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
        else:
            print("no model loaded random init")

        ################ Initialize the tokenizer ####################
        self.tokenizer = MBartTokenizer.from_pretrained(self.config['model']['tokenizer'],
                                                          src_lang='en_XX', tgt_lang='en_XX')
        # This token is used in generation; note the space before period in your original code.
        self.end_sym = ' .'
        self.max_txt_len = 64

        ################ Set the Optimizer ####################
        self.lr = self.args.lr
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.05)

        # These were used in your original CSV logging (for train); kept here if needed.
        self.train_decoded_teacher = []
        self.train_step_outputs = []

        # Instead of running expensive generation on every validation batch,
        # we accumulate source inputs, teacher-forced outputs, and references.
        self._val_src_inputs = []          # To run beam search later
        self._val_teacher_decoded = []       # Teacher forced predictions (via argmax)
        self._val_refs = []                # Ground-truth text

        # For test logging (unchanged)
        self.test_decoded = []
        self.test_step_outputs = []

        # Create a BLEU instance to reuse
        self.bleu_metric = BLEU()

    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):

        (src_input, tgt_input, _, _) = batch
        outputs, loss = self.model(src_input, tgt_input)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        (src_input, tgt_input, _, _) = batch
        outputs, loss = self.model(src_input, tgt_input)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Only accumulate data during eval epochs (and skip epoch 0)
        if (self.current_epoch + 1) % self.eval_freq == 0 and self.current_epoch != 0:
            # Store ground-truth references
            refs = self.tokenizer.batch_decode(tgt_input['input_ids'], skip_special_tokens=True)
            self._val_refs.extend(refs)
            # Compute teacher-forced predictions (via argmax) – inexpensive
            teacher_preds = self.teacher_forcing_generate(outputs)
            self._val_teacher_decoded.extend(teacher_preds)
            # Store the source input to later run beam search (which is expensive)
            self._val_src_inputs.append(src_input)
        return loss

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.eval_freq == 0 and self.current_epoch != 0:
            # Run beam search generation *once per stored batch*
            all_gen_outputs = []
            for src_in in self._val_src_inputs:
                with torch.no_grad():
                    out = self.model.generate(
                        src_in,
                        max_new_tokens=150,  # You can adjust this for speed vs. quality
                        num_beams=4,
                        decoder_start_token_id=self.tokenizer.lang_code_to_id['en_XX']
                    )
                gen_text = self.tokenizer.batch_decode(out, skip_special_tokens=True)
                all_gen_outputs.extend(gen_text)

            # Prepare CSV data for this rank
            new_data = {
                "hypotheses": all_gen_outputs,
                "hypotheses_teacher": self._val_teacher_decoded,
                "targets": self._val_refs
            }
            file_path = self.csv_dire + f"val_outputs_{self.current_epoch+1}_{self.trainer.global_rank}.csv"
            self.add_data_to_csv(file_path, new_data, columns=["hypotheses", "hypotheses_teacher", "targets"])

            # Now, read the CSV files from all ranks (DDP) to gather the full validation set
            all_hypotheses = []
            all_hypotheses_teacher = []
            all_refs = []
            for idx in range(self.trainer.world_size):
                rank_file = self.csv_dire + f"val_outputs_{self.current_epoch+1}_{idx}.csv"
                if os.path.exists(rank_file):
                    df = pd.read_csv(rank_file, sep='|')
                    all_hypotheses.extend([str(item) for item in df['hypotheses'].tolist()])
                    all_hypotheses_teacher.extend([str(item) for item in df['hypotheses_teacher'].tolist()])
                    all_refs.extend([str(item) for item in df['targets'].tolist()])

            # Log text examples to TensorBoard and/or Weights & Biases
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_text("hypotheses_teacher",
                                                "\n".join(all_hypotheses_teacher[:5]), self.current_epoch)
                self.logger.experiment.add_text("hypotheses",
                                                "\n".join(all_hypotheses[:5]), self.current_epoch)
                self.logger.experiment.add_text("targets",
                                                "\n".join(all_refs[:5]), self.current_epoch)
            if isinstance(self.logger, WandbLogger):
                self._log_to_wandb(all_refs[:5], all_hypotheses[:5], all_hypotheses_teacher[:5],
                                   split="val", epoch=self.current_epoch)

            print(len(all_refs), len(all_hypotheses))
            bleu = self.bleu_metric.corpus_score(all_hypotheses, [all_refs]).score
            self.log("val_bleu", bleu, prog_bar=True, sync_dist=True)
            print(bleu)
            print('*' * 50)
            print(len(all_refs), len(all_hypotheses_teacher))
            teacher_bleu = self.bleu_metric.corpus_score(all_hypotheses_teacher, [all_refs]).score
            print(teacher_bleu)
            self.log("val_teacher_bleu", teacher_bleu, prog_bar=True, sync_dist=True)

            # Clear stored validation data
            self._val_src_inputs.clear()
            self._val_teacher_decoded.clear()
            self._val_refs.clear()

    def _log_to_wandb(self, targets, hypotheses, hypotheses_teacher, split: str, epoch: int):
        """Log text examples to Weights & Biases."""
        columns = ["Target", "hypotheses", "hypotheses_teacher"]
        data = [
            [t, h, h_teacher]
            for t, h, h_teacher in zip(targets, hypotheses, hypotheses_teacher)
        ]
        table = wandb.Table(columns=columns, data=data)
        self.logger.experiment.log({f"{split}_outputs_{epoch}": table})

    def test_step(self, batch, batch_idx):
        src_input, tgt_input, _ = batch
        outputs, loss = self.model(src_input, tgt_input)
        self.log("test_loss", loss, sync_dist=True)

        self.test_decoded = self.generate(src_input)
        self.test_step_outputs = self.tokenizer.batch_decode(tgt_input['input_ids'], skip_special_tokens=True)
        tgt_refs = [str(item) + ' .' for item in self.test_step_outputs]
        hypotheses = [str(item) + ' .' for item in self.test_decoded]
        new_data = {"hypotheses": hypotheses, "targets": tgt_refs}
        file_path = self.csv_dire + f"test_outputs_{self.trainer.global_rank}.csv"
        self.add_data_to_csv(file_path, new_data, columns=["hypotheses", "targets"])

        return loss

    def on_test_epoch_end(self):
        all_hypotheses = []
        all_refs = []
        for idx in range(self.trainer.world_size):
            rank_file = self.csv_dire + f"test_outputs_{idx}.csv"
            if os.path.exists(rank_file):
                df = pd.read_csv(rank_file, sep='|')
                all_hypotheses.extend(df['hypotheses'].tolist())
                all_refs.extend(df['targets'].tolist())

        print(len(all_refs), len(all_hypotheses))
        bleu = self.bleu_metric.corpus_score(all_hypotheses, [all_refs]).score
        self.log("test_bleu", bleu, prog_bar=True, sync_dist=True)
        self.test_decoded = []
        self.test_step_outputs = []

    def teacher_forcing_generate(self, logits):
        """
        Args:
            logits: [batch_size, seq_len, vocab_size], 
                    where the logit at position i predicts the i-th token.
        Returns:
            Decoded strings aligned so that position i in `predicted` 
            corresponds to the i-th token in the reference.
        """
        # Skip the final position, which corresponds to predicting the token
        # after the sequence ends.
        predicted = torch.argmax(logits[:, :-1, :], dim=-1)

        # Decode the tokens
        generated_texts = self.tokenizer.batch_decode(predicted, skip_special_tokens=True)
        return generated_texts


    # def teacher_forcing_generate(self, logits):
    #     predicted = torch.argmax(logits, dim=-1)
    #     generated_texts = self.tokenizer.batch_decode(predicted, skip_special_tokens=True)
    #     return generated_texts

    def generate(self, src_input):
        max_new_tokens, num_beams, decoder_start_token_id = 150, 4, self.tokenizer.lang_code_to_id['en_XX']
        out = self.model.generate(
            src_input,
            max_new_tokens,
            num_beams,
            decoder_start_token_id
        )
        generated_texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return generated_texts

    def calc_loss(self, outputs, targets):
        vocab_siz = outputs.size(-1)
        return self.criterion(outputs.reshape(-1, vocab_siz), targets.reshape(-1))

    def add_weight_decay(self, weight_decay, skip_list=()):
        decay = []
        no_decay = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            elif name.endswith(".bias") or "LayerNorm.weight" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {'params': no_decay, 'weight_decay': 0.0},
            {'params': decay, 'weight_decay': weight_decay}
        ]

    def configure_optimizers(self):
        print(f'lr: {self.lr}')
        optimizer = torch.optim.AdamW(self.add_weight_decay(weight_decay=0.01), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.15,
                anneal_strategy='cos'
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

    def add_data_to_csv(self, file_path, new_data, columns):
        file_exists = os.path.exists(file_path)
        df = pd.DataFrame(new_data, columns=columns)
        if file_exists:
            df.to_csv(file_path, mode='a', index=False, header=False, sep='|')
            print(f"Data appended to {file_path}.")
        else:
            df.to_csv(file_path, mode='w', index=False, header=True, sep='|')
            print(f"New file created with data: {file_path}.")
