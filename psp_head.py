import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import importlib

import fasttext.util
import pickle
# from dataloaders.data_utils.file_utils import read_pickle
# import ignite.distributed as idist

class HeadModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        emb_lang,
        emb_pkl_dir,
        trainable_emb,
        dropout=0.5,
        class_temperature=0.1,
        time_temperature=0.1,
        dynamic_time_temperatures=False,
        dynamic_class_temperatures=False,
    ):
        super().__init__()
        if not dynamic_time_temperatures:
            self.register_buffer("time_temperature", torch.tensor(time_temperature))
        else:
            self.time_temperature = torch.nn.Parameter(torch.ones(1) * time_temperature)
        if not dynamic_class_temperatures:
            self.register_buffer("class_temperature", torch.tensor(class_temperature))
        else:
            self.class_temperature = torch.nn.Parameter(
                torch.ones(1) * class_temperature
            )

        if hidden_dim is not None:
            self.fc_hidden = nn.Linear(in_dim, hidden_dim)
        else:
            self.fc_hidden = nn.Identity()
            hidden_dim = in_dim

        # if idist.get_local_rank() == 0 or idist.get_world_size() == 0:
        fasttext.util.download_model(emb_lang, if_exists="ignore")
        # exit(0)
        # if idist.get_world_size() > 0:
        #     idist.barrier()
        ft = fasttext.load_model(f"cc.{emb_lang}.300.bin")

        with open(emb_pkl_dir, 'rb') as f:
            dict_processed_words = pickle.load(f) # DICTIONARY CONTAINING THE {<PSUEDOGLOSS>:<ID>}
        dict_lem_to_id = dict_processed_words["dict_lem_to_id"]
        vector = torch.zeros((len(dict_lem_to_id), 300))
        for key, value in dict_lem_to_id.items():
            vector[value] = torch.tensor(ft.get_word_vector(key))

        self.vocab_embedding = torch.nn.Parameter(vector.permute(1, 0).unsqueeze(-1))
        self.vocab_embedding.requires_grad = trainable_emb
        zero_embedding = torch.zeros(hidden_dim, 1, 1)
        self.register_buffer("zero_embedding", zero_embedding)

        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

    def logit_compare_embed(self, out, vocab):
        N, T, C = out.shape

        vocab = torch.cat([vocab, self.zero_embedding], dim=1)
        _, V, M = vocab.shape

        out = F.normalize(out, dim=-1)

        vocab = F.normalize(vocab, dim=0)

        fc_out = torch.bmm(
            out.reshape(N, T, C),
            vocab.reshape(C, V * M).unsqueeze(0).repeat(N, 1, 1),
        ).reshape(N, T, V, M)

        fc_out = F.adaptive_avg_pool3d(fc_out, (T, V, 1)).squeeze(-1)

        logits = fc_out.reshape(N, T, V)

        return logits

    def forward(self, x, mask):
        b, t, c = x.shape

        y = self.fc_hidden(self.dropout(x))

        time_res = self.logit_compare_embed(y, self.vocab_embedding)
        cls_temp = torch.clamp(self.class_temperature, 0.01, 1.0)
        cls_softmax = (time_res / cls_temp).softmax(axis=-1)
        time_mask = (
            (~mask)
            .type(time_res.dtype)
            .masked_fill(~mask, torch.finfo(time_res.dtype).min)
            .unsqueeze(-1)
        )
        time_temp = torch.clamp(self.time_temperature, 0.01, 1.0)
        time_softmax = (time_res / time_temp + time_mask).softmax(axis=-2)

        softmax_scores = cls_softmax * time_softmax
        class_scores = softmax_scores.sum(axis=-2)
        logits = class_scores[:,:]

        return {
            "time_res": time_res,
            "softmax_scores": softmax_scores,
            "logits": logits,
            "mask": mask,
        }

class GaussianHeadModel(nn.Module):
    """
    This model:
      1) Takes video features x of shape [B, T, D].
      2) Learns an embedding for each possible gloss, shape [V, D].
      3) Computes a dot-product alignment => shape [B, T, V].
      4) Applies a Gaussian weighting to encourage each gloss v to only match frames near its ideal time.
      5) Applies a double-softmax over both the time and vocab dimensions.
      6) Aggregates to final [B, V] logits.
    """
    def __init__(self,
                 num_classes: int,   # Size of pseudo-gloss vocabulary
                 in_dim: int,        # Dimension of input frame features (e.g., 1024)
                 hidden_dim: int,    # Projection dimension (for both frames and gloss embeddings)
                 sigma: float = 10.0,
                 dropout: float = 0.5,
                 use_double_softmax: bool = True,
                 scale_time: bool = True,
                 temperature: float = 0.1):
        """
        :param num_classes: Number of gloss tokens.
        :param in_dim: Input frame feature dimension.
        :param hidden_dim: Dimension to project both frame features and gloss embeddings.
        :param sigma: Initial standard deviation for Gaussian weighting.
        :param dropout: Dropout probability.
        :param use_double_softmax: Whether to use double-softmax over time & vocab.
        :param scale_time: If True, estimate each gloss’s time index by linear mapping.
        :param temperature: Temperature for scaling the combined scores.
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.use_double_softmax = use_double_softmax
        self.scale_time = scale_time
        self.temperature = temperature

        # 1) Linear projection from in_dim to hidden_dim
        self.frame_proj = nn.Linear(in_dim, hidden_dim)
        # Scale down initial projections
        # (We multiply by a constant factor later in forward.)
        
        # Register sigma as a learnable parameter, but constrain it via a minimum value.
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        # We'll register a buffer for sigma_min to prevent it from collapsing.
        self.register_buffer("sigma_min", torch.tensor(2.0))

        # 2) Gloss embeddings: shape [num_classes, hidden_dim]
        self.gloss_embedding = nn.Embedding(num_classes, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.frame_proj.weight)
        nn.init.normal_(self.gloss_embedding.weight, std=0.02)

    def forward(self, x, mask=None):
        """
        :param x: Tensor of shape [B, T, in_dim] (video frame features)
        :param mask: Optional binary mask of shape [B, T] (1 for valid frames, 0 for padding)
        :return: A dict with keys:
                 "logits": [B, num_classes] – final gloss logits,
                 "alignment": [B, T, num_classes] – the per-frame alignment distribution,
                 "scores": [B, T, num_classes] – the combined scores before softmax.
        """
        # Ensure no non-finite values are present.
        assert torch.isfinite(x).all(), "Non-finite values in head input!"

        B, T, _ = x.shape

        # 1) Project frames to hidden_dim and apply dropout.
        x = self.dropout(x)
        # Scale down the projection for stability.
        x = self.frame_proj(x) * 0.1  # scaling factor
        # Normalize each feature vector.
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)

        # 2) Obtain gloss embeddings and normalize.
        g = self.gloss_embedding.weight
        g = g / (g.norm(dim=-1, keepdim=True) + 1e-9)
        # Optionally scale gloss embeddings.
        g = g * 0.5

        # 3) Compute dot-product scores: [B, T, num_classes]
        scores = torch.matmul(x, g.transpose(0, 1))
        # We scale scores by an inverse temperature factor (here, 1/0.1).
        scores = scores / self.temperature

        # 4) Compute Gaussian weighting.
        # Determine time indices for frames and gloss tokens.
        t_idx = torch.arange(T, device=x.device).float()  # shape [T]
        if self.scale_time:
            v_idx = torch.arange(self.num_classes, device=x.device).float() + 0.5
            v_idx = v_idx / self.num_classes * T  # shape [num_classes]
        else:
            v_idx = torch.arange(self.num_classes, device=x.device).float()
        # Compute squared distance matrix: we want shape [num_classes, T]
        dist = (t_idx.unsqueeze(0) - v_idx.unsqueeze(1)) ** 2  # [V, T]

        # Clamp sigma to ensure it doesn't go below sigma_min.
        sigma = torch.clamp(self.sigma, min=self.sigma_min)
        sigma_sq = sigma**2 + 1e-6

        # Compute Gaussian weights: w(v,t) = exp(- (t - v_idx)^2 / (2*sigma^2) )
        weights = torch.exp(-dist / (2.0 * sigma_sq))  # shape [V, T]
        # We want to add the log of weights to our scores, so transpose to [T, V] and add epsilon.
        weights_t_v = weights.transpose(0, 1)  # shape [T, V]
        log_w = torch.log(weights_t_v + 1e-12).unsqueeze(0)  # shape [1, T, V]

        # Add the log Gaussian weights to scores.
        combined_scores = scores + log_w

        # 5) Apply mask if provided.
        if mask is not None:
            # mask: [B, T] --> unsqueeze to [B, T, 1]
            mask = mask.unsqueeze(-1)
            # It is critical that every sample has at least one valid frame.
            if not mask.any(dim=1).all():
                raise ValueError("One or more samples have all frames masked!")
            combined_scores = combined_scores.masked_fill(~mask.bool(), -1e9)

        # 6) Compute double softmax in log space for stability.
        # Compute log softmax over vocab dimension (axis=-1) and over time dimension (axis=-2)
        log_cls_softmax = F.log_softmax(combined_scores, dim=-1)   # [B, T, V]
        log_time_softmax = F.log_softmax(combined_scores, dim=-2)    # [B, T, V]
        # Combine these: elementwise sum in log-space corresponds to product in normal space.
        log_alignment = log_cls_softmax + log_time_softmax
        alignment = torch.exp(log_alignment)  # [B, T, V]

        # 7) Aggregate over time: use logsumexp over time dimension to get [B, V]
        logits_log = torch.logsumexp(log_alignment, dim=1)
        # For numerical stability, subtract the max and then exponentiate.
        logits = torch.exp(logits_log - torch.max(logits_log, dim=-1, keepdim=True)[0])
        # Now logits should be positive and finite.
        
        return {
            "logits": logits,         # [B, num_classes]
            "alignment": alignment,     # [B, T, num_classes]
            "scores": combined_scores   # [B, T, num_classes]
        }
    

if __name__ == "__main__":
    import torch

    model = HeadModel(
        in_dim=1024,
        hidden_dim=300,
        num_classes=2,
        emb_lang="en",
        emb_pkl_dir="data/ytsl/processed_words.pkl",
        trainable_emb=True,
        dropout=0.5,
        class_temperature=0.1,
        time_temperature=0.1,
        dynamic_time_temperatures=False,
        dynamic_class_temperatures=False,
    )
    x = torch.randn(2, 10, 1024)
    mask = torch.ones(2, 10).bool()
    model(x, mask)