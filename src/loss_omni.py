# losses_omni_jepa_infonce.py
# A drop-in loss module for (1) JEPA alignment stage and (2) InfoNCE semantic stage.
# - Works in single-GPU and DDP.
# - DDP InfoNCE uses "global negatives + remote detach" to avoid autograd break from dist.all_gather.
# - Supports paired batches (q[i] matches d[i]) and in-example negatives (q[i] matches y[i,0]).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


@torch.no_grad()
def ddp_all_gather_counts(local_count: int, device: torch.device) -> Tuple[int, int, list[int]]:
    """
    Gather per-rank local batch sizes. Returns (world, rank, counts).
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    t = torch.tensor([local_count], device=device, dtype=torch.long)
    ts = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(ts, t)
    counts = [int(x.item()) for x in ts]
    return world, rank, counts


def l2norm(x: Tensor) -> Tensor:
    return F.normalize(x, dim=-1)


# ----------------------------
# JEPA Stage-1
# ----------------------------

class MLPredictor(nn.Module):
    """
    A small predictor g(.) for JEPA.
    Default: Linear -> GELU -> Linear, same in/out dim.
    """
    def __init__(self, dim: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class JEPALoss(nn.Module):
    """
    JEPA: || g(z_c) - stopgrad(z_t) ||_2^2

    Args:
        predictor: g(.)
        normalize: L2-normalize both sides before MSE (often stabilizes)
        reduction: 'mean' or 'sum'
    """
    def __init__(
        self,
        predictor: nn.Module,
        normalize: bool = True,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super().__init__()
        self.predictor = predictor
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, z_c: Tensor, z_t: Tensor) -> Tensor:
        # Align dtype/device with predictor params to avoid bf16/fp32 matmul mismatch.
        ref_param = next(self.predictor.parameters())
        z_c = z_c.to(device=ref_param.device, dtype=ref_param.dtype)
        pred = self.predictor(z_c)
        tgt = z_t.detach().to(device=pred.device, dtype=pred.dtype)

        if self.normalize:
            pred = l2norm(pred)
            tgt = l2norm(tgt)

        return F.mse_loss(pred, tgt, reduction=self.reduction)


# ----------------------------
# InfoNCE Stage-2
# ----------------------------

class InfoNCELoss(nn.Module):
    """
    Local InfoNCE as cross-entropy over similarities.

    Use cases:
      1) paired: q: [B,D], d: [B,D], positives aligned by index (i -> i)
      2) grouped: q: [B,D], d: [B*K,D], positives at d[i*K] for each i (target_per_qry=K)
      3) explicit targets: pass target indices

    Args:
        temperature: tau
        normalize: use cosine similarity (L2 normalized)
    """
    def __init__(self, temperature: float = 0.02, normalize: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = normalize

    def forward(
        self,
        q: Tensor,
        d: Tensor,
        target: Optional[Tensor] = None,
        target_per_qry: Optional[int] = None,
        reduction: str = "mean",
    ) -> Tensor:
        if self.normalize:
            q = l2norm(q)
            d = l2norm(d)

        if target is None:
            if q.size(0) == d.size(0):
                target = torch.arange(q.size(0), device=q.device, dtype=torch.long)
            else:
                # d is grouped per query: [pos, neg1, ..., negK-1] repeated
                if target_per_qry is None:
                    if d.size(0) % q.size(0) != 0:
                        raise ValueError(f"d.size(0)={d.size(0)} not divisible by q.size(0)={q.size(0)}. "
                                         "Provide target or target_per_qry.")
                    target_per_qry = d.size(0) // q.size(0)
                target = torch.arange(
                    0, q.size(0) * target_per_qry, target_per_qry,
                    device=q.device, dtype=torch.long
                )

        logits = q @ d.t()
        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


class DDPInfoNCELoss(nn.Module):
    """
    DDP InfoNCE with global negatives, using remote-detach to keep autograd intact.

    Assumption (paired mode):
      - q: [B,D], d: [B,D] on each rank
      - positives aligned by index within each rank (i -> i)
      - global doc matrix is rank-ordered concatenation of per-rank docs
      - labels are offset by sum(counts[:rank])

    This updates:
      - local q encoder (always)
      - local d encoder (because local docs are inserted without detach)
    Remote docs are used only as constant negatives.
    """
    def __init__(self, temperature: float = 0.02, normalize: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = normalize

    def forward(self, q: Tensor, d: Tensor, reduction: str = "mean") -> Tensor:
        if not is_ddp():
            # fallback to local paired
            if q.size(0) != d.size(0):
                raise ValueError(f"DDPInfoNCELoss expects paired local batches. Got q={q.shape}, d={d.shape}")
            if self.normalize:
                q = l2norm(q)
                d = l2norm(d)
            logits = q @ d.t()
            target = torch.arange(q.size(0), device=q.device, dtype=torch.long)
            return F.cross_entropy(logits / self.temperature, target, reduction=reduction)

        if q.size(0) != d.size(0):
            raise ValueError(f"DDPInfoNCELoss expects paired local batches. Got q={q.shape}, d={d.shape}")

        device = q.device
        world, rank, counts = ddp_all_gather_counts(q.size(0), device=device)
        max_b = max(counts) if counts else q.size(0)

        # Pad docs to max_b for all_gather
        if d.size(0) < max_b:
            pad = d.new_zeros((max_b - d.size(0), d.size(1)))
            d_pad = torch.cat([d, pad], dim=0)
        else:
            d_pad = d

        gathered = [torch.empty_like(d_pad) for _ in range(world)]
        dist.all_gather(gathered, d_pad)

        # Build global doc matrix with remote detach
        parts = []
        for r, (g, b) in enumerate(zip(gathered, counts)):
            if r == rank:
                parts.append(d)              # keep local grad
            else:
                parts.append(g[:b].detach()) # remote constant negatives
        all_d = torch.cat(parts, dim=0) if parts else d

        if self.normalize:
            q = l2norm(q)
            all_d = l2norm(all_d)

        offset = sum(counts[:rank])
        target = torch.arange(q.size(0), device=device, dtype=torch.long) + offset

        logits = q @ all_d.t()
        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


class InExampleContrastiveLoss(nn.Module):
    """
    In-example categorization loss.
    x: [B, D]
    y: [B, K, D]   (K = 1 + n_hard_negatives), where y[:,0] is the positive

    This does NOT use DDP gather by default (keeps it simple & stable).
    If you need DDP gather, do it outside and keep shapes consistent.
    """
    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, ndim: Optional[int] = None, normalize: bool = True):
        super().__init__()
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = float(temperature)
        self.ndim = ndim
        self.normalize = normalize

    def forward(self, x: Tensor, y: Tensor, reduction: str = "mean") -> Tuple[Tensor, Dict[str, Tensor]]:
        # x: [B,D], y: [B,K,D]
        if x.dim() != 2 or y.dim() != 3:
            raise ValueError(f"Expected x:[B,D], y:[B,K,D]. Got x={x.shape}, y={y.shape}")

        if self.ndim is not None:
            x = x[:, : self.ndim]
            y = y[:, :, : self.ndim]

        if self.normalize:
            x = l2norm(x)
            y = l2norm(y)

        B, K, D = y.shape
        if x.size(0) != B:
            raise ValueError(f"x and y batch mismatch: x={x.shape}, y={y.shape}")

        # logits: [B,K]
        logits = torch.einsum("bd,bkd->bk", x, y) * self.temperature
        labels = torch.zeros(B, device=x.device, dtype=torch.long)
        preds = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels, reduction=reduction)

        detail = {"logits": logits, "labels": labels, "preds": preds}
        return loss, detail


# ----------------------------
# Two-stage wrapper
# ----------------------------

@dataclass
class TwoStageLossOutput:
    loss: Tensor
    components: Dict[str, Tensor]


class OmniTwoStageLoss(nn.Module):
    """
    A wrapper that exposes:
      - stage="jepa": JEPA(z_c, z_t)
      - stage="infonce": InfoNCE(q, d) (local or DDP)
      - stage="mixed": alpha * JEPA + (1-alpha) * InfoNCE

    Notes:
      - JEPA is typically stage-1 to align modality spaces.
      - InfoNCE is stage-2 for semantic alignment.
    """
    def __init__(
        self,
        emb_dim: int,
        jepa_temperature: float = 0.02,  # kept for API symmetry (not used in MSE)
        infonce_temperature: float = 0.02,
        use_ddp_infonce: bool = True,
        jepa_predictor_hidden: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize

        self.predictor = MLPredictor(dim=emb_dim, hidden=jepa_predictor_hidden)
        self.jepa = JEPALoss(self.predictor, normalize=normalize, reduction="mean")

        if use_ddp_infonce:
            self.infonce = DDPInfoNCELoss(temperature=infonce_temperature, normalize=normalize)
        else:
            self.infonce = InfoNCELoss(temperature=infonce_temperature, normalize=normalize)

    def forward(
        self,
        stage: Literal["jepa", "infonce", "mixed"],
        *,
        # JEPA inputs:
        z_c: Optional[Tensor] = None,
        z_t: Optional[Tensor] = None,
        # InfoNCE inputs:
        q: Optional[Tensor] = None,
        d: Optional[Tensor] = None,
        # Mixed:
        alpha: float = 0.5,
        reduction: str = "mean",
    ) -> TwoStageLossOutput:
        components: Dict[str, Tensor] = {}

        if stage == "jepa":
            if z_c is None or z_t is None:
                raise ValueError("stage='jepa' requires z_c and z_t")
            loss = self.jepa(z_c, z_t)
            components["jepa"] = loss
            return TwoStageLossOutput(loss=loss, components=components)

        if stage == "infonce":
            if q is None or d is None:
                raise ValueError("stage='infonce' requires q and d")
            loss = self.infonce(q, d, reduction=reduction)
            components["infonce"] = loss
            return TwoStageLossOutput(loss=loss, components=components)

        if stage == "mixed":
            if z_c is None or z_t is None or q is None or d is None:
                raise ValueError("stage='mixed' requires z_c, z_t, q, d")
            alpha = float(alpha)
            alpha = max(0.0, min(1.0, alpha))
            loss_jepa = self.jepa(z_c, z_t)
            loss_nce = self.infonce(q, d, reduction=reduction)
            loss = alpha * loss_jepa + (1.0 - alpha) * loss_nce
            components["jepa"] = loss_jepa
            components["infonce"] = loss_nce
            components["alpha"] = torch.tensor(alpha, device=loss.device)
            return TwoStageLossOutput(loss=loss, components=components)

        raise ValueError(f"Unknown stage: {stage}")


# ----------------------------
# Backward safety helper (optional)
# ----------------------------

def assert_loss_has_grad(loss: Tensor, name: str = "loss") -> None:
    """
    Useful for debugging: ensure loss is differentiable.
    """
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(loss)}")
    if not loss.requires_grad or loss.grad_fn is None:
        raise RuntimeError(f"{name} has no grad_fn (requires_grad={loss.requires_grad}). "
                           "This often indicates a detached graph (e.g., non-autograd all_gather or .item()).")
