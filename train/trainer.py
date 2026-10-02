"""Training loop for MultiMarketModel."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.device import default_num_workers, get_device, is_cuda, setup_device
from config.settings import (
    BATCH_SIZE,
    DIR_CHECKPOINTS,
    EPOCHS,
    LEARNING_RATE,
    LOSS_WEIGHT_5D,
    LOSS_WEIGHT_20D,
    WEIGHT_DECAY,
)
from models.multi_head import MultiMarketModel
from train.dataset import MultiMarketDataset, estimate_class_weights

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    num_workers: Optional[int] = None  # None → auto (CUDA=4, MPS/CPU=0)
    device: str = field(default_factory=get_device)
    checkpoint_dir: Path = DIR_CHECKPOINTS
    max_symbols_per_market: Optional[int] = None


def _loss_fn(
    logits: dict,
    y5: torch.Tensor,
    y20: torch.Tensor,
    crit5: nn.Module,
    crit20: nn.Module,
) -> torch.Tensor:
    return LOSS_WEIGHT_5D * crit5(logits["5d"], y5) + LOSS_WEIGHT_20D * crit20(logits["20d"], y20)


def _to_device(batch, device: str, non_blocking: bool):
    x, y5, y20, mid = batch
    return (
        x.to(device, non_blocking=non_blocking),
        y5.to(device, non_blocking=non_blocking),
        y20.to(device, non_blocking=non_blocking),
        mid.to(device, non_blocking=non_blocking),
    )


@torch.no_grad()
def evaluate(model: MultiMarketModel, loader: DataLoader, crit5, crit20, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    non_blocking = is_cuda(device)
    for batch in loader:
        x, y5, y20, mid = _to_device(batch, device, non_blocking)
        logits = model(x, mid)
        loss = _loss_fn(logits, y5, y20, crit5, crit20)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train(cfg: TrainConfig | None = None) -> Path:
    cfg = cfg or TrainConfig()
    device = setup_device(cfg.device)
    cfg.device = device
    if cfg.num_workers is None:
        cfg.num_workers = default_num_workers(device)
    logger.info("training on device=%s num_workers=%s batch_size=%s", device, cfg.num_workers, cfg.batch_size)

    train_ds = MultiMarketDataset("train", max_symbols_per_market=cfg.max_symbols_per_market)
    val_ds = MultiMarketDataset("val", max_symbols_per_market=cfg.max_symbols_per_market)
    if len(train_ds) == 0:
        raise RuntimeError("empty train dataset; download + label data first")
    logger.info("dataset sizes train=%s val=%s", len(train_ds), len(val_ds))

    pin = is_cuda(device)
    loader_common = dict(
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        loader_common["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    model = MultiMarketModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model params=%.2fK", n_params / 1e3)

    w5 = estimate_class_weights(train_ds, horizon=5).to(device)
    w20 = estimate_class_weights(train_ds, horizon=20).to(device)
    logger.info("class weights 5d=%s 20d=%s", w5.tolist(), w20.tolist())
    crit5 = nn.CrossEntropyLoss(weight=w5, label_smoothing=0.05)
    crit20 = nn.CrossEntropyLoss(weight=w20, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = cfg.checkpoint_dir / "best_val_loss.pt"
    non_blocking = is_cuda(device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            x, y5, y20, mid = _to_device(batch, device, non_blocking)
            opt.zero_grad(set_to_none=True)
            logits = model(x, mid)
            loss = _loss_fn(logits, y5, y20, crit5, crit20)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
            pbar.set_postfix(loss=running / max(seen, 1))
        sched.step()

        train_loss = running / max(seen, 1)
        val_loss = evaluate(model, val_loader, crit5, crit20, device) if len(val_ds) else train_loss
        logger.info("epoch %s train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)

        latest = cfg.checkpoint_dir / "latest.pt"
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "val_loss": val_loss,
            "train_loss": train_loss,
            "device": device,
        }
        torch.save(payload, latest)
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(payload, best_path)
            logger.info("saved best checkpoint -> %s", best_path)

    return best_path


def load_model(path: Path | str, device: Optional[str] = None) -> MultiMarketModel:
    device = setup_device(device or get_device())
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MultiMarketModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
