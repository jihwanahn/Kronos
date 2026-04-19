"""Orchestrator: fine-tune Kronos-base on the full KRX universe.

Design:
- Reuses the existing `train_tokenizer` and `train_model` functions from the
  vanilla scripts so we don't duplicate 800 lines of training loop code.
- Injects `KRXMultiTickerDataset` via monkey-patching the two modules'
  `create_dataloaders` before calling the training functions.
- Adds `data_dir`, `ticker_val_ratio`, `reserve_tail` to the config object so
  the KRX dataset picks them up.

Usage:
    python train_krx.py --config configs/config_krx.yaml
    python train_krx.py --config configs/config_krx.yaml --smoke   # 1-epoch sanity
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from config_loader import CustomFinetuneConfig  # noqa: E402
import finetune_tokenizer as ft_tok  # noqa: E402
import finetune_base_model as ft_bm  # noqa: E402
from model import Kronos, KronosTokenizer  # noqa: E402

from krx_multi_dataset import KRXMultiTickerDataset  # noqa: E402


def augment_config(config: CustomFinetuneConfig) -> None:
    """Pull KRX-specific fields from the YAML through the underlying loader."""
    get = config.loader.get
    config.data_dir = get("data.data_dir") or config.data_path
    config.ticker_val_ratio = float(get("data.ticker_val_ratio", 0.1))
    config.reserve_tail = int(get("data.reserve_tail", 60))


def build_krx_loaders(config, seed_offset: int = 0):
    train_ds = KRXMultiTickerDataset(
        data_dir=config.data_dir,
        data_type="train",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + seed_offset,
        ticker_val_ratio=config.ticker_val_ratio,
        reserve_tail=config.reserve_tail,
    )
    val_ds = KRXMultiTickerDataset(
        data_dir=config.data_dir,
        data_type="val",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + seed_offset,
        ticker_val_ratio=config.ticker_val_ratio,
        reserve_tail=config.reserve_tail,
    )

    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=dist.get_world_size(),
                           rank=dist.get_rank(), shuffle=True)
        if use_ddp else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=dist.get_world_size(),
                           rank=dist.get_rank(), shuffle=False, drop_last=False)
        if use_ddp else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        persistent_workers=(config.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler,
        persistent_workers=(config.num_workers > 0),
    )
    return train_loader, val_loader, train_ds, val_ds, train_sampler, val_sampler


def run_tokenizer_phase(config, device):
    print("\n" + "=" * 60)
    print("PHASE 1: Tokenizer fine-tuning")
    print("=" * 60)

    os.makedirs(config.tokenizer_save_path, exist_ok=True)
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = ft_tok.setup_logging(config.exp_name, log_dir, rank=0)
    ft_tok.set_seed(config.seed)

    if config.pre_trained_tokenizer:
        print(f"Loading pretrained tokenizer: {config.pretrained_tokenizer_path}")
        tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    else:
        raise RuntimeError("Expected pre_trained_tokenizer=true for KRX fine-tune")
    tokenizer = tokenizer.to(device)

    # Inject KRX dataloader
    ft_tok.create_dataloaders = lambda cfg: build_krx_loaders(cfg, seed_offset=0)

    best = ft_tok.train_tokenizer(tokenizer, device, config, config.tokenizer_save_path, logger)
    print(f"Tokenizer phase done. best_val_loss={best:.4f}")
    return best


def run_basemodel_phase(config, device):
    print("\n" + "=" * 60)
    print("PHASE 2: Base predictor fine-tuning")
    print("=" * 60)

    os.makedirs(config.basemodel_save_path, exist_ok=True)
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = ft_bm.setup_logging(config.exp_name, log_dir, rank=0)

    # Use fine-tuned tokenizer from phase 1
    tokenizer_path = config.finetuned_tokenizer_path
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Fine-tuned tokenizer not found at {tokenizer_path}. Run phase 1 first.")
    print(f"Loading fine-tuned tokenizer: {tokenizer_path}")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    print(f"Loading pretrained predictor: {config.pretrained_predictor_path}")
    model = Kronos.from_pretrained(config.pretrained_predictor_path).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M")

    # Inject KRX dataloader
    ft_bm.create_dataloaders = lambda cfg: build_krx_loaders(cfg, seed_offset=1)

    best = ft_bm.train_model(model, tokenizer, device, config, config.basemodel_save_path, logger)
    print(f"Basemodel phase done. best_val_loss={best:.4f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs" / "config_krx.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="1-epoch tokenizer + 1-epoch basemodel sanity run")
    ap.add_argument("--skip-tokenizer", action="store_true",
                    help="Skip phase 1 (reuse existing fine-tuned tokenizer)")
    ap.add_argument("--skip-basemodel", action="store_true", help="Skip phase 2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  cuda_available={torch.cuda.is_available()}")

    config = CustomFinetuneConfig(args.config)
    augment_config(config)
    if args.smoke:
        config.tokenizer_epochs = 1
        config.basemodel_epochs = 1

    print(f"exp_name={config.exp_name}")
    print(f"data_dir={config.data_dir}")
    print(f"lookback={config.lookback_window}  predict={config.predict_window}")
    print(f"batch={config.batch_size}  tok_epochs={config.tokenizer_epochs}  "
          f"base_epochs={config.basemodel_epochs}")
    print(f"save_root={config.base_save_path}")

    t0 = time.time()
    if config.train_tokenizer and not args.skip_tokenizer:
        run_tokenizer_phase(config, device)
    if config.train_basemodel and not args.skip_basemodel:
        run_basemodel_phase(config, device)
    print(f"\nTotal wall-clock: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
