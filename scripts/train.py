from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as C
from src.data import load_split_dataset
from src.model import build_model
from src.utils import create_train_val_dataloaders, fix_random, resolve_device, train_model


def main():
    dataset_path = C.DATA_DIR / C.dataset_filename
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} does not exist. Please run `python scripts/make_tiny_dataset.py` first."
        )

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fix_random(C.seed)
    device = resolve_device(C.device)

    dataset = load_split_dataset(dataset_path, "train")
    train_loader, val_loader = create_train_val_dataloaders(
        dataset=dataset,
        batch_size=C.batch_size,
        val_fraction=C.val_fraction,
        seed=C.seed,
    )

    model = build_model(C, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)

    print("---------------------------------")
    print(device)
    print("---------------------------------")
    print("train start")
    print("---------------------------------")

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=C.epochs,
        grad_clip=C.grad_clip,
        device=device,
        output_dir=C.OUTPUT_DIR,
    )

    summary = {
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "device": str(device),
        "train_size_after_split": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
    }
    with open(C.OUTPUT_DIR / "train_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print("---------------------------------")
    print(f"best_epoch={result['best_epoch']}")
    print(f"best_val_loss={result['best_val_loss']:.6f}")
    print("---------------------------------")


if __name__ == "__main__":
    main()
