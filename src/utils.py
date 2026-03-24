from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split


def fix_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_train_val_dataloaders(dataset, batch_size: int, val_fraction: float, seed: int):
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def save_loss_plot(history: dict, output_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"]["total"], label="train_total")
    plt.plot(history["train"]["action"], label="train_action")
    plt.plot(history["train"]["bldi"], label="train_bldi")
    plt.plot(history["val"]["total"], label="val_total", linestyle="--")
    plt.plot(history["val"]["action"], label="val_action", linestyle="--")
    plt.plot(history["val"]["bldi"], label="val_bldi", linestyle="--")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_model(model, train_loader, val_loader, optimizer, epochs: int, grad_clip: float, device: torch.device, output_dir: Path):
    history = {
        "train": {"total": [], "action": [], "bldi": []},
        "val": {"total": [], "action": [], "bldi": []},
    }
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_totals = {"total": 0.0, "action": 0.0, "bldi": 0.0}
        for batch in train_loader:
            bld, stimulus, action, bldi = [tensor.to(device) for tensor in batch]
            output_act, output_bldi, _, _, _ = model(bld, stimulus)
            loss, each_loss = model.calc_loss(output_act, action, output_bldi, bldi)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_totals["total"] += loss.item()
            train_totals["action"] += each_loss["action_loss"].item()
            train_totals["bldi"] += each_loss["bldi_loss"].item()

        train_batches = len(train_loader)
        history["train"]["total"].append(train_totals["total"] / train_batches)
        history["train"]["action"].append(train_totals["action"] / train_batches)
        history["train"]["bldi"].append(train_totals["bldi"] / train_batches)

        model.eval()
        val_totals = {"total": 0.0, "action": 0.0, "bldi": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                bld, stimulus, action, bldi = [tensor.to(device) for tensor in batch]
                output_act, output_bldi, _, _, _ = model(bld, stimulus)
                loss, each_loss = model.calc_loss(output_act, action, output_bldi, bldi)
                val_totals["total"] += loss.item()
                val_totals["action"] += each_loss["action_loss"].item()
                val_totals["bldi"] += each_loss["bldi_loss"].item()

        val_batches = len(val_loader)
        epoch_val_loss = val_totals["total"] / val_batches
        history["val"]["total"].append(epoch_val_loss)
        history["val"]["action"].append(val_totals["action"] / val_batches)
        history["val"]["bldi"].append(val_totals["bldi"] / val_batches)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_dir / "checkpoint_best_val.pt")

        print(
            f"epoch={epoch + 1:03d} "
            f"train_total={history['train']['total'][-1]:.6f} "
            f"val_total={history['val']['total'][-1]:.6f} "
            f"train_action={history['train']['action'][-1]:.6f} "
            f"val_action={history['val']['action'][-1]:.6f} "
            f"train_bldi={history['train']['bldi'][-1]:.6f} "
            f"val_bldi={history['val']['bldi'][-1]:.6f}"
        )

    torch.save(model.state_dict(), output_dir / "last_epoch.pt")
    with open(output_dir / "loss_history.json", "w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)
    save_loss_plot(history, output_dir / "loss.png")

    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss, "history": history}


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)
