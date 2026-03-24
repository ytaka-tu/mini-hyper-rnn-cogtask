from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as C
from src.data import load_npz
from src.model import build_model
from src.utils import fix_random, resolve_device


def make_derangement(length: int, shift: int) -> np.ndarray:
    if length < 2:
        raise ValueError("Derangement requires at least 2 samples.")
    shift = shift % length
    if shift == 0:
        shift = 1
    return (np.arange(length) + shift) % length


def evaluate_predictions(
    model,
    bld: torch.Tensor,
    stimulus: torch.Tensor,
    action: torch.Tensor,
    bldi: torch.Tensor,
):
    with torch.no_grad():
        output_act, output_bldi, _, _, _ = model.open_loop(bld, stimulus)

    bold_mse = F.mse_loss(output_bldi[:, 1:, :], bldi[:, 1:, :]).item()
    act_bce = F.binary_cross_entropy(output_act[:, 1:, :], action[:, 1:, :]).item()
    act_accuracy = ((output_act[:, 1:, :] >= 0.5) == (action[:, 1:, :] >= 0.5)).float().mean().item()
    return {
        "output_act": output_act,
        "output_bldi": output_bldi,
        "bold_mse": bold_mse,
        "act_bce": act_bce,
        "act_accuracy": act_accuracy,
    }


def plot_subject_prediction(subject_id: str, output_np: np.ndarray, target_np: np.ndarray, output_labels: np.ndarray):
    fig, axes = plt.subplots(output_np.shape[1], 1, figsize=(12, 2.2 * output_np.shape[1]), sharex=True)
    x_axis = np.arange(output_np.shape[0])

    for index, axis in enumerate(axes):
        axis.plot(x_axis, target_np[:, index], label="target", linewidth=1.5, color="black")
        axis.plot(x_axis, output_np[:, index], label="prediction", linewidth=1.2, color="tab:orange", alpha=0.85)
        axis.set_ylabel(output_labels[index])
        axis.grid(True, axis="x", alpha=0.3)
        if index < C.action_dim:
            axis.set_ylim(-0.2, 1.2)
        if index == 0:
            axis.legend(loc="upper right")

    axes[-1].set_xlabel("time step")
    fig.suptitle(subject_id)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"pred_vs_target_{subject_id}.png")
    plt.close(fig)


def main():
    dataset_path = C.DATA_DIR / C.dataset_filename
    checkpoint_path = C.OUTPUT_DIR / C.best_checkpoint
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} does not exist. Please run `python scripts/make_tiny_dataset.py` first."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} does not exist. Please run `python scripts/train.py` first."
        )

    fix_random(C.seed)
    device = resolve_device(C.device)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    arrays = load_npz(dataset_path)
    bld = torch.tensor(arrays["test_bld"], dtype=torch.float32, device=device)
    stimulus = torch.tensor(arrays["test_stimulus"], dtype=torch.float32, device=device)
    action = torch.tensor(arrays["test_action"], dtype=torch.float32, device=device)
    bldi = torch.tensor(arrays["test_bldi"], dtype=torch.float32, device=device)
    subject_ids = arrays["test_subject_ids"]
    output_labels = arrays["output_labels"]

    model = build_model(C, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("---------------------------------")
    print(device)
    print("---------------------------------")
    print("test start")
    print("---------------------------------")

    aligned_eval = evaluate_predictions(model, bld, stimulus, action, bldi)
    print(f"BOLD MSE :{aligned_eval['bold_mse']:.5e}")
    print(f"act BCE :{aligned_eval['act_bce']:.5e}")
    print(f"act ACC :{aligned_eval['act_accuracy']:.5e}")

    derangement_shifts = [1, len(subject_ids) // 3, len(subject_ids) // 2]
    shuffled_results = []
    for shift in derangement_shifts:
        permutation = make_derangement(len(subject_ids), shift)
        shuffled_bld = bld[torch.tensor(permutation, device=device)]
        shuffled_eval = evaluate_predictions(model, shuffled_bld, stimulus, action, bldi)
        shuffled_results.append(
            {
                "shift": int(shift),
                "bold_mse": shuffled_eval["bold_mse"],
                "act_bce": shuffled_eval["act_bce"],
                "act_accuracy": shuffled_eval["act_accuracy"],
            }
        )

    shuffled_bold_mse = float(np.mean([result["bold_mse"] for result in shuffled_results]))
    shuffled_act_bce = float(np.mean([result["act_bce"] for result in shuffled_results]))
    shuffled_act_accuracy = float(np.mean([result["act_accuracy"] for result in shuffled_results]))

    print("---------------------------------")
    print("mismatched bld baseline")
    print("---------------------------------")
    print(f"BOLD MSE :{shuffled_bold_mse:.5e}")
    print(f"act BCE :{shuffled_act_bce:.5e}")
    print(f"act ACC :{shuffled_act_accuracy:.5e}")
    print("---------------------------------")
    print("aligned - mismatched")
    print("---------------------------------")
    print(f"BOLD MSE delta :{aligned_eval['bold_mse'] - shuffled_bold_mse:.5e}")
    print(f"act BCE delta  :{aligned_eval['act_bce'] - shuffled_act_bce:.5e}")
    print(f"act ACC delta  :{aligned_eval['act_accuracy'] - shuffled_act_accuracy:.5e}")

    output_act_np = aligned_eval["output_act"].detach().cpu().numpy()[:, 1:, :]
    output_bldi_np = aligned_eval["output_bldi"].detach().cpu().numpy()[:, 1:, :]
    output_np = np.concatenate((output_act_np, output_bldi_np), axis=2)
    action_np = action.detach().cpu().numpy()[:, 1:, :]
    bldi_np = bldi.detach().cpu().numpy()[:, 1:, :]
    target_np = np.concatenate((action_np, bldi_np), axis=2)

    for index, subject_id in enumerate(subject_ids):
        plot_subject_prediction(str(subject_id), output_np[index], target_np[index], output_labels)

    metrics = {
        "aligned": {
            "bold_mse": aligned_eval["bold_mse"],
            "act_bce": aligned_eval["act_bce"],
            "act_accuracy": aligned_eval["act_accuracy"],
        },
        "mismatched_bld_mean": {
            "bold_mse": shuffled_bold_mse,
            "act_bce": shuffled_act_bce,
            "act_accuracy": shuffled_act_accuracy,
        },
        "delta_aligned_minus_mismatched": {
            "bold_mse": aligned_eval["bold_mse"] - shuffled_bold_mse,
            "act_bce": aligned_eval["act_bce"] - shuffled_act_bce,
            "act_accuracy": aligned_eval["act_accuracy"] - shuffled_act_accuracy,
        },
        "mismatched_bld_runs": shuffled_results,
        "n_test_subjects": int(len(subject_ids)),
        "n_generated_figures": int(len(subject_ids)),
    }
    with open(C.OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)


if __name__ == "__main__":
    main()
