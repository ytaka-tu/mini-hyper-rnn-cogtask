from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as C
from src.utils import fix_random


LATENT_DIM = 4


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_subject_code_and_bld(rng: np.random.Generator, n_subjects: int, decoder: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subject_code = rng.normal(size=(n_subjects, LATENT_DIM)).astype(np.float32)
    bld = subject_code @ decoder + 0.03 * rng.normal(size=(n_subjects, C.bld_dim)).astype(np.float32)
    bld /= np.std(bld, axis=0, keepdims=True) + 1e-6
    return subject_code, bld.astype(np.float32)


def make_structured_stimulus(rng: np.random.Generator, n_subjects: int) -> np.ndarray:
    time_axis = np.linspace(0.0, 2.0 * math.pi, C.seq_len, dtype=np.float32)
    stimulus = np.zeros((n_subjects, C.seq_len, C.stimulus_dim), dtype=np.float32)

    for subject_index in range(n_subjects):
        phase = rng.uniform(0.0, 2.0 * math.pi, size=3).astype(np.float32)
        stimulus[subject_index, :, 0] = np.sin(time_axis + phase[0])
        stimulus[subject_index, :, 1] = np.cos(0.5 * time_axis + phase[1])
        stimulus[subject_index, :, 2] = np.sin(1.5 * time_axis + phase[2])

        pulse_positions = rng.choice(C.seq_len, size=max(4, C.seq_len // 8), replace=False)
        stimulus[subject_index, pulse_positions, 3] = 1.0
        stimulus[subject_index, pulse_positions, 4] = -1.0

        raw_noise = rng.normal(scale=0.18, size=(C.seq_len, C.stimulus_dim - 5)).astype(np.float32)
        for step in range(1, C.seq_len):
            raw_noise[step] = 0.82 * raw_noise[step - 1] + 0.18 * raw_noise[step]
        stimulus[subject_index, :, 5:] = raw_noise

    return np.tanh(stimulus).astype(np.float32)


def init_teacher_parameters(rng: np.random.Generator):
    hidden = C.hidden_size
    output_dim = C.action_dim + C.bldi_dim

    decoder = rng.normal(scale=0.65 / np.sqrt(LATENT_DIM), size=(LATENT_DIM, C.bld_dim)).astype(np.float32)

    base = {
        "weight_ih": rng.normal(scale=0.32 / np.sqrt(C.stimulus_dim), size=(hidden, C.stimulus_dim)).astype(np.float32),
        "weight_hh": (0.60 * np.eye(hidden) + rng.normal(scale=0.03, size=(hidden, hidden))).astype(np.float32),
        "bias_ih": rng.normal(scale=0.03, size=(hidden,)).astype(np.float32),
        "bias_hh": rng.normal(scale=0.03, size=(hidden,)).astype(np.float32),
        "weight_out": rng.normal(scale=0.26 / np.sqrt(hidden), size=(output_dim, hidden)).astype(np.float32),
        "bias_out": np.zeros((output_dim,), dtype=np.float32),
    }

    basis = {
        "weight_ih": rng.normal(scale=0.04 / np.sqrt(C.stimulus_dim), size=(LATENT_DIM, hidden, C.stimulus_dim)).astype(np.float32),
        "weight_hh": rng.normal(scale=0.015 / np.sqrt(hidden), size=(LATENT_DIM, hidden, hidden)).astype(np.float32),
        "bias_ih": rng.normal(scale=0.015, size=(LATENT_DIM, hidden)).astype(np.float32),
        "bias_hh": rng.normal(scale=0.015, size=(LATENT_DIM, hidden)).astype(np.float32),
        "weight_out": rng.normal(scale=0.08 / np.sqrt(hidden), size=(LATENT_DIM, output_dim, hidden)).astype(np.float32),
        "bias_out": rng.normal(scale=0.03, size=(LATENT_DIM, output_dim)).astype(np.float32),
    }

    return decoder, base, basis


def subject_specific_params(subject_code: np.ndarray, base: dict, basis: dict) -> dict[str, np.ndarray]:
    params = {}
    params["weight_ih"] = base["weight_ih"][None, :, :] + np.einsum("nl,lij->nij", subject_code, basis["weight_ih"])
    params["weight_hh"] = base["weight_hh"][None, :, :] + np.einsum("nl,lij->nij", subject_code, basis["weight_hh"])
    params["bias_ih"] = base["bias_ih"][None, :] + subject_code @ basis["bias_ih"]
    params["bias_hh"] = base["bias_hh"][None, :] + subject_code @ basis["bias_hh"]
    params["weight_out"] = base["weight_out"][None, :, :] + np.einsum("nl,loi->noi", subject_code, basis["weight_out"])
    params["bias_out"] = base["bias_out"][None, :] + subject_code @ basis["bias_out"]
    return params


def simulate_subject_specific_rnn(
    stimulus: np.ndarray,
    params: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_subjects = stimulus.shape[0]
    hidden = np.zeros((n_subjects, C.hidden_size), dtype=np.float32)
    action = np.zeros((n_subjects, C.seq_len, C.action_dim), dtype=np.float32)
    bldi = np.zeros((n_subjects, C.seq_len, C.bldi_dim), dtype=np.float32)

    for step in range(C.seq_len):
        x_t = stimulus[:, step, :]
        hidden = np.tanh(
            np.einsum("ni,nji->nj", x_t, params["weight_ih"])
            + np.einsum("ni,nji->nj", hidden, params["weight_hh"])
            + params["bias_ih"]
            + params["bias_hh"]
        ).astype(np.float32)

        readout = (
            np.einsum("ni,noi->no", hidden, params["weight_out"])
            + params["bias_out"]
        ).astype(np.float32)

        action_logits = 7.5 * readout[:, : C.action_dim]
        action_prob = sigmoid(action_logits)
        action[:, step, :] = (action_prob > 0.5).astype(np.float32)

        smooth_bldi = 0.90 * np.tanh(readout[:, C.action_dim :]) + 0.10 * hidden[:, : C.bldi_dim]
        bldi[:, step, :] = smooth_bldi + 0.001 * rng.normal(size=smooth_bldi.shape).astype(np.float32)

    return action, bldi.astype(np.float32)


def generate_split(
    rng: np.random.Generator,
    n_subjects: int,
    split_name: str,
    decoder: np.ndarray,
    base: dict,
    basis: dict,
):
    subject_code, bld = make_subject_code_and_bld(rng, n_subjects, decoder)
    stimulus = make_structured_stimulus(rng, n_subjects)
    params = subject_specific_params(subject_code, base, basis)
    action, bldi = simulate_subject_specific_rnn(stimulus, params, rng)

    subject_ids = np.array([f"{split_name}_subject_{index:03d}" for index in range(n_subjects)])
    return {
        "bld": bld,
        "stimulus": stimulus,
        "action": action,
        "bldi": bldi,
        "subject_ids": subject_ids,
    }


def main():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fix_random(C.seed)
    rng = np.random.default_rng(C.seed)

    decoder, base, basis = init_teacher_parameters(rng)
    train_split = generate_split(rng, C.train_size, "train", decoder, base, basis)
    test_split = generate_split(rng, C.test_size, "test", decoder, base, basis)

    action_labels = np.array([f"action_{index}" for index in range(C.action_dim)])
    bldi_labels = np.array([f"bldi_region_{index:02d}" for index in range(C.bldi_dim)])
    output_labels = np.concatenate([action_labels, bldi_labels])

    dataset_path = C.DATA_DIR / C.dataset_filename
    np.savez_compressed(
        dataset_path,
        train_bld=train_split["bld"],
        train_stimulus=train_split["stimulus"],
        train_action=train_split["action"],
        train_bldi=train_split["bldi"],
        train_subject_ids=train_split["subject_ids"],
        test_bld=test_split["bld"],
        test_stimulus=test_split["stimulus"],
        test_action=test_split["action"],
        test_bldi=test_split["bldi"],
        test_subject_ids=test_split["subject_ids"],
        action_labels=action_labels,
        bldi_labels=bldi_labels,
        output_labels=output_labels,
    )

    metadata = {
        "seed": C.seed,
        "train_size": C.train_size,
        "test_size": C.test_size,
        "seq_len": C.seq_len,
        "stimulus_dim": C.stimulus_dim,
        "action_dim": C.action_dim,
        "bldi_dim": C.bldi_dim,
        "bld_dim": C.bld_dim,
        "hidden_size": C.hidden_size,
        "hypnet_mid1": C.hypnet_mid1,
        "hypnet_mid2": C.hypnet_mid2,
        "teacher_latent_dim": LATENT_DIM,
        "dataset_file": C.dataset_filename,
    }
    with open(C.DATA_DIR / C.metadata_filename, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    print(f"Saved tiny dataset to: {dataset_path}")
    print(
        "Shapes: "
        f"train_bld={train_split['bld'].shape}, "
        f"train_stimulus={train_split['stimulus'].shape}, "
        f"train_action={train_split['action'].shape}, "
        f"train_bldi={train_split['bldi'].shape}"
    )
    print(
        "Shapes: "
        f"test_bld={test_split['bld'].shape}, "
        f"test_stimulus={test_split['stimulus'].shape}, "
        f"test_action={test_split['action'].shape}, "
        f"test_bldi={test_split['bldi'].shape}"
    )
    print(f"Train action positive rate: {train_split['action'].mean():.4f}")
    print(f"Test action positive rate: {test_split['action'].mean():.4f}")


if __name__ == "__main__":
    main()
