# Mini Hyper-RNN Reproduction

This repository contains a lightweight, self-contained reproduction of the essential training and testing logic used in the published study code for the paper *Digital Twin Brain: Generating Multitask Behavior from Connectomes for Personalized Therapy*, published in [BMEF](https://spj.science.org/doi/10.34133/bmef.0231), based on the original project code released at [OSF](https://osf.io/zuqf6/).

The original research code uses restricted data and large-scale inputs that are not practical to redistribute or run on modest hardware. This repository keeps the core modeling logic the same, but replaces the original data with small synthetic simulation data and reduces model size so that the full workflow can be reproduced at low computational cost.

## What This Reproduces

This mini reproduction preserves the essential logic of the original workflow:

1. `bld` is used as subject-level input to a hypernetwork.
2. The hypernetwork generates subject-specific parameters for an `RNNCell + Linear` main network.
3. `stimulus` is fed into the main network in open-loop form.
4. The model jointly predicts `action` and `bldi`.
5. Training minimizes `BCE(action) + MSE(bldi)`.
6. Testing runs `open_loop`, computes prediction errors, and plots `output` and `target` together.

## Variable Meanings

- `bld`: subject-level rsFCM features
- `stimulus`: time-varying task input sequence
- `action` / `act`: action labels predicted at each time step
- `bldi`: BOLD signal targets predicted at each time step

## What Is Different From The Original Study Code

The following parts were intentionally reduced or replaced:

- data size
- model size
- sequence length
- hidden size
- number of epochs and other hyperparameters
- original restricted data replaced by small synthetic simulation data

The core train/test logic, however, follows the same structure as the original code.

## Directory Layout

- `config.py`: small-scale experimental settings
- `src/model.py`: hypernetwork and end-to-end model
- `src/data.py`: dataset loading utilities
- `src/utils.py`: random seed handling, dataloader split, training loop, loss plotting
- `scripts/make_tiny_dataset.py`: synthetic dataset generation
- `scripts/train.py`: training entry point
- `scripts/test.py`: testing, metric calculation, rsFCM contribution check, and plotting
- `data/raw/`: generated tiny dataset
- `outputs/`: checkpoints, loss curves, metrics, and figures

## How To Run

Run the following commands from this folder:

```bash
pip install -r requirements.txt
python scripts/make_tiny_dataset.py
python scripts/train.py
python scripts/test.py
```

## Requirements

This code was prepared and tested with the following environment:

- `Python 3.10.13`
- `numpy 1.23.2`
- `matplotlib 3.9.2`
- `torch 1.13.1`
- `functorch 1.13.1`

The pinned package versions are listed in `requirements.txt`. Note that the runtime environment used here reported `torch 1.13.1+cu117` and `functorch 1.13.1+cu117`; the pinned versions in `requirements.txt` correspond to the same major/minor tested versions.

## Training Logic

`scripts/train.py` loads the training split, divides it into train/validation subsets, builds the hypernetwork model, and trains it with Adam.

At each training step:

1. `bld`, `stimulus`, `action`, and `bldi` are loaded.
2. The model predicts `action` and `bldi` from `bld` and `stimulus`.
3. The loss is computed as binary cross-entropy for `action` plus mean squared error for `bldi`.
4. Gradients are backpropagated and the parameters are updated.
5. Validation loss is evaluated every epoch.
6. The best validation checkpoint is saved as `outputs/checkpoint_best_val.pt`.

## Testing Logic

`scripts/test.py` loads the best validation checkpoint and evaluates the model on the test split.

It reports:

- `BOLD MSE`
- `act BCE`
- `act ACC`

It also generates subject-wise plots of prediction versus target and saves them in `outputs/fig/`.

In addition, `scripts/test.py` includes an rsFCM contribution check:

- one evaluation uses the correctly matched `bld` for each test subject
- another evaluation uses shuffled, mismatched `bld`
- the resulting difference in `action` and `bldi` prediction accuracy is saved in `outputs/test_metrics.json`

This provides a direct test of whether rsFCM (`bld`) contributes to prediction.

## Main Outputs

- `data/raw/tiny_hyper_rnn_dataset.npz`: synthetic shared dataset
- `outputs/checkpoint_best_val.pt`: best validation checkpoint
- `outputs/last_epoch.pt`: last-epoch checkpoint
- `outputs/loss.png`: training and validation loss curves
- `outputs/train_summary.json`: training summary
- `outputs/test_metrics.json`: aligned vs shuffled-rsFCM test metrics
- `outputs/fig/pred_vs_target_*.png`: subject-wise prediction plots

## Current Small-Scale Settings

The current default settings in `config.py` are:

- `train_size = 640`
- `test_size = 64`
- `seq_len = 48`
- `bld_dim = 64`
- `action_dim = 3`
- `bldi_dim = 4`
- `hidden_size = 16`
- `epochs = 120`
- `batch_size = 32`

These values can be adjusted if an even smaller or slightly larger demonstration is needed.
