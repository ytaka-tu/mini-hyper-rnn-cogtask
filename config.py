from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "fig"


seed = 4

# Dataset sizes
train_size = 640
test_size = 64
seq_len = 48
stimulus_dim = 9
action_dim = 3
bldi_dim = 4
bld_dim = 64

# Model sizes
hidden_size = 16
hypnet_mid1 = 64
hypnet_mid2 = 64
dropout = 0.0

# Optimization
batch_size = 32
lambda_act = 1.0
lambda_bldi = 1.0
lr = 3e-4
epochs = 120
weight_decay = 0.0
grad_clip = 0.05
val_fraction = 0.2

# Runtime
device = "cpu"

# Filenames
dataset_filename = "tiny_hyper_rnn_dataset.npz"
metadata_filename = "dataset_metadata.json"
best_checkpoint = "checkpoint_best_val.pt"
last_checkpoint = "last_epoch.pt"
loss_plot = "loss.png"
loss_history = "loss_history.json"
