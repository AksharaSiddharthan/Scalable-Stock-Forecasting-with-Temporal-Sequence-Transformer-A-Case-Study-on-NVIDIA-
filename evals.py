import torch 
from torch.utils.data import DataLoader, TensorDataset
from models import Non_AR_TST, LearnablePositionEncoding, SinusoidalPositionalEncoding, MultiHeadAttention
from train import train, evaluate_model, plot_forecast
from datasets import DataSheet
import json
import pandas as pd

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Separate model and training config
model_keys = [
    "in_dim", "out_dim", "embed_dim", "ffn_dim", "num_heads",
    "num_layers", "seq_len", "pred_len", "pe", "attn", "nonlinearity", "dropout_p"
]
model_config = {k: config[k] for k in model_keys}
training_config = {k: config[k] for k in config if k not in model_keys}

# Map encodings
encoding_map = {
    "LearnablePositionEncoding": LearnablePositionEncoding,
    "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
}
attention_map = {
    "MultiHeadAttention": MultiHeadAttention,
}

# Validate encoding/attention strings
if model_config["pe"] not in encoding_map:
    raise ValueError(f"Unknown positional encoding: {model_config['pe']}")
if model_config["attn"] not in attention_map:
    raise ValueError(f"Unknown attention type: {model_config['attn']}")

model_config["pe"] = encoding_map[model_config["pe"]]
model_config["attn"] = attention_map[model_config["attn"]]

# Initialize model
model = Non_AR_TST(**model_config).to(device)

# Load dataset
df = pd.read_csv("nvidia_stock.csv")
print("Available columns:", df.columns.tolist())

data = DataSheet(
    path="nvidia_stock.csv",
    features=["Open", "Close", "High", "Low", "Volume"],
    target=["Open"],
    standardize=True,
    seq_len=model_config["seq_len"],
    pred_len=model_config["pred_len"],
    pos="abs",
    split_type="stratified",
    period_1_end="06-03-2020",
    period_2_end="01-01-2023",
    train_test_val_split=(0.4, 0.3, 0.3)
)

batch_size = training_config.get("batch_size", 8)

train_loader = DataLoader(TensorDataset(*data.train_split()), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(TensorDataset(*data.validate_split()), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(*data.test_split()), batch_size=batch_size, shuffle=False)

# Train and evaluate
trained_model = train(model, train_loader, val_loader, test_loader, training_config, device)
metrics = evaluate_model(trained_model, train_loader, val_loader, test_loader, data, device)
plot_forecast(trained_model, train_loader, val_loader, test_loader, data, seq_len=data.seq_len, pred_len=data.pred_len, device=device)
