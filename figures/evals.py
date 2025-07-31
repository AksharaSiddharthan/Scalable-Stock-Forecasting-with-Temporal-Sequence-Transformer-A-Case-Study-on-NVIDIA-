import torch
from torch.utils.data import DataLoader, TensorDataset
from models import Non_AR_TST
from train import *
from dataset import DataSheet
import json

# Load model config
with open("config.json") as f:
    config = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model from config
model = Non_AR_TST.from_config("config.json").to(device)

# Load dataset
data = DataSheet(
    path=r"C:\Users\aksha\OneDrive\Documents\BDM PROJECT\nvidia_stock.csv",
    features=["OPEN", "CLOSE", "HIGH", "LOW"],
    target=["OPEN"],
    standardize=True,
    seq_len=14,
    pred_len=14,
    pos="abs",
    split_type="stratified",
    period_1_end="06-03-2020",
    period_2_end="01-01-2023",
    train_test_val_split=(0.4, 0.3, 0.3)
)

batch_size = 8

# Create DataLoaders
train_loader = DataLoader(TensorDataset(*data.train_split()), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(TensorDataset(*data.validate_split()), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(*data.test_split()), batch_size=batch_size, shuffle=False)

# Train and evaluate
trained_model = train(model, train_loader, val_loader, test_loader, config, device)
metrics = evaluate_model(trained_model, train_loader, val_loader, test_loader, data, device)
plot_forecast(trained_model, train_loader, val_loader, test_loader, data, seq_len=data.seq_len, pred_len=data.pred_len, device=device)
