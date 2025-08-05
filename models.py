import torch  
import torch.nn as nn
import torch.nn.functional as F
from modules import Projection, SinusoidalPositionalEncoding, LearnablePositionEncoding, MultiHeadAttention, Decoder, FFN
import json

# === MAPPINGS ===
ENCODING_MAP = {
    "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
    "LearnablePositionEncoding": LearnablePositionEncoding,
}

ATTENTION_MAP = {
    "MultiHeadAttention": MultiHeadAttention,
}

NONLINEARITY_MAP = {
    "ReLU": F.relu,
    "GELU": F.gelu,
    "SiLU": F.silu,
    "LeakyReLU": F.leaky_relu
}


# === Forecasting Transformer ===
class Non_AR_TST(nn.Module):
    """
    Non-Autoregressive Transformer model for Time Series Forecasting.
    """

    def __init__(self, in_dim: int, out_dim: int, embed_dim: int,
                 ffn_dim: int, num_heads: int, num_layers: int,
                 seq_len: int, pred_len: int, dropout_p=0,
                 pe=SinusoidalPositionalEncoding, attn=MultiHeadAttention, nonlinearity=torch.relu):
        
        super(Non_AR_TST, self).__init__()
        self.embed_dim = embed_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.nonlinearity = nonlinearity
        self.dropout_p = dropout_p

        self.input_proj = Projection(self.embed_dim, self.in_dim)
        self.pe = pe(self.embed_dim, dropout_p) if pe != LearnablePositionEncoding else pe(seq_len + pred_len, self.embed_dim, dropout_p)
        self.attn = attn

        self.layers = nn.ModuleList([
            Decoder(
                self.embed_dim,
                self.num_heads,
                self.ffn_dim,
                self.attn,
                self.nonlinearity,
                self.dropout_p,
            ) for _ in range(self.num_layers)
        ])

        self.out_proj = nn.Linear(self.embed_dim, self.out_dim)

    def attn_mask(self, device):
        seq_len = self.seq_len
        pred_len = self.pred_len
        total_len = seq_len + pred_len
        mask = torch.zeros(total_len, total_len, device=device)
        mask[:seq_len, :seq_len] = 1
        mask[seq_len:, :seq_len] = 1
        return mask.bool()

    def forward(self, x, timestamps):
        mask = self.attn_mask(x.device)
        batch_size = x.size(0)
        future_placeholder = torch.zeros(batch_size, self.pred_len, x.size(-1), device=x.device)
        full_input = torch.cat([x, future_placeholder], dim=1)

        extended_timestamps = torch.cat(
            [timestamps, timestamps[:, -1:] + torch.arange(1, self.pred_len + 1, device=x.device).float().unsqueeze(0)],
            dim=1
        )

        x = self.input_proj(full_input)
        x = self.pe(x, extended_timestamps)
        for layer in self.layers:
            x = layer(x, mask)
        x = x[:, -self.pred_len:, :]
        x = self.out_proj(x)
        return x

    def export_config(self, path):
        with open(path + "/config.json", mode="w+") as f:
            config = {
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "embed_dim": self.embed_dim,
                "ffn_dim": self.ffn_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "pe": self.pe.__class__.__name__,
                "attn": self.attn.__name__,
                "nonlinearity": self.nonlinearity.__name__ if hasattr(self.nonlinearity, "__name__") else str(self.nonlinearity),
                "dropout_p": self.dropout_p
            }
            json.dump(config, f, indent=4)

    @classmethod
    def from_config(cls, config_file_path):
        with open(config_file_path) as f:
            config = json.load(f)
        config["pe"] = ENCODING_MAP[config["pe"]]
        config["attn"] = ATTENTION_MAP[config["attn"]]
        config["nonlinearity"] = NONLINEARITY_MAP[config["nonlinearity"]]
        return cls(**config)

    def save_checkpoint(self, path):
        raise NotImplementedError("Checkpoint saving is not implemented yet.")


# === Time Series Classification Variant ===
class Non_AR_TSCT(nn.Module):
    """
    Non-Autoregressive Transformer model for Time Series Classification.
    """

    def __init__(self, in_dim: int, out_classes: int, embed_dim: int,
                 ffn_dim: int, num_heads: int, num_layers: int,
                 seq_len: int, pred_len: int, dropout_p=0,
                 pe=SinusoidalPositionalEncoding, attn=MultiHeadAttention, nonlinearity=torch.relu):

        super(Non_AR_TSCT, self).__init__()
        self.embed_dim = embed_dim
        self.in_dim = in_dim
        self.out_dim = out_classes
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.nonlinearity = nonlinearity
        self.dropout_p = dropout_p

        self.input_proj = Projection(self.embed_dim, self.in_dim)
        self.pe = pe(self.embed_dim, dropout_p) if pe != LearnablePositionEncoding else pe(seq_len + pred_len, self.embed_dim, dropout_p)
        self.attn = attn

        self.layers = nn.ModuleList([
            Decoder(
                self.embed_dim,
                self.num_heads,
                self.ffn_dim,
                self.attn,
                self.nonlinearity,
                self.dropout_p,
            ) for _ in range(self.num_layers)
        ])

        self.out_proj = nn.Linear(self.embed_dim, self.out_dim)
        self.head_activation = nn.Sigmoid() if out_classes == 1 else nn.Softmax(dim=-1)

    def attn_mask(self, device):
        seq_len = self.seq_len
        pred_len = self.pred_len
        total_len = seq_len + pred_len
        mask = torch.zeros(total_len, total_len, device=device)
        mask[:seq_len, :seq_len] = 1
        mask[seq_len:, :seq_len] = 1
        return mask.bool()

    def forward(self, x, timestamps):
        mask = self.attn_mask(x.device)
        batch_size = x.size(0)
        future_placeholder = torch.zeros(batch_size, self.pred_len, x.size(-1), device=x.device)
        full_input = torch.cat([x, future_placeholder], dim=1)

        extended_timestamps = torch.cat(
            [timestamps, timestamps[:, -1:] + torch.arange(1, self.pred_len + 1, device=x.device).float().unsqueeze(0)],
            dim=1
        )

        x = self.input_proj(full_input)
        x = self.pe(x, extended_timestamps)
        for layer in self.layers:
            x = layer(x, mask)
        x = x[:, -self.pred_len:, :]
        x = self.out_proj(x)
        x = self.head_activation(x)
        return x

    def export_config(self, path):
        with open(path + "/config.json", mode="w+") as f:
            config = {
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "embed_dim": self.embed_dim,
                "ffn_dim": self.ffn_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "pe": self.pe.__class__.__name__,
                "attn": self.attn.__name__,
                "nonlinearity": self.nonlinearity.__name__ if hasattr(self.nonlinearity, "__name__") else str(self.nonlinearity),
                "dropout_p": self.dropout_p
            }
            json.dump(config, f, indent=4)

@classmethod
def from_config(cls, config_path):
    with open(config_path) as f:
        config = json.load(f)

    model_keys = ['in_dim', 'out_dim', 'embed_dim', 'ffn_dim', 'num_heads',
                  'num_layers', 'seq_len', 'pred_len', 'pe', 'attn',
                  'nonlinearity', 'dropout_p']

    model_config = {k: v for k, v in config.items() if k in model_keys}
    return cls(**model_config)


    def save_checkpoint(self, path):
        raise NotImplementedError("Checkpoint saving is not implemented yet.")
