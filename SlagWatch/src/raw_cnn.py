"""A binary detector for one raw SSFD window, with shared patch encoder."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"experiments/cnn_v1"
CFG = json.loads((RUN/"protocol.json").read_text(encoding="utf8"))


class PatchCNN(nn.Module):
    def __init__(self, model_id="C0"):
        super().__init__()
        if model_id not in ("C0","C1"):
            raise ValueError("Unknown CNN configuration")
        self.model_id = model_id
        self.register_buffer("axis_mean",torch.zeros(3))
        self.register_buffer("axis_std",torch.ones(3))
        self.encoder = nn.Sequential(
            nn.Conv1d(3,32,7,stride=2,padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32,64,5,padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),nn.Dropout(.2),nn.Linear(64,1))
    def forward(self, raw):
        x = (raw-self.axis_mean)/self.axis_std
        if self.model_id == "C1":
            x = x / x.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-6)
        x = x.transpose(1,2)
        # (batch, axes, seven patches, 512) -> (batch * seven, axes, 512)
        x = x.unfold(2,512,448).permute(0,2,1,3).reshape(-1,3,512)
        return self.encoder(x).reshape(-1,7).mean(1)


def tensor_waves(waves):
    return torch.from_numpy(np.ascontiguousarray(waves, dtype=np.float32))


@torch.inference_mode()
def score_batch(net, raw):
    net.eval()
    return torch.cat([net(x).sigmoid() for x in raw.split(CFG["batch_size"])]).cpu().numpy()


def load_net(path):
    ck = torch.load(path, map_location="cpu",weights_only=True)
    net = PatchCNN(ck["model_id"])
    net.load_state_dict(ck["state"])
    return net.eval(), ck
