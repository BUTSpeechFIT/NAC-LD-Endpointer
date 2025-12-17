import torchaudio
import torch.nn as nn
import torch
from src.utils.logger import logger
from .model import Model

class LSTM_Model(Model):
    def __init__(
        self, 
        use_system_embed=False,
        **kwargs
    ):
        super(LSTM_Model, self).__init__(**kwargs)
        self.model = self.make_lstm(kwargs)
        self.linear = nn.Linear(kwargs["hidden_size"], kwargs["output_size"])
        if use_system_embed:
            self.system_embed = nn.Embedding(2, kwargs["input_size"])
    
    def forward(self, x, h=None, c=None, init_hidden=False, system_ids=None):
        if init_hidden:
            h, c = self.init_lstm_hidden(x.size(0), x.device)
        x = x.permute(0, 2, 1)
        if system_ids is not None and self.system_embed is not None:
            x = x + self.system_embed(system_ids)
        x, (_, _) = self.model(x, (h, c))
        x = self.linear(x)
        return x

    def infer(self, x, h=None, c=None, init_hidden=False):
        assert x.size(0) == 1, "Inference only supports batch size of 1"
        if init_hidden:
            h, c = self.init_lstm_hidden(x.size(0), x.device)
        x = x.permute(0, 2, 1)
        x, (_, _) = self.model(x, (h, c))
        x = self.linear(x)
        return x

    def infer_ar(self, x, h=None, c=None, init_hidden=False):
        assert x.size(0) == 1, "Inference only supports batch size of 1"
        num_decode_steps = x.size(2)
        if init_hidden:
            h, c = self.init_lstm_hidden(x.size(0), x.device)
        full_output = None
        for decode_idx in range(num_decode_steps):
            x_ = x[:, :, decode_idx].unsqueeze(2)
            x_ = x_.permute(0, 2, 1)
            x_, (h, c) = self.model(x_, (h, c))
            out = self.linear(x_)
            if full_output is None:
                full_output = out
            else:
                full_output = torch.cat([full_output, out], dim=1)
        return full_output
    