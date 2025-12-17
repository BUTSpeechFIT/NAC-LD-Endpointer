import torchaudio
import torch.nn as nn
import torch
from src.utils.logger import logger

class Model(nn.Module):
    def __init__(
        self, **kwargs
    ):
        super(Model, self).__init__()
        self.mel_embed = None
        self.system_embed = None
        self.loss_fn = None
        if kwargs["loss_fn"] == "frame_level_cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            if hasattr(kwargs, "loss_weights"):
                logger.info(f"Using loss weights - {kwargs['loss_weights']}")
                self.loss_fn = nn.CrossEntropyLoss(weight=kwargs["loss_weights"], ignore_index=-1)
                
    def forward(self, x):
        raise NotImplementedError
    
    def infer(self, x):
        return NotImplementedError

    def infer_ar(self, x):
        return NotImplementedError

    def make_lstm(self, kwargs):
        model = nn.LSTM(
            input_size=kwargs["input_size"] if "project" not in kwargs else kwargs["project"],
            hidden_size=kwargs["hidden_size"],
            num_layers=kwargs["num_layers"],
            batch_first=True,
            dropout=kwargs["dropout"],
            bidirectional=kwargs["bidirectional"],
        )
        return model

    def init_lstm_hidden(self, batch_size, device):
        h_o = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size)
        c_0 = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size)
        return h_o.to(device), c_0.to(device)
        
    def loss(self, pred, gnd, delay_frames=None):
        pred = pred.permute(0, 2, 1)
        loss = self.loss_fn(pred, gnd)
        if delay_frames is not None:
            pred = pred[:, :, delay_frames:]
            gnd = gnd[:, delay_frames:]
        pred_label = torch.argmax(pred, dim=1)
        accuracy = (pred_label == gnd).float().mean()
        pred_label = pred_label.view(-1).detach().cpu().tolist()
        gnd_label = gnd.contiguous().view(-1).detach().cpu().tolist()
        loss = {"total": loss, "accuracy": accuracy}
        return loss, (gnd_label, pred_label)
    
    