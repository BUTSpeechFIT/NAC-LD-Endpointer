import wandb
import matplotlib.pyplot as plt

class WandbLogger():
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg is not None:
            self.run = wandb.init(reinit=True,
                                name=cfg.run_name,
                                project=cfg.wandb.wandb_project,
                                config=cfg,
                                mode="online" if cfg.wandb.use_wandb else "disabled",
                                )
    def log(self, dct):
        wandb.log(dct)

    def log_plots(self, plt, name):
        # raise NotImplementedError("log_plots not implemented")
        wandb.log({name: wandb.Image(plt)})

    def summary(self, dct):
        for key in dct:
            wandb.run.summary[key] = dct[key]

    def end_run(self):
        self.run.finish()

    def log_audio(Self, aud, name, sample_rate=22050):
        wandb.log({name: wandb.Audio(aud,  sample_rate=sample_rate)})