import argparse
from src.utils.common import load_config, load_run
from src.data import load_data
import os

parser = argparse.ArgumentParser(description='Input configs')
parser.add_argument('--config', type=str, required=True, help='config file path')
parser.add_argument('--infer', default=None)

def main(): 
    cfg = load_config([args.config])
    if args.infer is not None:
        cfg.wandb.run_name = args.infer
        print("Infering with run name: ", args.infer)
        
    model, cfg, trainer, feat_extractor = load_run(cfg)
    loaders = load_data(cfg, feat_extractor)
    trainer(cfg, model, loaders, config_paths=[args.config])

if __name__ == '__main__':
    args = parser.parse_args()
    main()    
    