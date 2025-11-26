import argparse
from src.utils.common import load_config, load_run
from src.data import load_data
import os
##disable wandb online
os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser(description='Input configs')
parser.add_argument('--exp_config', type=str, default='configs/infer_reformer.yaml', help='config file path')
parser.add_argument("--root_path", default="/mnt/matylda4/")
parser.add_argument('--infer', default=None)

def main(): 
    cfg = load_config([args.exp_config])
    if args.infer is not None:
        cfg.wandb.run_name = args.infer
        print("Infering with run name: ", args.infer)
        
    if hasattr(cfg.data, 'root_path'):
        cfg.data.root_path = args.root_path
    if hasattr(cfg, "infer_params"):
        if hasattr(cfg.infer_params, 'root_path'):
            cfg.infer_params.root_path = args.root_path
        
    model, cfg, trainer, feat_extractor = load_run(cfg)
    loaders = load_data(cfg, feat_extractor)
    trainer(cfg, model, loaders, config_paths=[args.exp_config, args.extra_config], break_mode=args.break_mode)

if __name__ == '__main__':
    args = parser.parse_args()
    main()    
    