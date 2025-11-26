import torch
import torchaudio
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.utils import data_utils
from src.utils.run_utils import resample_audio

from src.utils.logger import logger

def preprocess_spokenwoz(cfg):
    """
    Preprocess SpokenWOZ dataset to the required format
    Args:
        cfg (edict): Configuration dictionary
    Saves:
        Preprocessed data in the specified save path
    """
    data_folder = cfg.data.datasets.spokenwoz.raw_path
    train_audios = os.path.join(data_folder, "audio_5700_train_dev")
    val_audios = os.path.join(data_folder, "audio_5700_train_dev")
    test_audios = os.path.join(data_folder, "audio_5700_test")
    train_dev_json = os.path.join(data_folder, "text_5700_train_dev/data.json")
    test_json = os.path.join(data_folder, "text_5700_test/data.json")
    val_ids = os.path.join(data_folder, "text_5700_train_dev/valListFile.json")
    
    train_val_data = data_utils.load_data_from_file(train_dev_json, reader="json")
    val_ids = data_utils.load_data_from_file(val_ids, reader="txt")
    train_data = {k: v for k, v in train_val_data.items() if k not in val_ids}
    val_data = {k: v for k, v in train_val_data.items() if k in val_ids}
    test_data = data_utils.load_data_from_file(test_json, reader="json")

    for mode, data, audios in zip(
        cfg.data.datasets.spokenwoz.modes,
        [train_data, val_data, test_data],
        [train_audios, val_audios, test_audios],
    ):  
        save_path = cfg.data.save_paths.preprocessed_data_path.strip().format(dataset="spokenwoz", mode=mode)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            if not cfg.data.datasets.spokenwoz.override_preprocessed_data:
                logger.info(f"Preprocessed data already exists for spokenwoz, skipping...")
                continue
        preprocessed_json = {}  
        for key in data:
            audio_path = os.path.join(audios, key + ".wav")
            preprocessed_json[key] = {
                "audio_filepath": audio_path,
            }
            segments = []
            for turn_data in data[key]["log"]:
                segments.append({
                    "start_time": turn_data["words"][0]["BeginTime"] / 1000.0,
                    "end_time": turn_data["words"][-1]["EndTime"] / 1000.0,
                    "turn": turn_data["tag"],
                    "text": turn_data["text"],
                })
            preprocessed_json[key]["segments"] = segments
        data_utils.write_data_to_file(preprocessed_json, save_path, writer="json")