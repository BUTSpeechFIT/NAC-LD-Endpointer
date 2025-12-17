import torch
import torchaudio
import os
from src.utils import textgrid
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.utils import data_utils
from src.utils.run_utils import resample_audio
from pathlib import Path
from src.utils.logger import logger

def preprocess_humdial(cfg):
    data_folder = cfg.data.datasets.humdial.raw_path
    train_folder = os.path.join(data_folder, "HD-Track2/HD-Track2-train")
    dev_folder = os.path.join(data_folder, "HD-Track2/HD-Track2-dev")
    
    for mode, folder in zip(
        ["train", "val"],
        [train_folder, dev_folder],
    ):  
        save_path = cfg.data.save_paths.preprocessed_data_path.strip().format(dataset="humdial", mode=mode)
        save_path = os.path.join(cfg.data.save_paths.dump, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            if "humdial" not in cfg.data.override_preprocessed_data:
                continue
        audio_files, text_files = [], []
        preprocessed_json = {} 
        ##collect all audio and text files for specified languages
        for language in cfg.data.datasets.humdial.languages:
            mode_ = "dev" if mode == "val" else "train"
            folder_lang = os.path.join(folder, f"HD-Track2-{mode_}-" + language)
            lang_audio_files = data_utils.get_files(Path(folder_lang), extension='.wav')
            lang_text_files = data_utils.get_files(Path(folder_lang), extension='.TextGrid')
            audio_files.extend(lang_audio_files)
            text_files.extend(lang_text_files)
        ##now we need to extract data from text grid files
        for text_file in tqdm(text_files, desc=f"Preprocessing HumDial {mode} data"):
            key = text_file.stem
            corresponding_audio_file = text_file.with_suffix('.wav')
            parent_folder_for_text_file = str(text_file.parent)
            id = parent_folder_for_text_file.split('/')[-1].replace(' ', "__") + '_' + key
            interval = textgrid.TextGrid.load(str(text_file))
            preprocessed_json[id] = {"audio_filepath": str(corresponding_audio_file), "segments": []}
            for tier in interval:
                segments = tier.simple_transcript
                for segment in segments:
                    if segment[2].strip() == "": continue
                    start, end, text = segment
                    start, end = round(float(start), 4), round(float(end),4)
                    label_data = {
                        "turn": "user",
                        "text": text,
                        "start_time": start,
                        "end_time": end
                    }
                    preprocessed_json[id]["segments"].append(label_data)
        data_utils.write_data_to_file(preprocessed_json, save_path, writer="json")