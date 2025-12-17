import torch
from src.utils.data_utils import load_data_from_file, load_full_audio
from src.utils.run_utils import resample_audio
import os
import shutil
import torchaudio
from tqdm import tqdm
from pathlib import Path

class DataLoaderHF(torch.utils.data.Dataset):
    def __init__(self, folder, audio_folder, resampled_audio_folder, reset_resample, sr, resample_sr, num_samples=None):    
        self.eval_data, self.duplicate_keys, self.key2time = {}, {}, {}
        files = sorted(os.listdir(folder))
        os.makedirs(resampled_audio_folder, exist_ok=True)
        if reset_resample:
            shutil.rmtree(resampled_audio_folder)
        audios = [f for f in os.listdir(audio_folder) if not f.startswith(".")]
        num_audios = len(audios)
        resampled_audios = [f for f in os.listdir(resampled_audio_folder) if not f.startswith(".")]
        num_resampled_audios = len(resampled_audios)
        if num_audios != num_resampled_audios:
            for audio in tqdm(audios, desc="Resampling"):
                audio_path = os.path.join(audio_folder, audio)
                y = load_full_audio(audio_path, sr)
                y_16Khz = resample_audio(y, sr, resample_sr).unsqueeze(0)
                save_path = os.path.join(resampled_audio_folder, audio)
                torchaudio.save(save_path, y_16Khz, resample_sr)
        audio_folder = resampled_audio_folder
        if num_samples is not None:
            files = files[:num_samples]
        for json_file in files:
            json_data = load_data_from_file(os.path.join(folder, json_file))
            self.parse_data(json_data, Path(json_file).stem, audio_folder)
            # break
        self.keys = list(self.eval_data.keys())#[16255:]
        self.resample_sr = resample_sr
        
    def parse_data(self, json_data, file_name, audio_folder):
        audio_file_path = os.path.join(audio_folder, file_name + '.wav')
        for turn_id in json_data:
            start = json_data[turn_id]['start']
            ref_label_timestamps = json_data[turn_id]['ref']
            text = json_data[turn_id]['text']
            ref_keyname = "___".join((file_name, turn_id, "ref"))
            assert ref_keyname not in self.eval_data, ref_keyname
            duration = (ref_label_timestamps[-1] - ref_label_timestamps[0]) / 16000
            if duration > 30:
                continue
            self.eval_data[ref_keyname] = {
                'audio_file': audio_file_path,
                'timestamps': ref_label_timestamps,
                'text': text,
                'start': start
            }
            
            self.key2time[ref_keyname] = ref_label_timestamps
            prev_timestamps, prev_keyname = None, None
            for threshold in json_data[turn_id]['hyp']:
                threshold_timestamps = json_data[turn_id]['hyp'][threshold]
                duration = (threshold_timestamps[-1] - threshold_timestamps[0]) / 16000
                if duration > 30:
                    continue
                keyname = "___".join((file_name, turn_id, "hyp", threshold))
                assert keyname not in self.eval_data
                self.key2time[keyname] = threshold_timestamps
                if prev_timestamps is not None:
                    
                    if prev_timestamps == threshold_timestamps:
                        self.duplicate_keys[keyname] = prev_keyname
                        continue      
                    
                    else:
                        if threshold_timestamps == ref_label_timestamps:
                            self.duplicate_keys[keyname] = ref_keyname
                            continue
                        
                prev_timestamps = threshold_timestamps  
                prev_keyname = keyname
                    
                
                self.eval_data[keyname] = {
                    'audio_file': audio_file_path,
                    'timestamps': threshold_timestamps,
                    'text': text,
                    'start': start
                }
            # break
    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        key = self.keys[idx]
        audio_path = self.eval_data[key]['audio_file']
        start = self.eval_data[key]['start']
        begin, end = self.eval_data[key]['timestamps']
        text = self.eval_data[key]['text']
        
        y = load_full_audio(audio_path, self.resample_sr)
        y = y[round(start*self.resample_sr):]
        # y_16Khz = resample_audio(y, self.sr, self.resample_sr)
        y_slice = y[begin:end]
        # print(y_slice.shape, y_slice.shape[0] / 16000, key)
        return {"audio":y_slice.numpy(), "key": key, "original_text": text}