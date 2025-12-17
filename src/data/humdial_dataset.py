import torch
import torchaudio
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.utils import data_utils
from src.utils.run_utils import resample_audio
from pathlib import Path
from src.utils.logger import logger

class HumDial_dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, feat_extractor=None):
        
        self.cfg = cfg
        processed_labels_save_path = os.path.join(cfg.data.root_path, cfg.data.paths[mode].processed_labels_save_path)
        data_jsons = os.listdir(processed_labels_save_path)
        audio_files = data_utils.get_files(Path(os.path.join(cfg.data.root_path, cfg.data.paths[mode].data_path)), extension=".wav")
        id2audios = {}
        for audio_file in audio_files:
            audio_id = Path(audio_file).stem
            parent_folder = str(Path(audio_file).parent).split('/')[-1].replace(' ', '__')
            id = parent_folder + "_" + audio_id
            id2audios[id] = audio_file
        
        self.keys = set()
        total_skipped_due_to_short_duration, available = 0, 0
        for json_file in tqdm(data_jsons, desc=f"Loading {mode} data"):
            json_id = Path(json_file).stem
            data_json = data_utils.load_data_from_file(os.path.join(processed_labels_save_path, json_file), reader="json")
            data_json, skipped = self.filter_jsons(cfg, data_json, mode, json_file)
            total_skipped_due_to_short_duration += skipped
            available += len(data_json)
            if len(data_json) > 0:
                self.keys.add(json_id)
        logger.info(f"Total skipped / available samples: {total_skipped_due_to_short_duration} / {available}")
        self.audio_ids = id2audios
        if hasattr(cfg.data.paths[mode], "num_samples"):
            if cfg.data.paths[mode].num_samples is not None:
                logger.info(f"Reducing number of {mode} samples to {cfg.data.paths[mode].num_samples}")
                self.keys = sorted(self.keys)[:cfg.data.paths[mode].num_samples]
        self.keys = list(sorted(self.keys)) 
        self.label_mapping = data_utils.get_token_to_id_mapping(cfg)
        
        self.mode = mode
        self.get_audio_feature(feat_extractor)
        if hasattr(cfg.data.audio_params, "dump_feats"):
            if cfg.data.audio_params.dump_feats:
                self.dump_features(cfg)
                
        assert cfg.data.label_params.use_fixed_context_training == True, "Only fixed context training is supported for SpokenWOZ dataset"
        self.max_length = None
        if hasattr(cfg.data, "max_length"):
            logger.info(f"Setting max length to {cfg.data.max_length}")
            self.max_length = cfg.data.max_length
    
    def dump_features(self, cfg):
        location = os.path.join(cfg.data.root_path, cfg.data.audio_params.feat_location[self.mode])
        os.makedirs(location, exist_ok=True)
        num_files = len(os.listdir(location))
        num_audios = len(os.listdir(self.audio_folder))
        if num_files == num_audios:
            self.feat_folder = location
            return
        logger.info(f"Dumping features to {location}")
        for audio in tqdm(os.listdir(self.audio_folder), desc="Dumping features"):
            audio_path = os.path.join(self.audio_folder, audio)
            y = data_utils.load_full_audio(audio_path, self.sr)
            ###extracting for full audio is too memory intensive
            ###so, we extract for 10 sec, take the last 5 sec and concatenate
            ###this is done so that every sample has atleast 5 sec of context
            ###I want chunks such that the last 5 sec of the previous chunk is the first 5 sec of the next chunk
            ###for first chunk, I take the first 5 sec of the audio
            chunk_length = cfg.data.label_params.context_in_sec
            save_length = chunk_length // 2
            num_chunks = int(len(y) // (self.sr * save_length))
            data = {}
            for idx in range(num_chunks):
                ##take chunk_length audio, and concat save_length audio from the end
                start_idx = int(idx * self.sr * save_length)
                end_idx = int(start_idx + self.sr * chunk_length)
                y_chunk = y[start_idx:end_idx]
                feat = self.audio_feature(y_chunk, all_outputs=True)
                for key in self.audio_feature.keys():
                    if key not in data:
                        data[key] = feat[key].squeeze().cpu()
                    else:
                        data[key] = torch.cat((data[key], feat[key].squeeze().cpu()[int(cfg.data.audio_params.freq * save_length):, :]), dim=0)
                        
                    
            save_path = os.path.join(location, audio.replace(".wav", ".pt"))
            torch.save(data, save_path)
    
    def filter_jsons(self, cfg, data_json, mode, json_file):
        
        skipped = 0
        if not self.cfg.data.label_params.use_fixed_context_training:
            raise NotImplementedError("Only fixed context training is supported for SpokenWOZ dataset")
        # for label in data_json:
        segments = []
        for entry in data_json:
            # for label in entry:
            
            final_end_time = entry[-1]["end_time"]
            max_start_time = final_end_time - cfg.data.label_params.context_in_sec - cfg.data.label_params.extra_offset
            turn_start_times = [label["start_time"] for label in entry if label["start_time"] < max_start_time and label["turn"] in ["user", "system"]]
            if len(turn_start_times) == 0:
                skipped += 1
                continue
            segments.append(entry)
        length_filtered_save_path = os.path.join(cfg.data.root_path, cfg.data.paths[mode].length_filtered_save_path)
        save_folder = "_".join((
            length_filtered_save_path,
            str(cfg.data.label_params.context_in_sec),
            str(cfg.data.label_params.extra_offset),
        ))
        os.makedirs(save_folder, exist_ok=True)
        if len(segments) > 0:
            if os.path.exists(save_folder):
                if not cfg.data.override_filtered_data:
                    # logger.logger.info(f"Processed output already exists at {save_folder}. Skipping...")
                    return segments, skipped
            data_utils.write_data_to_file(segments, os.path.join(save_folder, json_file), writer="json")
        return segments, skipped
    
    def audio_feature_v2(self, y):
        stft = torch.stft(
            y, n_fft=self.cfg.data.audio_params.n_fft, 
            hop_length=self.cfg.data.audio_params.hop_length,           
            win_length=self.cfg.data.audio_params.win_length, 
            return_complex=False,
        )
        spectrogram = torch.abs(stft[..., 0])
        mel_frames = self.mel_scale_transform(spectrogram)
        return mel_frames

    def get_audio_feature(self, feat_extractor):
        cfg = self.cfg
        if cfg.data.audio_params.audio_feature == "logmel":
            self.audio_feature = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.data.audio_params.sr,
                n_fft=cfg.data.audio_params.n_fft,
                win_length=cfg.data.audio_params.win_length,
                hop_length=cfg.data.audio_params.hop_length,
                n_mels=cfg.data.audio_params.n_mels,
                power=cfg.data.audio_params.power,
            )
            self.sr = cfg.data.audio_params.sr
        elif cfg.data.audio_params.audio_feature == "logmel-v2":
            self.mel_scale_transform = torchaudio.transforms.MelScale(
                    n_mels=cfg.data.audio_params.n_mels,
                    sample_rate=cfg.data.audio_params.sr,
                    n_stft=cfg.data.audio_params.n_fft // 2 + 1, # n_stft is n_fft // 2 + 1
                )
            self.audio_feature = self.audio_feature_v2
            self.sr = cfg.data.audio_params.sr
                
        else:
            self.audio_feature = feat_extractor
            self.sr = cfg.data.audio_params.sr
            # if not hasattr(cfg.data.audio_params, "target_sr"):
                # cfg.data.audio_params.target_sr = cfg.data.audio_params.sr
            if self.cfg.data.audio_params.sr != self.cfg.data.audio_params.target_sr:
                self.sr = self.cfg.data.audio_params.target_sr
                # num_audios = len(os.listdir(self.audio_folder))
                num_audios = len(self.keys)
                resampled_audios_path = os.path.join(self.cfg.data.root_path, self.cfg.data.paths[self.mode].resampled_audios)  
                target_sr_folder = "_".join((resampled_audios_path, str(self.cfg.data.audio_params.target_sr)))
                if not os.path.exists(target_sr_folder):
                    os.makedirs(target_sr_folder, exist_ok=True)
                num_resampled_audios = len(os.listdir(target_sr_folder))
                # print(num_audios, num_resampled_audios)
                if num_audios == num_resampled_audios:
                    self.audio_folder = target_sr_folder
                    for audio_id in self.audio_ids:
                        audio_path = os.path.join(target_sr_folder, audio_id + ".wav")
                        self.audio_ids[audio_id] = audio_path
                    return
                logger.info(f"Resampling audios to {self.cfg.data.audio_params.target_sr}Hz, saving at {target_sr_folder}") 
                preserve_channels = False
                if hasattr(self.cfg.data, "multi_audio_stream"):
                    preserve_channels = self.cfg.data.multi_audio_stream
                for audio_id in tqdm(self.audio_ids, desc="Resampling"):
                    audio_path = self.audio_ids[audio_id]
                    save_path = os.path.join(target_sr_folder, audio_id + ".wav")
                    self.audio_ids[audio_id] = save_path
                    if os.path.exists(save_path):
                        
                        continue
                    y = data_utils.load_full_audio(audio_path, self.cfg.data.audio_params.sr, preserve_channels=preserve_channels)
                    y_reKhz = resample_audio(y, self.cfg.data.audio_params.sr, self.cfg.data.audio_params.target_sr)
                    if len(y_reKhz.shape) == 1:
                        y_reKhz = y_reKhz.unsqueeze(0)
                    torchaudio.save(save_path, y_reKhz, self.cfg.data.audio_params.target_sr)
                self.audio_folder = target_sr_folder
                
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        length_filtered_save_path = os.path.join(self.cfg.data.root_path, self.cfg.data.paths[self.mode].length_filtered_save_path)
        saved_folder = "_".join((
            length_filtered_save_path,
            str(self.cfg.data.label_params.context_in_sec),
            str(self.cfg.data.label_params.extra_offset),
        ))
        label_data = data_utils.load_data_from_file(os.path.join(saved_folder, key + ".json"), reader="json")
        # print(label_data)
        if self.cfg.data.label_params.use_fixed_context_training:
            fixed_context_labels, start_time, end_time = data_utils.convert_continous_labels_to_fixed_context_frames(self.cfg, label_data, key)    
        # print(fixed_context_labels, start_time, end_time)
        # audio_path = os.path.join(self.audio_folder, key + ".wav")
        audio_path = self.audio_ids[key]
        # print(audio_path)
        preserve_channels = False
        if hasattr(self.cfg.data, "multi_audio_stream"):
            preserve_channels = self.cfg.data.multi_audio_stream
        if hasattr(self.cfg.data, "zero_system"):
            if self.cfg.data.zero_system:
                preserve_channels = True
        y = data_utils.load_audio_segment(audio_path, start_time, end_time, self.sr, preserve_channels=preserve_channels)
        if hasattr(self.cfg.data, "zero_system"):
            if self.cfg.data.zero_system:
                y = y[0]
                preserve_channels = False
        with torch.no_grad():
            yd = y
            if self.cfg.data.audio_params.audio_feature not in ["logmel", "logmel-v2"] and preserve_channels:
                yd = y.numpy()
                yd = [yd[0], yd[1]]
            if hasattr(self.cfg.data.label_params, "add_end_silence"):
                if self.cfg.data.label_params.add_end_silence:
                    silence_after_system_end = self.cfg.data.label_params.silence_dur
                    silence_samples = int(silence_after_system_end * self.sr)
                    if preserve_channels:
                        silence_tensor = torch.zeros((yd.shape[0], silence_samples))
                    else:
                        silence_tensor = torch.zeros(silence_samples)
                    yd = torch.cat((yd, silence_tensor), dim=-1)
            melspec = self.audio_feature(yd)
        audio_len = melspec.shape[-1]
        if hasattr(self.cfg.data.label_params, "add_end_silence"):
            if self.cfg.data.label_params.add_end_silence:
                audio_len = audio_len - int(self.cfg.data.label_params.silence_dur * (self.sr / self.cfg.data.audio_params.hop_length))
        aligned_labels, texts = data_utils.align_labels_with_frames(fixed_context_labels, audio_len, self.label_mapping)
        
        texts = self.cfg.data.text_delim.join(texts)    
        aligned_labels = torch.from_numpy(np.array(aligned_labels)).long()
        if hasattr(self.cfg.data.label_params, "add_end_silence"):
            if self.cfg.data.label_params.add_end_silence:
                user_end_label = self.label_mapping[self.cfg.data.special_tokens.user_end]
                aligned_labels = torch.cat((aligned_labels, torch.full((int(self.cfg.data.label_params.silence_dur * (self.sr / self.cfg.data.audio_params.hop_length)),), user_end_label, dtype=torch.long)), dim=0)
        
        if hasattr(self.cfg.data, "vap"):
            vap_labels = data_utils.load_data_from_file(os.path.join(self.cfg.data.root_path, self.cfg.data.paths[self.mode].full_vad_out_save_path, key + ".json"), reader="json")
            aligned_vap_labels = data_utils.align_vap_labels_with_frames(vap_labels, start_time, end_time, aligned_labels.shape[-1])
        assert y.shape[-1] == round(end_time - start_time) * self.sr, f"Shape mismatch: {y.shape[-1]} != {round(end_time - start_time) * self.cfg.data.audio_params.sr}\nStart time: {start_time}, End time: {end_time}, Key: {key}\nAudio path: {audio_path}\n audio shape: {y.shape}, sr: {self.sr}, cfg sr: {self.cfg.data.audio_params.sr}, melspec shape: {melspec.shape}, aligned labels shape: {aligned_labels.shape}"
        assert melspec.shape[-1] == aligned_labels.shape[-1], f"Shape mismatch: {melspec.shape[-1]} != {aligned_labels.shape[-1]}"
        if self.max_length is not None:
            if hasattr(self.cfg.data.label_params, "add_end_silence"):
                if self.cfg.data.label_params.add_end_silence:
                    self.max_length += int(self.cfg.data.label_params.silence_dur * (self.sr / self.cfg.data.audio_params.hop_length))
                    
            # exit()
            if melspec.shape[-1] > self.max_length:
                if preserve_channels:   
                    melspec = melspec[:, :, :self.max_length]
                else:
                    melspec = melspec[:, :self.max_length]
                aligned_labels = aligned_labels[:self.max_length]
                aligned_vap_labels = aligned_vap_labels[:, :self.max_length] if hasattr(self.cfg.data, "vap") else None
        if hasattr(self.cfg.data, "vap"):
            ##remove last 2 sec - 50 frames
            melspec = melspec[:, :, :-50] 
            aligned_vap_labels = torch.from_numpy(aligned_vap_labels).float()
            # assert melspec.shape[-1] == aligned_labels.shape[-1] == aligned_vap_labels.shape[-1], f"Shape mismatch: {melspec.shape[-1]} != {aligned_labels.shape[-1]} != {aligned_vap_labels.shape[-1]}, {start_time, end_time, key}"
            return melspec, aligned_labels, aligned_vap_labels, (audio_path, 0, texts, start_time, end_time, key)
        return melspec, aligned_labels, (audio_path, 0, texts, start_time, end_time, key)


class HumDial_dataset_infer(HumDial_dataset):
    def __init__(self, cfg, mode, feat_extractor=None):
        self.cfg = cfg
        processed_labels_save_path = os.path.join(cfg.data.root_path, cfg.data.paths[mode].processed_labels_save_path)
        data_jsons =sorted(os.listdir(processed_labels_save_path))
        audio_files = data_utils.get_files(Path(os.path.join(cfg.data.root_path, cfg.data.paths[mode].data_path)), extension=".wav")
        id2audios = {}
        for audio_file in audio_files:
            audio_id = Path(audio_file).stem
            parent_folder = str(Path(audio_file).parent).split('/')[-1].replace(' ', '__')
            id = parent_folder + "_" + audio_id
            id2audios[id] = audio_file
        self.keys = set()
        for json_file in tqdm(data_jsons, desc=f"Loading {mode} data"):
            data_json = data_utils.load_data_from_file(os.path.join(processed_labels_save_path, json_file), reader="json")
            self.keys.add(Path(json_file).stem)
        # audios = os.path.join(cfg.data.root_path, cfg.data.paths[mode].audios)
        self.audio_ids = id2audios
        if hasattr(cfg.data, "num_samples"):
            if cfg.data.num_samples is not None:
                logger.info(f"Reducing number of {mode} samples to {cfg.data.num_samples}")
                self.keys = sorted(self.keys)[:cfg.data.num_samples]
        self.keys = sorted(list(self.keys))#[797:]
        self.label_mapping = data_utils.get_token_to_id_mapping(cfg)
        self.mode = mode
        self.get_audio_feature(feat_extractor)
            
    def __getitem__(self, idx):
        key = self.keys[idx]
        processed_labels_save_path = os.path.join(self.cfg.data.root_path, self.cfg.data.paths[self.mode].processed_labels_save_path)
        label_data = data_utils.load_data_from_file(os.path.join(processed_labels_save_path, key + ".json"), reader="json")
        label_data, start_time, end_time = data_utils.convert_continous_labels_to_list(self.cfg, label_data)
        audio_path = self.audio_ids[key]
        preserve_channels, multi_stream = False, False
        if hasattr(self.cfg.data, "multi_audio_stream"):
            preserve_channels = self.cfg.data.multi_audio_stream
            multi_stream = True

        if hasattr(self.cfg.infer_params, "system_stream"):
            if not self.cfg.infer_params.system_stream:
                preserve_channels = True
        if hasattr(self.cfg.data, "zero_system"):
            if self.cfg.data.zero_system:
                preserve_channels = True
        y = data_utils.load_audio_segment(audio_path, start_time, end_time, self.sr, preserve_channels=preserve_channels)
        if hasattr(self.cfg.data, "zero_system"):
            if self.cfg.data.zero_system:
                y = y[0]
                preserve_channels = False
        if hasattr(self.cfg.infer_params, "system_stream"):
            if not self.cfg.infer_params.system_stream:
                    ##noise close to 0
                    y[1, :] = torch.randn_like(y[1, :]) * 0.0001
                    # y[1, :] = torch.zeros_like(y[1, :])
        if not multi_stream and preserve_channels:
            y = y.mean(0)  # Convert to mono if multi_stream is False and preserve_channels is True
        probs, entropy = None, None
        with torch.no_grad():
            yd = y
            if self.cfg.data.audio_params.audio_feature != "logmel" and preserve_channels:
                yd = y.numpy()



                yd = [yd[0], yd[1]]
            melspec = self.audio_feature(yd)
            # if len(melspec) == 3:
            #     melspec, probs, entropy = melspec
        aligned_labels, texts = data_utils.align_labels_with_frames(label_data, melspec.shape[-1], self.label_mapping)
        
        
        texts = self.cfg.data.text_delim.join(texts)    
        aligned_labels = torch.from_numpy(np.array(aligned_labels)).long()

        if hasattr(self.cfg.data, "vap"):
            vap_labels = data_utils.load_data_from_file(os.path.join(self.cfg.data.root_path, self.cfg.data.paths[self.mode].full_vad_out_save_path, key + ".json"), reader="json")
            aligned_vap_labels = data_utils.align_vap_labels_with_frames(vap_labels, start_time, end_time, aligned_labels.shape[-1])
            aligned_vap_labels = torch.from_numpy(aligned_vap_labels).float()


        assert y.shape[-1] == round(round(end_time - start_time, 3) * self.sr), f"Shape mismatch: {y.shape[-1]} != {round(end_time - start_time) * self.sr,  start_time, end_time, key}"
        assert melspec.shape[-1] == aligned_labels.shape[-1], f"Shape mismatch: {melspec.shape[-1]} != {aligned_labels.shape[-1],  start_time, end_time}"
        if probs is not None:
            return melspec, aligned_labels, (audio_path, label_data, texts, start_time, end_time, key), (probs, entropy)
        if hasattr(self.cfg.data, "vap"):
            ##remove last 2 sec - 50 frames
            # melspec = melspec[:, :, :-50] 
            # aligned_vap_labels = aligned_vap_labels[:, :-50]
            # assert melspec.shape[-1] == aligned_labels.shape[-1] == aligned_vap_labels.shape[-1], f"Shape mismatch: {melspec.shape[-1]} != {aligned_labels.shape[-1]} != {aligned_vap_labels.shape[-1]}, {start_time, end_time, key}"
            return melspec, aligned_labels, aligned_vap_labels, (audio_path, label_data, texts, start_time, end_time, key)
        return melspec, aligned_labels, (audio_path, label_data, texts, start_time, end_time, key)


