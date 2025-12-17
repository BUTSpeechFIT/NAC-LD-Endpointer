
import os
import torch
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from src.utils import data_utils, logger
from src.utils.run_utils import resample_audio

def handle_and_add_turns(cfg, dataset):
    """
    Handle overlapping segments and add turn-begin and turn-end tokens as required
    Args:
        cfg: Configuration object
        dataset: Name of the dataset to process
    Saves:
        Processed data with turn tokens added to specified path in cfg
    """
    ### NOTE: Clean this up and generalise to all datasets - specific to spokenwoz currently
    for mode in cfg.data.modes:
        processed_save_path = cfg.data.save_paths.processed_data_path.format(dataset=dataset, mode=mode)
        processed_save_path = os.path.join(cfg.data.save_paths.dump, processed_save_path)
        if os.path.exists(processed_save_path):
            if dataset not in cfg.data.override_processed_data:
                logger.logger.info(f"Processed output already exists at {processed_save_path}. Skipping...")
                return

        skip_existing, skip_current = 0, 0
        modify_existing, modify_current = 0, 0
        vad_out_save_path = cfg.data.save_paths.vad_data_path.format(dataset=dataset, mode=mode)
        vad_out_save_path = os.path.join(cfg.data.save_paths.dump, vad_out_save_path)
        vad_data = data_utils.load_data_from_file(vad_out_save_path)
        save_data = {}
        
        for key in vad_data:
            json_data = vad_data[key]
            processed_labels = []
            split_segment = False
            file_segments = []
            raw_data = []
            for label_data in json_data["segments"]:
                turn = label_data["turn"]
                text = label_data["text"]
                seg_start_time = round(label_data["start_time"], 4)
                seg_end_time = round(label_data["end_time"], 4)
                raw_data.append((turn, seg_start_time, seg_end_time, text))
                assert seg_end_time > seg_start_time, f"Invalid segment with end time <= start time: {seg_end_time} <= {seg_start_time}"
                if processed_labels == []:
                    ##if the segment starts after the selected start time, 
                    ## we will add a turn-begin token at the start
                    if seg_start_time > 0:
                        if len(file_segments) == 0: 
                                prefix_end = cfg.data.special_tokens.system_end
                                if turn == cfg.data.special_tokens.system:
                                    prefix_end = cfg.data.special_tokens.user_end
                                processed_labels.append(
                                    {
                                        "turn": prefix_end,
                                        "start_time": 0,
                                        "end_time": seg_start_time,
                                        "text": ""
                                    }
                                )
                    processed_labels.append(
                        {
                            "turn": turn,
                            "start_time": seg_start_time,
                            "end_time": seg_end_time,
                            "text": text
                        }
                    )
                ##for all segments from second segment onwards
                else:
                    ##so we will keep this simple and focus on the user timestamps in first pass
                    if turn == cfg.data.special_tokens.user:
                        previous_turn_end_time = processed_labels[-1]["end_time"]
                        if seg_start_time < previous_turn_end_time:
                            previous_turn_end_time = seg_start_time
                            if previous_turn_end_time <= processed_labels[-1]["start_time"]: ##we will need to remove the previous segment entirely
                                skip_existing += 1
                                processed_labels.pop()
                                continue
                            processed_labels[-1]["end_time"] = previous_turn_end_time
                            modify_existing += 1
                            
                        processed_labels.append(
                            {
                                "turn": cfg.data.special_tokens.user,
                                "start_time": seg_start_time,
                                "end_time": seg_end_time,
                                "text": text
                            }
                        )

                    elif turn == cfg.data.special_tokens.system:
                        previous_turn_end_time = processed_labels[-1]["end_time"]
                        if seg_start_time < previous_turn_end_time:
                            seg_start_time = previous_turn_end_time ##consider this as system barge-in and ignore for endpointing
                            if seg_start_time >= seg_end_time:
                                skip_current += 1
                                continue
                            modify_current += 1
                        processed_labels.append(
                            {
                                "turn": cfg.data.special_tokens.system,
                                "start_time": seg_start_time,
                                "end_time": seg_end_time,
                                "text": text
                            }
                        )
            #now we fill in the turn-end tokens
            processed_labels_with_ends = []
            for idx in range(len(processed_labels)):
                if idx == 0:
                    processed_labels_with_ends.append(processed_labels[idx])
                    continue
                prev_label = processed_labels[idx - 1]
                curr_label = processed_labels[idx]
                if prev_label["turn"] == cfg.data.special_tokens.system:
                    end_token = cfg.data.special_tokens.system_end
                else:
                    end_token = cfg.data.special_tokens.user_end
                prev_end = prev_label["end_time"]
                current_start = curr_label["start_time"]
                if current_start > prev_end:
                    processed_labels_with_ends.append(
                        {
                            "turn": end_token,
                            "start_time": prev_end,
                            "end_time": current_start,
                            "text": ""
                        }
                    )
                processed_labels_with_ends.append(processed_labels[idx])
            
            ##now we verify all time stamps are in order and filled
            segment_times = 0
            for idx in range(len(processed_labels_with_ends)):
                if idx == 0:
                    segment_times += processed_labels_with_ends[idx]["end_time"] - processed_labels_with_ends[idx]["start_time"]
                    continue
                prev_label = processed_labels_with_ends[idx - 1]
                curr_label = processed_labels_with_ends[idx]
                assert round(curr_label["start_time"],4) >= round(prev_label["end_time"],4), f"Timestamps not in order for {key}: {curr_label} after {prev_label}"
                assert round(curr_label["start_time"],4) == round(prev_label["end_time"],4), f"Timestamps gap found for {key}: {curr_label} after {prev_label}"
                segment_times += curr_label["end_time"] - curr_label["start_time"]
                assert round(segment_times, 4) == round(processed_labels_with_ends[idx]["end_time"],4), f"Duration mismatch at idx {idx} for {key}: {segment_times} != {processed_labels_with_ends[idx]['end_time'] - processed_labels_with_ends[0]['start_time']}"
            assert round(segment_times,4) == round(processed_labels_with_ends[-1]["end_time"], 4), f"Total duration mismatch for {key}: {segment_times} != {processed_labels_with_ends[-1]['end_time'] - processed_labels_with_ends[0]['start_time']}"
                        
            save_data[key] = {
                "audio_filepath": json_data["audio_filepath"],
                "segments": processed_labels_with_ends
            }

        data_utils.write_data_to_file(save_data, processed_save_path, writer="json")

## Define global variables for worker processes
_worker_vad_model = None
_worker_sr = None

def init_vad_worker(sr):
    """Initialize each worker with its own VAD model"""
    global _worker_vad_model, _worker_sr
    import torch
    # Adjust the import based on your actual VAD implementation
    from silero_vad import load_silero_vad
    _worker_vad_model = load_silero_vad()
    _worker_sr = sr

def _process_single_item_vad(args):
    """Worker function for VAD processing"""
    key, item = args
    y, _ = librosa.load(item["audio_filepath"], sr=_worker_sr, mono=True)
    save_item = {"audio_filepath": item["audio_filepath"], "segments": []}
    
    for segment in item["segments"]:
        y_seg = y[round(segment["start_time"] * _worker_sr):round(segment["end_time"] * _worker_sr)]
        from silero_vad import get_speech_timestamps 
        vad_out = get_speech_timestamps(
            y_seg,
            _worker_vad_model, 
            return_seconds=False,
            sampling_rate=_worker_sr,
            min_silence_duration_ms=0,
            min_speech_duration_ms=0,
            neg_threshold=0.9
        )
        
        if vad_out == []:
            vad_begin = 0
            vad_end = segment["end_time"] - segment["start_time"]
        else:
            vad_begin = vad_out[0]["start"] / _worker_sr
            vad_end = vad_out[-1]["end"] / _worker_sr
            
        save_item["segments"].append({
            "turn": segment["turn"],
            "text": segment["text"],
            "old_start_time": segment["start_time"],
            "old_end_time": segment["end_time"],
            "vad_start_time": vad_begin,
            "vad_end_time": vad_end,
            "start_time": segment["start_time"] + vad_begin,
            "end_time": segment["start_time"] + vad_end
        })
    return key, save_item


def process_vad(cfg, dataset):
    """
    Process VAD for all items in the dataset using multiprocessing
    We need VAD to trim the beginning and end silences for each segment for accurate endpointing
    We use a modified version of Silero VAD, where trailing silences are also removed
    Args:
        cfg: Configuration object
        dataset: Name of the dataset to process
    Saves:
        VAD processed data to specified path in cfg
    """
    for mode in cfg.data.modes:
        save_path = cfg.data.save_paths.vad_data_path.format(dataset=dataset, mode=mode)
        save_path = os.path.join(cfg.data.save_paths.dump, save_path)
        if os.path.exists(save_path):
            if dataset not in cfg.data.override_vad_data:
                # logger.logger.info(f"VAD processed output already exists at {save_path}. Skipping...")
                continue
        preprocessed_data_path = cfg.data.save_paths.preprocessed_data_path.format(dataset=dataset, mode=mode)
        preprocessed_data_path = os.path.join(cfg.data.save_paths.dump, preprocessed_data_path)
        data_json = data_utils.load_data_from_file(preprocessed_data_path)
        args_list = [(key, item) for key, item in data_json.items()]
        with multiprocessing.Pool(
            processes=cfg.data.num_vad_workers,
            initializer=init_vad_worker,
            initargs=(cfg.data.audio_params.sr,)
        ) as pool:
            results = list(tqdm(
                pool.imap(_process_single_item_vad, args_list), 
                total=len(data_json), 
                desc=f"Processing VAD for {dataset} - {mode}"
            ))
        save_data = dict(results)
        data_utils.write_data_to_file(save_data, save_path, writer="json")

def handle_length_filtering(cfg, data_json, mode, dataset):
    """
    Handle length filtering of segments based on duration criteria
    Args:
        cfg: Configuration object
        data_json: Input data in JSON format
        mode: Mode of the dataset (e.g., train, val, test)
        dataset: Name of the dataset
    Returns:
        Filtered data JSON
    """
    save_path = cfg.data.save_paths.filtered_data_path.format(
        dataset=dataset, 
        mode=mode, 
        context_in_sec=cfg.data.label_params.context_in_sec, 
        extra_offset=cfg.data.label_params.extra_offset
        )
    save_path = os.path.join(cfg.data.save_paths.dump, save_path)
    if os.path.exists(save_path):
        if dataset not in cfg.data.override_filtered_data:
            logger.logger.info(f"Length filtered output already exists at {save_path}. Skipping...")
            return data_utils.load_data_from_file(save_path, reader="json")
    filtered_data = {}
    total_skipped_due_to_short_duration = 0
    for key in data_json:
        seg_duration = data_json[key]["segments"][-1]["end_time"] - data_json[key]["segments"][0]["start_time"]
        if seg_duration < cfg.data.label_params.context_in_sec + cfg.data.label_params.extra_offset:
            total_skipped_due_to_short_duration += 1
            continue
        filtered_data[key] = {}
        filtered_data[key]["audio_filepath"] = data_json[key]["audio_filepath"]
        filtered_data[key]["segments"] = data_json[key]["segments"]
    data_utils.write_data_to_file(filtered_data, save_path, writer="json")
    return filtered_data

class endpointing_dataset(torch.utils.data.Dataset):
    """
    Dataset class for endpointing tasks
    Args:
        cfg: Configuration object
        mode: Mode of the dataset (e.g., train, val, test)
        dataset: Name of the dataset
        feat_extractor: Feature extractor for audio features
    Returns:
        Dataset object for endpointing tasks
    """
    def __init__(self, cfg, mode, dataset, feat_extractor=None):
        self.cfg = cfg
        processed_labels_save_path = cfg.data.save_paths.processed_data_path.format(dataset=dataset, mode=mode)
        processed_labels_save_path = os.path.join(cfg.data.save_paths.dump, processed_labels_save_path)
        json_data = data_utils.load_data_from_file(processed_labels_save_path, reader="json")

        data_json = handle_length_filtering(cfg, json_data, mode, dataset)
        self.keys = data_json.keys()
        logger.logger.info(f"Total filtered / available samples: {len(self.keys)} / {len(json_data)}")
        if hasattr(cfg.data.datasets[dataset], "num_samples"):
            if mode in cfg.data.datasets[dataset]["num_samples"]:
                if cfg.data.datasets[dataset]["num_samples"][mode] is not None:
                    logger.logger.info(f"Reducing number of {mode} samples to {cfg.data.datasets[dataset]['num_samples'][mode]}")
                    self.keys = sorted(self.keys)[:cfg.data.datasets[dataset]['num_samples'][mode]]
        self.keys = list(sorted(self.keys))
        self.label_mapping = data_utils.get_token_to_id_mapping(cfg)
        self.mode = mode
        self.dataset = dataset
        self.data_json = data_json
        self.get_audio_feature(feat_extractor)

                
        assert cfg.data.label_params.use_fixed_context_training == True, "Only fixed context training is supported"
        self.max_length = None
        if hasattr(cfg.data, "max_length"):
            logger.logger.info(f"Setting max length to {cfg.data.max_length}")
            self.max_length = cfg.data.max_length

    def get_audio_feature(self, feat_extractor):
        """
        Initialize the audio feature extractor based on configuration, and handle resampling if necessary.
        Args:
            feat_extractor: Predefined feature extractor (if any)
        Saves:
            Updates the audio file paths in data_json if resampling is performed.
        """
        cfg = self.cfg
        if cfg.data.audio_params.audio_feature == "logmel":
            self.audio_feature = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.data.datasets[self.dataset].sr,
                n_fft=cfg.data.audio_params.n_fft,
                win_length=cfg.data.audio_params.win_length,
                hop_length=cfg.data.audio_params.hop_length,
                n_mels=cfg.data.audio_params.n_mels,
                power=cfg.data.audio_params.power,
            )
            self.sr = cfg.data.datasets[self.dataset].sr
        else:
            self.audio_feature = feat_extractor
            self.sr = cfg.data.datasets[self.dataset].sr
            if self.cfg.data.datasets[self.dataset].sr != self.cfg.data.audio_params.target_sr:
                self.sr = self.cfg.data.audio_params.target_sr
                num_audios = len(self.keys)
                resampled_audios_path = cfg.data.save_paths.resampled_audios_path.format(dataset=self.dataset, mode=self.mode, target_sr=self.cfg.data.audio_params.target_sr)
                resampled_audios_path = os.path.join(cfg.data.save_paths.dump, resampled_audios_path)
                os.makedirs(resampled_audios_path, exist_ok=True)
                num_resampled_audios = len(os.listdir(resampled_audios_path))
                if num_audios == num_resampled_audios:
                    # logger.logger.info(f"All audios already resampled to {self.cfg.data.audio_params.target_sr}Hz, loading from {resampled_audios_path}")
                    for key in self.keys:
                        self.data_json[key]["audio_filepath"] = os.path.join(resampled_audios_path, key + ".wav")
                    return
                logger.logger.info(f"Resampling audios to {self.cfg.data.audio_params.target_sr}Hz, saving at {resampled_audios_path}, Pending audios: {num_resampled_audios - num_audios}")
                preserve_channels = False
                if hasattr(self.cfg.data, "multi_audio_stream"):
                    preserve_channels = self.cfg.data.multi_audio_stream
                
                for key in tqdm(self.keys, desc=f"Resampling audios for {self.mode} - {self.dataset}"):
                    save_path = os.path.join(resampled_audios_path, key + ".wav")
                    if os.path.exists(save_path):
                        self.data_json[key]["audio_filepath"] = save_path
                        continue
                    y = data_utils.load_full_audio(self.data_json[key]["audio_filepath"], self.cfg.data.datasets[self.dataset].sr, preserve_channels=preserve_channels)
                    y_reKhz = resample_audio(y, self.cfg.data.datasets[self.dataset].sr, self.cfg.data.audio_params.target_sr)
                    if len(y_reKhz.shape) == 1:
                        y_reKhz = y_reKhz.unsqueeze(0)
                    torchaudio.save(save_path, y_reKhz, self.cfg.data.audio_params.target_sr)
                    self.data_json[key]["audio_filepath"] = save_path
                for key in self.keys:
                    self.data_json[key]["audio_filepath"] = os.path.join(resampled_audios_path, key + ".wav")
                
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        """
        Load frame-level turn labels for a fixed length segment, along with corresponding audio features.
        """
        key = self.keys[idx]
        label_data = self.data_json[key]["segments"]
        if self.cfg.data.label_params.use_fixed_context_training:
            fixed_context_labels, start_time, end_time = data_utils.convert_continous_labels_to_fixed_context_frames(self.cfg, label_data, key)    
        preserve_channels = False
        if hasattr(self.cfg.data, "multi_audio_stream"):
            preserve_channels = self.cfg.data.multi_audio_stream
        if hasattr(self.cfg.data, "zero_system"):
            if self.cfg.data.zero_system:
                preserve_channels = True
        print("Loading audio segment:", self.data_json[key]["audio_filepath"], start_time, end_time, "Preserve channels:", preserve_channels, self.sr)
        y = data_utils.load_audio_segment(self.data_json[key]["audio_filepath"], start_time, end_time, self.sr, preserve_channels=preserve_channels)
        if hasattr(self.cfg.data, "zero_system"):
            if self.cfg.data.zero_system:
                y = y[0]
                preserve_channels = False
        with torch.no_grad():
            yd = y
            if self.cfg.data.audio_params.audio_feature not in ["logmel", "logmel-v2"] and preserve_channels:
                yd = y.numpy()
                yd = [yd[0], yd[1]]
            melspec = self.audio_feature(yd)
        aligned_labels, texts = data_utils.align_labels_with_frames(fixed_context_labels, melspec.shape[-1], self.label_mapping)
        texts = self.cfg.data.text_delim.join(texts)    
        aligned_labels = torch.from_numpy(np.array(aligned_labels)).long()        
        assert y.shape[-1] == round(end_time - start_time) * self.sr, f"Shape mismatch: {y.shape[-1]} != {round(end_time - start_time) * self.cfg.data.datasets[self.dataset].sr}"
        assert melspec.shape[-1] == aligned_labels.shape[-1], f"Shape mismatch: {melspec.shape[-1]} != {aligned_labels.shape[-1]}"
        if self.max_length is not None:
            if melspec.shape[-1] > self.max_length:
                if preserve_channels:
                    melspec = melspec[:, :, :self.max_length]
                else:
                    melspec = melspec[:, :self.max_length]
                aligned_labels = aligned_labels[:self.max_length]
        return melspec, aligned_labels, (self.data_json[key]["audio_filepath"], 0, texts, start_time, end_time, key)


class endpointing_dataset_full_context(endpointing_dataset):
    def __init__(self, cfg, mode, dataset, feat_extractor=None):
        IGNORE_KEYS = [
            "SNG0601", 
            "SNG0646", 
            "SNG0653",
            "SNG0877", 
            "SNG0885", 
            "SNG0890", 
            "SNG0897", 
            "SNG0901",
            "SNG0903",
            "MUL0363",
        ]
        self.cfg = cfg
        processed_labels_save_path = cfg.data.save_paths.processed_data_path.format(dataset=dataset, mode=mode)
        processed_labels_save_path = os.path.join(cfg.data.save_paths.dump, processed_labels_save_path)
        data_json = data_utils.load_data_from_file(processed_labels_save_path, reader="json")
        self.keys = data_json.keys()
        if hasattr(cfg.data.datasets[dataset], "num_samples"):
            if cfg.data.datasets[dataset].num_samples[mode] is not None:
                logger.logger.info(f"Reducing number of {mode} samples to {cfg.data.datasets[dataset].num_samples[mode]}")
                self.keys = sorted(self.keys)[:cfg.data.datasets[dataset].num_samples[mode]]
        self.keys = sorted(list(self.keys))
        for k in IGNORE_KEYS:
            if k in self.keys:
                self.keys.remove(k)
        self.label_mapping = data_utils.get_token_to_id_mapping(cfg)
        self.mode = mode
        self.get_audio_feature(feat_extractor)
            
    def __getitem__(self, idx):
        key = self.keys[idx]
        label_data = self.data_json[key]["segments"]
        label_data, start_time, end_time = data_utils.convert_continous_labels_to_list(self.cfg, label_data)
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
        y = data_utils.load_audio_segment(self.data_json[key]["audio_filepath"], start_time, end_time, self.sr, preserve_channels=preserve_channels)
        if hasattr(self.cfg.data, "zero_system"):
            if self.cfg.data.zero_system:
                y = y[0]
                preserve_channels = False
        if hasattr(self.cfg.infer_params, "system_stream"):
            if not self.cfg.infer_params.system_stream:
                    y[1, :] = torch.randn_like(y[1, :]) * 0.0001
        if not multi_stream and preserve_channels:
            y = y.mean(0)  # Convert to mono if multi_stream is False and preserve_channels is True
        probs, entropy = None, None
        with torch.no_grad():
            yd = y
            if self.cfg.data.audio_params.audio_feature != "logmel" and preserve_channels:
                yd = y.numpy()
                yd = [yd[0], yd[1]]
            melspec = self.audio_feature(yd)

        aligned_labels, texts = data_utils.align_labels_with_frames(label_data, melspec.shape[-1], self.label_mapping)
        
        texts = self.cfg.data.text_delim.join(texts)    
        aligned_labels = torch.from_numpy(np.array(aligned_labels)).long()

        assert y.shape[-1] == round(round(end_time - start_time, 3) * self.sr), f"Shape mismatch: {y.shape[-1]} != {round(end_time - start_time) * self.sr,  start_time, end_time, key}"
        assert melspec.shape[-1] == aligned_labels.shape[-1], f"Shape mismatch: {melspec.shape[-1]} != {aligned_labels.shape[-1],  start_time, end_time}"
        if probs is not None:
            return melspec, aligned_labels, (self.data_json[key]["audio_filepath"], label_data, texts, start_time, end_time, key), (probs, entropy)
        return melspec, aligned_labels, (self.data_json[key]["audio_filepath"], label_data, texts, start_time, end_time, key)


