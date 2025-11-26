
from src.utils import data_utils, logger, textgrid
from silero_vad import load_silero_vad
import librosa
import torch
from tqdm import tqdm
import os
import torchaudio
from pathlib import Path
import multiprocessing
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
    for mode in cfg.data.datasets[dataset].modes:
        processed_save_path = cfg.data.save_paths.processed_data_path.format(dataset=dataset, mode=mode)
        if os.path.exists(processed_save_path):
            if not cfg.data.datasets[dataset].override_processed_data:
                logger.logger.info(f"Processed output already exists at {processed_save_path}. Skipping...")
                return

        skipped_due_to_overlap = 0
        vad_out_save_path = cfg.data.save_paths.vad_data_path.format(dataset=dataset, mode=mode)
        vad_data = data_utils.load_data_from_file(vad_out_save_path)
        save_data = {}
        for key in vad_data:
            json_data = vad_data[key]
            processed_labels = []
            split_segment = False
            file_segments = []
            for label_data in json_data["segments"]:
                turn = label_data["turn"]
                text = label_data["text"]
                seg_start_time = round(label_data["start_time"], 4)
                seg_end_time = round(label_data["end_time"], 4)
                if processed_labels == []:
                    ##if the segment starts after the selected start time, 
                    ## we will add bos token
                    if seg_start_time > 0:
                        if len(file_segments) == 0: 
                            if cfg.data.label_params.use_bos_token:
                                processed_labels.append(
                                    {
                                        "turn": cfg.data.special_tokens.bos,
                                        "start_time": 0,
                                        "end_time": seg_start_time,
                                        "text": ""
                                    }
                                )
                ##for all segments from second segment onwards
                else:
                    ##if start of current segment is lesser than end of previous 
                    ##segment,this is a problem,we can swap the end time of previous and start time of current
                    if seg_end_time < processed_labels[-1]["end_time"] and seg_start_time < processed_labels[-1]["end_time"]:
                        split_segment = True
                    if seg_start_time < processed_labels[-1]["end_time"] and seg_start_time < processed_labels[-1]["start_time"]:
                        split_segment = True
                    if round(seg_start_time, 3) == round(processed_labels[-1]["start_time"], 3):
                        split_segment = True
                    if round(seg_start_time, 3) < round(processed_labels[-1]["end_time"], 3) and round(seg_end_time, 3) == round(processed_labels[-1]["end_time"], 3):
                        split_segment = True
                    if seg_start_time < processed_labels[-1]["end_time"]:
                        if processed_labels[-1]["end_time"] - seg_start_time > 0.1:
                            split_segment = True
                    if split_segment:
                        if processed_labels > []: 
                            file_segments.append(processed_labels)
                        processed_labels = []
                        split_segment = False
                        skipped_due_to_overlap += 1
                        continue
                    
                    if seg_start_time < processed_labels[-1]["end_time"]:
                        if cfg.data.label_params.swap_timestamps_during_overlap:
                            _seg_start_time = processed_labels[-1]["end_time"]
                            processed_labels[-1]["end_time"] = seg_start_time
                            seg_start_time = _seg_start_time
                            processed_labels.append(
                                {
                                    "turn": cfg.data.special_tokens[processed_labels[-1]["turn"]+'_end'],
                                    "start_time": processed_labels[-1]["end_time"],
                                    "end_time": seg_start_time,
                                    "text": ""
                                }
                            )

                        else:
                            ##instead we modify the end time of previous segment,
                            ##to the start time of current segment, so that there is no overlap
                            processed_labels[-1]["end_time"] = seg_start_time
                    
                    ##if start of current segment is greater than end of previous segment,
                    ##we add a turn-end token between the two segments
                    else:
                        if cfg.data.label_params.use_tag_end_token:
                            if processed_labels[-1]["end_time"] != seg_start_time:
                                processed_labels.append(
                                    {
                                        "turn": cfg.data.special_tokens[processed_labels[-1]["turn"]+'_end'],
                                        "start_time": processed_labels[-1]["end_time"],
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
                
                
            if processed_labels != []:
                file_segments.append(processed_labels)
            save_data[key] = {
                "audio_filepath": json_data["audio_filepath"],
                "segments": file_segments
            }
            
            ## Verify the correctness of the processed labels
            for processed_labels in file_segments:
                if len(processed_labels) == 0: continue
                total_dur = processed_labels[-1]["end_time"] - processed_labels[0]["start_time"]
                sum_of_all_durs = sum([line["end_time"] - line["start_time"] for line in processed_labels])
                assert round(total_dur, 3) == round(sum_of_all_durs, 3), f"Total duration mismatch: {total_dur} != {sum_of_all_durs}"
                for idx, x in enumerate(processed_labels):
                    assert round(x["end_time"], 3) - round(x["start_time"], 3) != 0, f"Zero duration segment found: {x}"
                    assert x["end_time"] - x["start_time"] > 0, f"Negative duration segment found: {x}"
                    if idx > 0:
                        assert x["start_time"] >= processed_labels[idx-1]["end_time"], f"Overlapping segments found: {x}, {processed_labels[idx-1]}"
                        assert x["start_time"] - processed_labels[idx-1]["end_time"] == 0, f"Boundary gap found: {x}, {processed_labels[idx-1]}"
        
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
    for mode in cfg.data.datasets[dataset].modes:
        save_path = cfg.data.save_paths.vad_data_path.format(dataset=dataset, mode=mode)
        if os.path.exists(save_path):
            if not cfg.data.datasets[dataset].override_vad_data:
                logger.logger.info(f"VAD processed output already exists at {save_path}. Skipping...")
                continue
        data_json = data_utils.load_data_from_file(cfg.data.save_paths.preprocessed_data_path.format(dataset=dataset, mode=mode))
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
                logger.logger.info(f"Preprocessed data already exists for spokenwoz, skipping...")
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
    if os.path.exists(save_path):
        if not cfg.data.datasets[dataset].override_filtered_data:
            logger.logger.info(f"Length filtered output already exists at {save_path}. Skipping...")
            return data_utils.load_data_from_file(save_path, reader="json")
    filtered_data = {}
    total_skipped_due_to_short_duration = 0
    for key in data_json:
        segments = []
        for segment in data_json[key]["segments"]:
            seg_duration = segment[-1]["end_time"] - segment[0]["start_time"]
            if seg_duration < cfg.data.label_params.context_in_sec + cfg.data.label_params.extra_offset:
                total_skipped_due_to_short_duration += 1
                continue
            segments.append(segment)
        if len(segments) > 0:
            filtered_data[key] = {}
            filtered_data[key]["audio_filepath"] = data_json[key]["audio_filepath"]
            filtered_data[key]["segments"] = segments
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
        json_data = data_utils.load_data_from_file(processed_labels_save_path, reader="json")
        total_skipped_due_to_short_duration, available = 0, 0

        data_json = handle_length_filtering(cfg, json_data, mode, dataset)
        self.keys = data_json.keys()
        logger.logger.info(f"Total skipped / available samples: {total_skipped_due_to_short_duration} / {available}")
        if hasattr(cfg.data.datasets[dataset], "num_samples"):
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
                sample_rate=cfg.data.audio_params.sr,
                n_fft=cfg.data.audio_params.n_fft,
                win_length=cfg.data.audio_params.win_length,
                hop_length=cfg.data.audio_params.hop_length,
                n_mels=cfg.data.audio_params.n_mels,
                power=cfg.data.audio_params.power,
            )
            self.sr = cfg.data.audio_params.sr
        else:
            self.audio_feature = feat_extractor
            self.sr = cfg.data.audio_params.sr
            if self.cfg.data.audio_params.sr != self.cfg.data.audio_params.target_sr:
                self.sr = self.cfg.data.audio_params.target_sr
                num_audios = len(self.keys)
                resampled_audios_path = cfg.data.save_paths.resampled_audios_path.format(dataset=self.dataset, mode=self.mode, target_sr=self.cfg.data.audio_params.target_sr)
                os.makedirs(resampled_audios_path, exist_ok=True)
                num_resampled_audios = len(os.listdir(resampled_audios_path))
                if num_audios == num_resampled_audios:
                    for key in self.keys:
                        self.data_json[key]["audio_filepath"] = os.path.join(resampled_audios_path, key + ".wav")
                    return
                logger.logger.info(f"Resampling audios to {self.cfg.data.audio_params.target_sr}Hz, saving at {resampled_audios_path}")
                preserve_channels = False
                if hasattr(self.cfg.data, "multi_audio_stream"):
                    preserve_channels = self.cfg.data.multi_audio_stream
                for key in tqdm(self.keys, desc="Resampling audios"):
                    y = data_utils.load_full_audio(self.data_json[key]["audio_filepath"], self.cfg.data.audio_params.sr, preserve_channels=preserve_channels)
                    y_reKhz = resample_audio(y, self.cfg.data.audio_params.sr, self.cfg.data.audio_params.target_sr)
                    if len(y_reKhz.shape) == 1:
                        y_reKhz = y_reKhz.unsqueeze(0)
                    save_path = os.path.join(resampled_audios_path, key + ".wav")
                    torchaudio.save(save_path, y_reKhz, self.cfg.data.audio_params.target_sr)
                    self.data_json[key]["audio_filepath"] = save_path
                
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
        
        if hasattr(self.cfg.data, "vap"):
            vap_labels = data_utils.load_data_from_file(os.path.join(self.cfg.data.root_path, self.cfg.data.paths[self.mode].full_vad_out_save_path, key + ".json"), reader="json")
            aligned_vap_labels = data_utils.align_vap_labels_with_frames(vap_labels, start_time, end_time, aligned_labels.shape[-1])
        assert y.shape[-1] == round(end_time - start_time) * self.sr, f"Shape mismatch: {y.shape[-1]} != {round(end_time - start_time) * self.cfg.data.audio_params.sr}"
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


