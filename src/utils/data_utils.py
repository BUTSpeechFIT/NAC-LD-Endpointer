import json
import random
import torch
import os
import torchaudio
import re
from pathlib import Path

def get_files(path: Path, extension='.wav'):
    path = path.expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

def load_data_from_file(file_path, reader="json"):
    if reader == 'json':
        with open(file_path, 'r') as f:
            return json.load(f)
    elif reader == 'txt':
        with open(file_path, 'r') as f:
            data = f.read().split("\n")
            return [d for d in data if d != ""]
    else:
        raise NotImplementedError(f"File type not supported: {file_path}")

def write_data_to_file(data, file_path, writer="json", indent=4):
    if writer == 'json':
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
    elif writer == 'txt':
        raise NotImplementedError("txt writer not implemented")
    else:
        raise NotImplementedError(f"File type not supported: {file_path}")

def load_audio_segment(audio_path, start_time, end_time, sr, preserve_channels=False):
    num_frames = round((end_time - start_time) * sr)
    expected = round(round((end_time - start_time)) * sr)
    if num_frames != expected:
        if expected - num_frames < 20:
            num_frames = expected

    y, _sr = torchaudio.load(
            audio_path, 
            num_frames=round((end_time - start_time) * sr), 
            frame_offset=round(start_time * sr)
        )
    num_dims = len(y.squeeze().shape)
    if num_dims > 1:
        if not preserve_channels:
            y = torch.mean(y, dim=0)
    return y.squeeze()

def load_full_audio(audio_path, sr, preserve_channels=False):
    y = torchaudio.load(audio_path)[0]
    if not preserve_channels:
        y = torch.mean(y, dim=0)
    return y

def get_token_to_id_mapping(cfg):
    return {k:i for i, k in enumerate(cfg.data.special_tokens.values())}

def get_id_to_token_mapping(cfg):
    return {i:k for i, k in enumerate(cfg.data.special_tokens.values())}
        
def convert_continous_labels_to_list(cfg, raw_labels):
    list_of_labels = []
    for label in raw_labels[0]:
        if label["turn"] == cfg.data.special_tokens.bos:
            continue
        list_of_labels.append(
            [
                label["turn"],
                label["start_time"],
                label["end_time"],
                label["text"]
            ]
        )
    start_time = list_of_labels[0][1]
    end_time = list_of_labels[-1][2]
    return list_of_labels, start_time, end_time

def preprocess_text(text, normaliser):
    def easy_normalize_text(text):
        digit_to_word = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine"
        }
        def replace_digit(match):
            return digit_to_word[match.group()]
        return re.sub(r"\b[0-9]\b", replace_digit, text)

    assert isinstance(text, list), f"Expected list, got {type(text)}"
    lines = []
    for line in text:
        line = line.lower()
        line = normaliser.normalize(line)
        line = line.replace("?", " ")
        line = line.replace(" hundred and ", " ")
        line = line.replace("-", " ")
        line = line.replace(".", " ")
        line = line.replace(",", " ")
        line = re.sub(r"\bi'm\b", "i am", line)
        line = line.replace("'", "")
        line = line.replace("?", " ")
        line = easy_normalize_text(line)
        line = " ".join(line.split())
        lines.append(line.strip()) 
    return lines
    
def convert_continous_labels_to_fixed_context_frames(cfg, raw_labels, key):
    context_length = cfg.data.label_params.context_in_sec
    start_time = 0  
    context_labels = []
    select_segment = random.choice(raw_labels)
    raw_labels_ = raw_labels
    raw_labels = select_segment
    
    if cfg.data.label_params.use_random_start:
        final_end_time = raw_labels[-1]["end_time"]
        max_start_time = final_end_time - context_length - cfg.data.label_params.extra_offset
        turn_start_times = [label["start_time"] for label in raw_labels if label["start_time"] < max_start_time and label["turn"] in ["user", "system"]]
        if turn_start_times == []:
            print(turn_start_times, raw_labels, key)
            return None
        start_time = random.choice(turn_start_times)
    else:
        start_time = raw_labels[0]["start_time"]
        # start_time = 183.82z
    start_idx = [i for i, label in enumerate(raw_labels) if label["start_time"] == start_time][0]
    sliced = raw_labels[start_idx:]
    stop = False
    for turn_idx, label in enumerate(raw_labels):
        if turn_idx < start_idx:
            continue
        if context_labels == []:
            if label["end_time"] > start_time + context_length:
                label["end_time"] = start_time + context_length
                stop = True
        else: 
            total_length = round(context_labels[-1][2] - context_labels[0][1], 4)
            current_length_from_end = round(label["end_time"] - label["start_time"], 4)
            if total_length == context_length:
                break
            if round(total_length + current_length_from_end, 4) >= context_length:
                label["end_time"] = label["start_time"] + context_length - total_length
                stop = True
                                
        context_labels.append(
            [
                label["turn"],
                label["start_time"],
                label["end_time"],
                label["text"]
            ]
        )   
        if stop:
            break
    sum_of_all_segments = sum([label[2] - label[1] for label in context_labels])
    start_time = context_labels[0][1]
    end_time = context_labels[-1][2]
    return context_labels, start_time, end_time

def align_labels_with_frames(labels, num_frames, mapping):
    total_duration = labels[-1][2] - labels[0][1]
    
    duration_per_frame = total_duration / num_frames
    aligned_labels = []
    expected_time_used = 0
    texts = []
    prev_num_frames = 0
    for label_idx, label in enumerate(labels):
        texts.append(label[3])
        tag = mapping[label[0]]
        label_duration = label[2] - label[1]
        assert label_duration > 0, f"Label duration is negative: {label}"
        num_frames_in_label = label_duration / duration_per_frame
        aligned_labels.extend([tag] * round(num_frames_in_label))
        expected_time_used += label_duration
        time_used = len(aligned_labels) * duration_per_frame
        if time_used < expected_time_used:
            additional_duration = expected_time_used - time_used
            additional_num_frames = additional_duration / duration_per_frame
            aligned_labels.extend([tag] * round(additional_num_frames))
            time_used_ = len(aligned_labels) * duration_per_frame
        if time_used > expected_time_used:
            additional_duration = time_used - expected_time_used
            additional_num_frames = additional_duration / duration_per_frame
            aligned_labels = aligned_labels[:len(aligned_labels)-round(additional_num_frames)]
            time_used_ = len(aligned_labels) * duration_per_frame
        prev_num_frames = len(aligned_labels)
    if num_frames - len(aligned_labels) == 1:
        aligned_labels.append(aligned_labels[-1])   
    if len(aligned_labels) - num_frames == 1:
        aligned_labels = aligned_labels[:-1]
    
    assert len(aligned_labels) == num_frames, f"Expected {num_frames} frames, got {len(aligned_labels)} frames, {total_duration}, {labels}"
    return aligned_labels, texts       
    

import torch
from typing import Callable

@torch.no_grad()
def get_speech_timestamps(audio: torch.Tensor,
                          model,
                          threshold: float = 0.5,
                          sampling_rate: int = 16000,
                          min_speech_duration_ms: int = 250,
                          max_speech_duration_s: float = float('inf'),
                          min_silence_duration_ms: int = 100,
                          speech_pad_ms: int = 30,
                          return_seconds: bool = False,
                          visualize_probs: bool = False,
                          progress_tracking_callback: Callable[[float], None] = None,
                          neg_threshold: float = None,
                          window_size_samples: int = 512,):
    """
    This is the modified version of silero vad's get_speech_timestamps function
    Changes made to remove trailing silence after the end of speech
    """
    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")
    ##move audio to cuda
    # audio = audio.cuda()
    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
    else:
        step = 1

    if sampling_rate not in [8000, 16000]:
        raise ValueError("Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates")

    window_size_samples = 512 if sampling_rate == 16000 else 256
    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        # chunk = chunk.cuda()
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)
        # caculate progress and seng it to callback function
        progress = current_start_sample + window_size_samples
        if progress > audio_length_samples:
            progress = audio_length_samples
        progress_percent = (progress / audio_length_samples) * 100
        if progress_tracking_callback:
            progress_tracking_callback(progress_percent)

    triggered = False
    speeches = []
    current_speech = {}

    if neg_threshold is None:
        neg_threshold = threshold - 0.15
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = next_start = 0  # to save potential segment limits in case of maximum segment size reached
    first_start_threshold = 0.02
    first_trigger = True
    for i, speech_prob in enumerate(speech_probs):
        if first_trigger:
            if speech_prob >= first_start_threshold:
                first_trigger = False
                triggered = True
                current_speech['start'] = window_size_samples * i
                continue
            
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue

        if triggered and (window_size_samples * i) - current_speech['start'] > max_speech_samples:
            if prev_end:
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:  # previously reached silence (< neg_thres) and is still not speech (< thres)
                    triggered = False
                else:
                    current_speech['start'] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech['end'] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue
        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if ((window_size_samples * i) - temp_end) > min_silence_samples_at_max_speech:  # condition to avoid cutting in very short silence
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)
    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                # speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                # speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        # else:
            # speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        for speech_dict in speeches:
            speech_dict['start'] = round(speech_dict['start'] / sampling_rate, 1)
            speech_dict['end'] = round(speech_dict['end'] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    if visualize_probs:
        make_visualization(speech_probs, window_size_samples / sampling_rate)

    return speeches


