import os
import torch
from src.utils.logger import logger
import shutil
import torchaudio
import matplotlib.pyplot as plt
from src.utils import data_utils
from src.utils.data_utils import get_id_to_token_mapping, load_audio_segment
import numpy as np
from torch.nn import functional as F
import matplotlib.gridspec as gridspec
import librosa


def save_checkpoint(cfg, epoch, model, save_folder):
    """
    Save the model checkpoint.
    """
    if cfg.run_params.epoch_save_all:
        savename = os.path.join(save_folder, f"epoch_{epoch}.pt")
        logger.info(f"Saving model at epoch {epoch} to {savename}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, savename)
    if cfg.run_params.save_best_from_val_loss:
        savename = os.path.join(save_folder, "best_val_loss.pt")
    if cfg.run_params.save_best_from_val_acc:
        savename = os.path.join(save_folder, "best_val_acc.pt")
    logger.info(f"Saving model at epoch {epoch} to {savename}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        }, savename)

def load_checkpoint(path, model):
    """
    Load the model checkpoint.
    """
    logger.info(f"Loading model from {path}")
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
    

def setup_save_folder(cfg, config_paths):
    """
    Setup the save folder for the current run.
    """
    path = os.path.join(cfg.run_params.save_folder, cfg.run_name)
    if os.path.exists(path):
        if not cfg.run_params.overwrite_prev_run:
            logger.info(f'WARNING: {cfg.run_name} already exists. Exiting...')
            exit()
        else:
            logger.info(f'WARNING: {cfg.run_name} already exists. Overwriting. Set run_params.overwrite_prev_run to False to avoid this.')
    os.makedirs(path, exist_ok=True)
    ##get current working directory
    for config_path in config_paths:
        p = os.path.join(os.getcwd(), config_path)
        shutil.copy2(p, path)
    return path

def plot_metric(row_values, col_values, row_name, col_name, save_folder, scatter_data=None):
    plt.figure(figsize=(10, 4))
    plt.plot(row_values, col_values)
    plt.title(f"{row_name} vs {col_name}")
    plt.ylabel(col_name)
    plt.xlabel(row_name)
    if scatter_data is not None:
        plt.hlines(scatter_data, xmin=min(row_values), xmax=max(row_values), color='r')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"{row_name}_vs_{col_name}.png"))
    plt.clf()

def plot_and_save_metrics(metrics, cfg, read_other_metrics_from_file=False):
    if metrics is None:
        metrics = torch.load(os.path.join(cfg.infer_folder, cfg.run_params.asr_scores_save_filename))
        
    for metric in metrics:
        if "label" in metric:
            continue
        scatter_data = None
        if "wer" in metric: 
            scatter_data = metrics["true_label_wer"]
        elif "cer" in metric:
            scatter_data = metrics["true_label_cer"]
        plot_metric(
            list(metrics[metric].keys()),
            list(metrics[metric].values()),
            "threshold",
            metric,     
            cfg.infer_folder,
            scatter_data=scatter_data
        )
        logger.info(f"Saving {metric} to {os.path.join(cfg.infer_folder, f'{metric}.pt')}")
        torch.save(metrics[metric], os.path.join(cfg.infer_folder, f"{metric}.pt"))
    
    for i, key in enumerate(metrics):
        for j, key_ in enumerate(metrics):
            if i == j: continue
            if i > j: continue
            plot_metric(
                list(metrics[key].values()),
                list(metrics[key_].values()),
                key,
                key_,
                cfg.infer_folder
            )
    
    if "wer_metrics" in metrics and "cer_metrics" in metrics:
        if read_other_metrics_from_file:
            metrics["ep_cutoff"] = torch.load(os.path.join(cfg.infer_folder, cfg.run_params.ep_cutoff_save_filename))
            metrics["median_latency"] = torch.load(os.path.join(cfg.infer_folder, cfg.run_params.latency_save_filename))
        ep_thresholds = list(metrics["ep_cutoff"].keys())
        med_thresholds = list(metrics["median_latency"].keys())
        wer_thresholds = list(metrics["wer_metrics"].keys())
        cer_thresholds = list(metrics["cer_metrics"].keys())
        common_keys = set(ep_thresholds).intersection(set(med_thresholds)).intersection(set(wer_thresholds)).intersection(set(cer_thresholds))
        ep_cutoff_values = [metrics["ep_cutoff"][key] for key in common_keys]
        latency_values = [metrics["median_latency"][key] for key in common_keys]
        wer_values = [metrics["wer_metrics"][key] for key in common_keys]
        cer_values = [metrics["cer_metrics"][key] for key in common_keys]
        values = [ep_cutoff_values, latency_values, wer_values, cer_values]
        names = ["ep_cutoff", "latency", "wer", "cer"]
        for i, value in enumerate(values):
            for j, value_ in enumerate(values):
                if i == j: continue
                if names[i] in ["wer", "cer"] and names[j] in ["wer", "cer"]: continue
                if i > j: continue
                plot_metric(
                    value,
                    value_,
                    names[i],
                    names[j],
                    cfg.infer_folder
                )

 

def resample_audio(audio, sr, target_sr):
    return torchaudio.functional.resample(audio, sr, target_sr)

def windowed_max(signal, window_size=100):
    # Apply a sliding window max by reshaping and using max pooling
    signal_reshaped = signal.view(1, 1, -1)  # Add batch and channel dimensions
    windowed_max = F.max_pool1d(signal_reshaped, kernel_size=window_size, stride=1, padding=window_size//2)
    return windowed_max.squeeze()

def min_max_without_outliers_window(signal, window_size=100, quantile=0.99):
    # Get windowed maxima
    window_max = windowed_max(signal, window_size)
    
    # Calculate the mean of the window maxima
    threshold = torch.quantile(window_max, quantile)
    
    # Filter out values that are too large compared to the mean of max values
    mask = torch.abs(signal) <= threshold
    filtered_signal = signal[mask]
    
    # Return the minimum and maximum of the filtered signal
    return filtered_signal.min(), filtered_signal.max()

def prediction_visualization(
    cfg, 
    data, 
    label, 
    output, 
    metadata, 
    save_folder, 
    wandb=None, 
    idx=0, 
    save_name="pred_testing.png", 
    save_data_name="plot_data.pt",
    num_seconds=None,
    new_start_time=None,
    tick_period=1000,
    fig_size=(20, 10),
    latency_list=None,
    latency_index_list=None,
    plot_pitch=False,
    probs_and_entropy=None,
    latency_list_full=None,
    p_now=None,
):  
    """
    Visualize predictions and save the plot along with relevant data.
    """

    ## NOTE: Maybe clean this up in the future
    audio, _, texts, start_time, end_time, key = metadata
    if num_seconds is not None:
        actual_duration = end_time - start_time
        resolution = output.shape[1] / actual_duration
        if new_start_time is None:
            start_time = start_time.item()
        else:
            start_time = new_start_time
        end_time = start_time + num_seconds
        output = output[idx][int(start_time * resolution.item()):int(end_time * resolution.item())]
        key = key
        data = data[idx][:, :round(num_seconds * resolution.item())].detach().cpu().numpy()
    else:
        start_time = start_time[idx].item()
        end_time = end_time[idx].item()
        if output is not None:
            output = output[idx]
        key = key[idx]
        data = data[idx].detach().cpu().numpy()
    softmax_scores = None
    if output is not None:
        softmax_scores = torch.nn.functional.softmax(output, dim=-1).detach().cpu().numpy()
    # fig, ax = plt.subplots(3, 1, figsize=fig_size)
    fig = plt.figure(figsize=fig_size)
    plt.rcParams['font.family'] = 'Times New Roman'
    num_rows = 5 if plot_pitch else 3
    height_ratios = [0.5, 1, 0.5] if not plot_pitch else [0.5, 1, 0.5, 0.2, 0.2]
    if probs_and_entropy is not None:
        num_rows += 2
        height_ratios += [0.2, 0.2]
    gs = gridspec.GridSpec(num_rows, 1, height_ratios=height_ratios)
    ax = [plt.subplot(gs[i]) for i in range(num_rows)]
    mapping = get_id_to_token_mapping(cfg)
    texts = texts[idx].split(cfg.data.text_delim)
    sr = cfg.data.audio_params.target_sr if hasattr(cfg.data.audio_params, "target_sr") else cfg.data.audio_params.sr
    y = load_audio_segment(audio[idx], start_time, end_time, sr)
    save_data = {}
    save_data["audio"] = y
    ax[0].plot(y)
    ax[0].set_title(key)
    ax[0].set_xlim(0, len(y))
    y_min, y_max = min_max_without_outliers_window(y, window_size=100)
    ax[0].set_ylim(y_min, y_max)
    save_data["audio_min_max"] = (y_min, y_max)
    ax[0].set_xticks([x  for x in range(len(y)) if x % sr == 0], [x / sr for x in range(len(y)) if x % sr == 0], rotation=45)
    # colors = plt.cm.cividis(np.linspace(0, 1, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    save_data["softmax_scores"] = softmax_scores
    save_data["mapping"] = mapping
    save_data["colors"] = colors
    save_data["sr"] = sr
    if latency_list_full is not None:
        save_data["latency_list_full"] = latency_list_full
    if softmax_scores is not None:
        for i in range(softmax_scores.shape[-1]):
            ax[1].plot(softmax_scores[:, i], label=mapping[i], color=colors[i])
    if p_now is not None:
        ax[1].plot(p_now, label="p_now", color='black', linestyle='--')
    labels = label[idx].detach().cpu().numpy()
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            segments.append((labels[i-1], start, i))
            start = i
    segments.append((labels[-1], start, len(labels)))
    save_data["segments"] = segments
    ##now plot vline on segment boundary and add text for the segment in the middle
    softmax_range = np.max(softmax_scores) - np.min(softmax_scores) if softmax_scores is not None else 1
    extra_row_height = softmax_range * 0.2
    total_row_height = np.max(softmax_scores) + extra_row_height if softmax_scores is not None else 1
    save_data["total_row_height"] = total_row_height
    for _idx, seg in enumerate(segments):
        ax[1].vlines(seg[1:3], ymin=np.min(softmax_scores) if softmax_scores is not None else 0, ymax=total_row_height, color='black', linestyles='dashed')
        if seg[0] == -1:
            continue
        ax[1].text(seg[1] + (seg[2] - seg[1]) / 3, np.mean(softmax_scores) if softmax_scores is not None else 0.5, mapping[seg[0]], color='r', rotation=90)        
        ####################
        if texts[_idx].strip() == "":
            continue
        text_width = (seg[2] - seg[1]) / 4
        lines = []
        current_line = ""        
        for word in texts[_idx].split():
            if len(current_line) + len(word) + 1 <= text_width:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        if current_line:
            lines.append(current_line)
        max_len = 20
        y_range = np.linspace(np.min(softmax_scores) if softmax_scores is not None else 0, total_row_height, max(len(lines), max_len))
        for idx_, line in enumerate(lines):
            if idx_ >= max_len-1:
                line += "..."
            if idx > max_len - 1:
                break
            ax[1].text(x=seg[1]+2, y=y_range[max_len - idx_ - 1], s=line, ha='left', va='center', wrap=True)
    high_latency = False
    if latency_list is not None:
        save_data["latency_list"] = latency_list
        save_data["latency_index_list"] = latency_index_list
        for i, latency in enumerate(latency_list):
            previous_start = latency_index_list[i]["previous_turn_start"]
            latency_index = latency_index_list[i]["true_turn_end"] + latency
            if latency > 20:
                high_latency = True
            ax[1].vlines(previous_start, ymin=np.min(softmax_scores) if softmax_scores is not None else 0, ymax=total_row_height, color='blue', linestyles='dashed')
            ax[1].vlines(latency_index, ymin=np.min(softmax_scores) if softmax_scores is not None else 0, ymax=(np.min(softmax_scores) if softmax_scores is not None else 0 + np.max(softmax_scores) if softmax_scores is not None else 1) / 2, color='black', linestyles='dashed')
            ax[1].hlines([np.min(softmax_scores), (np.min(softmax_scores) if softmax_scores is not None else 0 + np.max(softmax_scores)) if softmax_scores is not None else 1 / 2], xmin=latency_index, xmax=latency_index - latency, color='black', linestyles='dashed')
            ax[1].scatter(latency_index, (np.min(softmax_scores) if softmax_scores is not None else 0 + np.max(softmax_scores) if softmax_scores is not None else 1) / 2, color='black', marker='*', s=100)
            if latency > 0:
                ax[1].text(latency_index + 3, (np.min(softmax_scores) if softmax_scores is not None else 0 + np.max(softmax_scores) if softmax_scores is not None else 1) / 2, f"{latency}", color='black', fontsize=12)
    
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    
    if softmax_scores is not None:
        ax[1].set_xticks(np.arange(0, softmax_scores.shape[0], tick_period))
        ax[1].set_xlim(0, softmax_scores.shape[0])
        ax[1].set_ylim(np.min(softmax_scores), total_row_height)
    if p_now is not None:
        ax[1].set_xticks(np.arange(0, len(p_now), tick_period))
        ax[1].set_xlim(0, len(p_now))
        
    ax[1].legend(prop={'size': 12})
    save_data["feat_data"] = data
    if data.shape[0] == 2:
        data = data[0]
    ax[2].imshow(np.log(np.abs(data)), origin='lower')
    ax[2].set_aspect('auto')
    
    ax[0].grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
    ax[1].grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
    ax[2].grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
    
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    
    if cfg.run_params.log_img_to_wandb:
        if wandb is not None:
            wandb.log_plots(plt, name="pred")

    plt.savefig(os.path.join(save_folder, save_name))
    torch.save(save_data, os.path.join(save_folder, save_data_name))
