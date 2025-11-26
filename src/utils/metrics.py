import torch

def evaluate_latency_parallel(cfg, softmax_out, label, turn_id, prev_turn_id, max_latency, label_data):
    """
    This is a parallelized version of evaluate_latency function across thresholds
    It computes the latency for all thresholds in one go
    Args:
        cfg: configuration object
        softmax_out: Tensor of shape (num_frames,) - model output probabilities
        label: Tensor of shape (num_frames,) - ground truth labels
        turn_id: int - the id of the turn to evaluate
        prev_turn_id: int - the id of the previous turn
        max_latency: int - maximum latency in frames
        label_data: list of tuples - each tuple contains (turn_id, start_time, end_time, text)
    Returns:
        latency_list: dict - keys are thresholds, values are lists of latencies for each occurrence
        latency_list_tensor: numpy array - shape (num_thresholds, num_occurrences)
        index_list: list of dicts - each dict contains information about the occurrence
    """
    softmax_out = softmax_out.detach().cpu()
    thresholds = torch.linspace(cfg.infer_params.threshold_range[0], cfg.infer_params.threshold_range[1], cfg.infer_params.threshold_range[2])
    thresholded_softmax_out = softmax_out > thresholds.unsqueeze(1) #num_thresholds x num_frames
    latency_list, index_list = [], []
    for i in range(softmax_out.shape[1]): ## We iterate over frames - and check for turn ends
        if label[i] != turn_id:
            continue
        if i > 0:
            if label[i-1] == turn_id:
                continue
        j, prev_turn_start = 1, 0
        while i - j > 0:
            if label[i-j] != prev_turn_id:
                prev_turn_start = i - j + 1   
                break
            j += 1

        j, next_turn_start = 1, 0
        while i + j < len(label):
            if label[i+j] != turn_id:
                next_turn_start = i + j
                break
            j += 1
        max_latency = min(max_latency, next_turn_start - i) if next_turn_start > 0 else max_latency        
        max_pre_frame = max(0, prev_turn_start)
        max_post_frame = min(len(label), i + max_latency)
        pre_post_tensor = thresholded_softmax_out[:, max_pre_frame:max_post_frame].int()
        all_occurrences_ = torch.where(pre_post_tensor.any(dim=1), torch.argmax(pre_post_tensor, dim=1), torch.tensor(max_latency+(i if max_latency-i > 0 else (i - prev_turn_start)))) 
        if i - max_latency > 0:
            all_occurrences_ = all_occurrences_ - (i - prev_turn_start - max_latency)
        all_occurrences = all_occurrences_ - (i if max_latency-i > 0 else max_latency)
        
        latency_list.append(all_occurrences)
        segment_text = None
        start = label_data[0][1].item()
        for x in label_data:
            x_turn = x[0][0]
            if x_turn != cfg.infer_params.previous_turns[0]:
                continue
            end_time_stamp_of_segment = round((x[2].item() - start) * cfg.data.audio_params.freq)
            if abs(i - end_time_stamp_of_segment) < 3:
                segment_text = x[-1]
        assert segment_text is not None, f"{i}, {i / cfg.data.audio_params.hop_length}, {index_list[-1]}"  
        index_list.append(
            {
                "previous_turn_start": prev_turn_start,
                "true_turn_end": i,
                "text": segment_text,
                "start": start,
            }
        )        
    if latency_list == []:
        turns = [x[0][0] for x in label_data]
        if cfg.infer_params.score_turns[0] not in turns:
            return None, None, None
        else:
            if turn_id in label:
                raise ValueError("No turns found")
            else:
                return None, None, None
    latency_list_tensor = torch.stack(latency_list).T
    if hasattr(cfg.data.audio_params, "lookahead_frames"):
        latency_list_tensor = latency_list_tensor + cfg.data.audio_params.lookahead_frames
    latency_list = latency_list_tensor.tolist()
    thresholds = thresholds.tolist()
    latency_list = {
        round(thresholds[i], 2): latency_list[i] for i in range(len(thresholds)) 
    }
    return latency_list, latency_list_tensor.detach().cpu().numpy(), index_list
    