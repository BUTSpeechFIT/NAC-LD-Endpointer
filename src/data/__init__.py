
import torch
from src.data.data_processing import process_vad, handle_and_add_turns, endpointing_dataset, endpointing_dataset_full_context
import multiprocessing

def load_data(cfg, feat_extractor):
    """
    Load data loaders based on the dataset specified in the config
    Args:
        cfg: Configuration object
        feat_extractor: Feature extractor object
    Returns:
        loaders (dict): Dictionary containing data loaders for each mode
    """

    supported_datasets = ['spokenwoz', 'humdial']
    dataset_instances, loaders = {}, {}
    for dataset in cfg.data.datasets:
        assert dataset in supported_datasets, f"Dataset {dataset} not supported. Supported datasets: {supported_datasets}"
        preproc_fn_name = f"preprocess_{dataset}"
        dataset_file = f"src.data.{dataset}_dataset"
        from importlib import import_module
        dataset_module = import_module(dataset_file)
        ### We need to convert the raw dataset to a fixed format first
        ### We expect the dataset to have the following format after preprocessing:
        ### Each audio file will have a corresponding .json file with the same name
        ### json format:
        ### {
        ###   "audio_filepath": "path/to/audio/file.wav",
        ###   "segments": [
        ###       {"start": xx.yy, "end": zz.ww, "turn": "...", "text": "..."},
        ###       ...
        ###   ],
        ### Where "segments" contain the turn-level information including start and end times
        ### Turn corresponds to label such as user, system, etc
        ### We will save these preprocessed files for further processing
        getattr(dataset_module, preproc_fn_name)(cfg) #here we preprocess and standardize the dataset
        process_vad(cfg, dataset) #here we use VAD to trim beginning and end silences for each segment
        handle_and_add_turns(cfg, dataset) #here we add all missing turns to the segments
        for mode in cfg.data.modes:
            dataset_class = endpointing_dataset
            if hasattr(cfg, "infer_params"):
                # dataset_class = getattr(dataset_module, f"{dataset}_dataset_infer")(cfg, mode, feat_extractor)
                dataset_class = endpointing_dataset_full_context
            dataset_instance = dataset_class(cfg, mode, dataset, feat_extractor)
            dataset_instances.setdefault(mode, []).append(dataset_instance)
    for mode in dataset_instances:
        dataset_instance = torch.utils.data.ConcatDataset(dataset_instances[mode])
        loaders[mode] = torch.utils.data.DataLoader(
            dataset_instance,
            batch_size=cfg.run_params.batch_size,
            shuffle=True if mode == 'train' else False,
            collate_fn=None, # We train with fixed length segments, so no need for custom collate_fn
            pin_memory=False
        )
    return loaders