from easydict import EasyDict as edict
from src.training.default_trainer import load_default_trainer
import os
import torch
import yaml
import math
import torchaudio
from src.utils.logger import logger
from huggingface_hub import snapshot_download
from torch.nn import functional as F

def load_config(yamlFiles):
    """
    Load config from yaml file(s)
    Args:
        yamlFiles (list): List of paths to yaml config files
    Returns:
        cfg (edict): EasyDict containing the merged configurations
    """
    assert isinstance(yamlFiles, list) and len(yamlFiles) == 1, "Please provide a list with one config file path"
    cfg = {}
    for yamlFile in yamlFiles:
        assert os.path.exists(yamlFile), f"Config file not found: {yamlFile}"
        with open(yamlFile) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg = edict(cfg)
    return cfg    

def load_run(cfg):
    """
    Load model, trainer, and feature extractor from config
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: Loaded model
        cfg (edict): Updated config
        trainer: Trainer function
        feat_extractor: Feature extractor
    """
    feat_extractor = None
    if hasattr(cfg, "infer_params"):
        logger.info("Loading inference config")
        infer_folder = os.path.join(cfg.infer_params.root_path, cfg.infer_params.checkpoint_folder, cfg.wandb.run_name)
        infer_cfg_files = [os.path.join(infer_folder, files) for files in os.listdir(infer_folder) if files.endswith(".yaml")]
        infer_cfg = load_config(infer_cfg_files)
        infer_modes = cfg.data.modes
        infer_wandb_mode = cfg.wandb.use_wandb
        infer_samples = cfg.data.num_samples
        infer_device = cfg.infer_params.device
        path = cfg.infer_params.root_path
        cfg = edict({**cfg, **infer_cfg})
        cfg.data.modes = infer_modes
        cfg.infer_folder = infer_folder
        cfg.run_params.infer = True
        cfg.run_params.device = infer_device
        cfg.run_params.batch_size = cfg.infer_params.batch_size
        cfg.wandb.use_wandb = infer_wandb_mode
        cfg.data.num_samples = infer_samples
        cfg.data.root_path = path
    if cfg.data.audio_params.audio_feature  not in  ["logmel", "logmel-v2"]:
        feat_extractor = globals()[cfg.data.audio_params.audio_feature](cfg)
        logger.info(f"Using {cfg.data.audio_params.audio_feature} as feature extractor")
    model = globals()[cfg.model.name](cfg)
    trainer = get_trainer(cfg)
    return model, cfg, trainer, feat_extractor

def get_trainer(cfg):
    """
    Get trainer function based on model name
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        trainer: Trainer function
    """
    if cfg.model.name in ["base_lstm", "ms_lstm_vap", "ms_lstm", "linear", "reformer", "mamba"]:
        return load_default_trainer
    else:
         raise NotImplementedError(f"Trainer not implemented for {cfg.model.name}")

def get_feat_size(cfg):
    """
    Get feature size based on audio feature type
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        cfg (edict): Updated config with input_size
    """
    if cfg.data.audio_params.audio_feature  in ["logmel", "logmel-v2"]:
        cfg.model_params.input_size = cfg.data.audio_params.n_mels
    else:
        cfg.model_params.input_size = cfg.data.audio_params.feat_size
    return cfg

def ms_lstm(cfg):
    """
    Get MS-LSTM model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: MS-LSTM model
    """
    from src.models.ms_lstm import MS_LSTM_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.special_tokens.keys())
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if cfg.model.use_loss_weights:
        cfg.model_params.loss_weights = torch.tensor(cfg.model.loss_weight_factors)
    if hasattr(cfg.data.audio_params, "lookahead_frames"):
        cfg.model_params.lookahead_frames = cfg.data.audio_params.lookahead_frames
    return MS_LSTM_Model(
        use_mel_embed=cfg.model.use_mel_embed,
        **cfg.model_params
    )
def ms_lstm_vap(cfg):
    """
    Get MS-LSTM for Voice Activity Prediction model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: MS-LSTM VAP model
    """
    from src.models.vap_lstm import MS_LSTM_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.special_tokens.keys())
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if cfg.model.use_loss_weights:
        cfg.model_params.loss_weights = torch.tensor(cfg.model.loss_weight_factors)
    return MS_LSTM_Model(
        use_mel_embed=cfg.model.use_mel_embed,
        **cfg.model_params
    )

def base_lstm(cfg):
    """
    Get Base LSTM model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: Base LSTM model
    """
    from src.models.base_lstm import LSTM_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.special_tokens.keys())
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if cfg.model.use_loss_weights:
        cfg.model_params.loss_weights = torch.tensor(cfg.model.loss_weight_factors)
    return LSTM_Model(
        use_mel_embed=cfg.model.use_mel_embed,
        use_system_embed=cfg.model.use_system_ip_embed, 
        **cfg.model_params
    )

def linear(cfg):
    """
    Get Linear model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: Linear model
    """
    from src.models.linear import Linear_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.special_tokens.keys())
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if cfg.model.use_loss_weights:
        cfg.model_params.loss_weights = torch.tensor(cfg.model.loss_weight_factors)
        
    return Linear_Model(
        use_mel_embed=cfg.model.use_mel_embed,
        use_system_embed=cfg.model.use_system_ip_embed, 
        **cfg.model_params
    )

def mimi(cfg):
    """
    Get Mimi model (https://huggingface.co/kyutai/mimi)
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        mimi_encode: Mimi encoder function
    """
    from transformers import MimiModel, AutoFeatureExtractor
    model = MimiModel.from_pretrained(cfg.data.audio_params.model_repo)
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.data.audio_params.model_repo)
    
    class encode():
        def __init__(
            self, 
            model, 
            processor, 
            sr, 
            device, 
            upsample=False,
            num_quantizers=2,
            no_quantisation=False,
            downsample=False,
            use_transformer=False,
        ):  
            model.config.num_quantizers = num_quantizers
            self.model = model
            self.feature_extractor = processor
            self.model.eval()
            self.device = cfg.run_params.device
            if not torch.cuda.is_available():
                if self.device != 'cpu':
                    logger.warning("CUDA not available, switching to CPU")
                self.device = 'cpu'
            self.model.to(self.device)
            self.model.quantizer.to(self.device)
            self.upsample = upsample
            self.sr = sr
            self.num_quantizers = num_quantizers
            self.no_quantisation = no_quantisation
            self.downsample = downsample
            self.use_transformer = use_transformer
            assert self.sr == self.feature_extractor.sampling_rate
            
        def __call__(self, wav):
            inputs = self.feature_extractor(raw_audio=wav, sampling_rate=self.sr, return_tensors="pt")
            inputs["input_values"] = inputs["input_values"].to(self.device)
            if self.no_quantisation:
                assert self.upsample == False
                padding_mask = torch.ones_like(inputs["input_values"]).bool()
                encoder_past_key_values = None
                return_dict = False
                embeddings = model.encoder(inputs["input_values"])
                if self.use_transformer:
                    embeddings = self.model.encoder_transformer(
                        embeddings.transpose(1, 2), 
                        past_key_values=encoder_past_key_values, 
                        return_dict=return_dict
                    )
                    embeddings = embeddings[0].transpose(1, 2)
                    if self.downsample:
                        embeddings = self.model.downsample(embeddings)
                else:
                    assert self.downsample == False

            else:
                codes = self.model.encode(inputs["input_values"], num_quantizers=self.num_quantizers).audio_codes
                embeddings = self.model.quantizer.decode(codes)
                if self.upsample:
                    embeddings = self.model.upsample(embeddings)
            return embeddings.squeeze()

    no_quantisation = False
    if hasattr(cfg.data.audio_params, "no_quantisation"):
        if cfg.data.audio_params.no_quantisation:
            no_quantisation = True
    downsample = False
    if hasattr(cfg.data.audio_params, "downsample"):
        if cfg.data.audio_params.downsample:
            downsample = True
    use_transformer = False
    if hasattr(cfg.data.audio_params, "use_transformer"):
        if cfg.data.audio_params.use_transformer:
            use_transformer = True
            
    mimi_encode = encode(
        model, 
        feature_extractor, 
        sr=cfg.data.audio_params.target_sr, 
        device=cfg.run_params.device,
        upsample=cfg.data.audio_params.upsample,
        num_quantizers=cfg.data.audio_params.num_quantisers,
        no_quantisation=no_quantisation,
        downsample=downsample,
        use_transformer=use_transformer,
    )
    return mimi_encode

def AudioDec(cfg):
    """
    Get AudioDec model 
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        codec_encode: AudioDec encoder function
    """
    from AudioDec.utils.audiodec import AudioDec, assign_model
    sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(cfg.data.audio_params.model_name)  
    audiodec = AudioDec(tx_device=cfg.run_params.device, rx_device=cfg.run_params.device)
    audiodec.load_transmitter(encoder_checkpoint)
    audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)
    
    class encode():
        def __init__(
            self, 
            model, 
            nq, 
            sr, 
            device, 
            reduction,
            downsample,
            kernal_size,
            stride
        ):
            self.model = model
            self.device = device
            self.nq = nq
            self.reduction = reduction
            self.sr = sr
            self.kernal_size = kernal_size
            self.stride = stride
            self.downsample = downsample
            assert reduction in ["sum"]
            
        def __call__(self, wav_24kHz):
            wav_24kHz = wav_24kHz[None, None, :].to(self.device)
            z = audiodec.tx_encoder.encode(wav_24kHz)
            idx, (probs, entropy) = audiodec.tx_encoder.quantize(z, return_probs=True)
            if self.nq != 8:
                assert self.nq < 8, "Invalid number of quantisers"
                idx = idx[:self.nq, :]
            code_vectors = audiodec.rx_encoder.lookup(idx).squeeze()
            assert self.reduction in ["sum"], "Only sum reduction is supported - inbuilt"
            code_vectors = code_vectors.squeeze().permute(1, 0)
            if self.downsample:
                code_vectors = self.causal_avg_pool(code_vectors.unsqueeze(0), self.kernal_size, self.stride)
            return code_vectors.squeeze()
    
        def causal_avg_pool(self, input, kernel_size, stride):
            padding = kernel_size - 1
            input_padded = F.pad(input, (padding, 0), mode='constant', value=0)
            return F.avg_pool1d(input_padded, kernel_size, stride=stride)
        
    codec_encode = encode(
        audiodec, 
        nq=cfg.data.audio_params.nq,
        sr=cfg.data.audio_params.target_sr, 
        device=cfg.run_params.device,
        reduction=cfg.data.audio_params.reduction,
        downsample=cfg.data.audio_params.downsample,
        kernal_size=cfg.data.audio_params.kernel_size,
        stride=cfg.data.audio_params.stride
    )
    return codec_encode

    
def Encodec(cfg):
    """
    Get Encodec model (https://huggingface.co/docs/transformers/en/model_doc/encodec)
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        codec_encode: Encodec encoder function
    """
    from transformers import EncodecModel, AutoProcessor
    model = EncodecModel.from_pretrained(cfg.data.audio_params.model_repo)
    processor = AutoProcessor.from_pretrained(cfg.data.audio_params.model_repo)
    num_quantizer2bw = {
        2: 1.5,
        4: 3,
        8: 6,
        16: 12,
        32: 24
    }
    bw = num_quantizer2bw[cfg.data.audio_params.num_quantisers]
    class encode():
        def __init__(
            self, 
            model, 
            processor, 
            bw, 
            sr, 
            device, 
            reduction,
            downsample,
            kernal_size,
            stride
        ):
            self.model = model
            self.model.to(device)
            self.model.eval()
            self.device = device
            self.processor = processor
            self.bw = bw
            self.reduction = reduction
            self.sr = sr
            self.kernal_size = kernal_size
            self.stride = stride
            self.downsample = downsample
            assert reduction in ["sum"]
            
        def __call__(self, wav_24kHz):
            inputs = processor(raw_audio=wav_24kHz, return_tensors="pt", sampling_rate=self.sr)
            inputs["input_values"] = inputs["input_values"].to(self.device)
            encoder_outputs = self.model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=self.bw)
            code_vectors = self.model.quantizer.decode(encoder_outputs.audio_codes.squeeze(0))
            if self.reduction == "sum":
                code_vectors = code_vectors.sum(dim=0)
            if self.downsample:
                code_vectors = self.causal_avg_pool(code_vectors.unsqueeze(0), self.kernal_size, self.stride)
            return code_vectors.squeeze()
    
        def causal_avg_pool(self, input, kernel_size, stride):
            padding = kernel_size - 1
            input_padded = F.pad(input, (padding, 0), mode='constant', value=0)
            return F.avg_pool1d(input_padded, kernel_size, stride=stride)
        
    codec_encode = encode(
        model, 
        processor, 
        bw, 
        sr=cfg.data.audio_params.target_sr, 
        device=cfg.run_params.device,
        reduction=cfg.data.audio_params.reduction,
        downsample=cfg.data.audio_params.downsample,
        kernal_size=cfg.data.audio_params.kernel_size,
        stride=cfg.data.audio_params.stride
    )
    return codec_encode    


    

    