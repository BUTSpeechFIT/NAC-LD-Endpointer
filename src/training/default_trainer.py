import os
import torch
import numpy as np
from tqdm import tqdm
from src.utils import metrics
from src.utils.logger import logger
from sklearn.metrics import confusion_matrix
from src.utils.wandb_logger import WandbLogger

from src.utils.run_utils import (
    load_checkpoint, 
    save_checkpoint, 
    setup_save_folder, 
    prediction_visualization,
    plot_and_save_metrics,
)

from src.utils.data_utils import (
    get_id_to_token_mapping, 
    get_token_to_id_mapping, 
)

class DefaultTrainer():
    def __init__(self, cfg, model, config_paths):
        self.cfg = cfg
        if not torch.cuda.is_available():
            cfg.run_params.device = "cpu"
            logger.warning("CUDA not available, using CPU.")
        self.model = model.to(cfg.run_params.device)
        print(model)
        os.environ["WANDB_SILENT"] = "true"
        if not cfg.wandb.use_wandb:
            os.environ["WANDB_MODE"] = "offline"
            
        self.wandb_logger = WandbLogger(cfg)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {num_params / 1_000_000:.2f}M")
        self.wandb_logger.summary({"num_params": num_params})
        if not hasattr(cfg, "infer_params"):
            self.save_folder = setup_save_folder(cfg, config_paths)
            self.best_loss = np.inf
            self.not_improved = 0
            self.best_acc = 0
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.run_params.lr)
            if hasattr(self.model, "scheduler"):
                raise NotImplementedError("Scheduler not implemented")
    
    def handle_delay(self, label):
        """
        Handle delay frames in the label data.
        Args:
            label: Tensor containing label data.
        Returns:
            Tuple of (modified label, delay frames).
        """
        delay_frames = None
        if hasattr(self.cfg.data.audio_params, "delay_frames"):
            delay_frames = self.cfg.data.audio_params.delay_frames
            if not hasattr(self.cfg.data.audio_params, "delay_prob"):
                label = torch.cat([torch.tensor([-1]*delay_frames).to(label.device).unsqueeze(0).repeat(len(label), 1), label[:, :-delay_frames]], dim=-1)
            else:
                label_ = []
                ###generate a list of 0, 1 from binomial distribution with p = delay_prob
                delay_prob = torch.bernoulli(torch.full((len(label), ), self.cfg.data.audio_params.delay_prob)).long()
                for i in range(len(label)):
                    if delay_prob[i] == 1:
                        label_.append(torch.cat([torch.tensor([-1]*delay_frames).to(label.device), label[i, :-delay_frames]]))
                    else:
                        label_.append(label[i])
                label = torch.stack(label_)
        return label, delay_frames

    def get_system_ids(self, data):
        system_id = get_token_to_id_mapping(self.cfg)[self.cfg.data.special_tokens.system]
        ###in label, whenever we have a match with system id, it is 1 else 0, in a new tensor
        ids = (data == system_id).long()
        if hasattr(self.cfg, "infer_params"):
            if hasattr(self.cfg.infer_params, "system_stream") and hasattr(self.cfg.infer_params, "infer_system_ids"):
                if not self.cfg.infer_params.system_stream and not self.cfg.infer_params.infer_system_ids:
                    ids = torch.zeros_like(ids)
        return ids

    def handle_system(self, label):
        if hasattr(self.cfg.model, "ignore_system_labels"):
            if self.cfg.model.ignore_system_labels:
                system_id = get_token_to_id_mapping(self.cfg)[self.cfg.data.special_tokens.system]
                label[label == system_id] = -1
                system_end_id = get_token_to_id_mapping(self.cfg)[self.cfg.data.special_tokens.system_end]
                label[label == system_end_id] = -1
        return label
        
    def train(self, mode, loader):
        self.handle_start_of_epoch(mode)
        pbar = tqdm(loader, desc=f"{mode}")
        for idx, data in enumerate(pbar):
            data, label, metadata = data
            data, label = self.to_device([data, label])
            if self.cfg.model.use_system_ip_embed:
                system_ids = self.get_system_ids(label)
            output = self.model(data, init_hidden=True, system_ids=system_ids)
            label_, delay_frames = self.handle_delay(label)
            label_ = self.handle_system(label_)
            loss, label_info = self.model.loss(output, label_, delay_frames)
            if mode == "train":
                self.optimizer.zero_grad()
                loss["total"].backward()   
                self.optimizer.step()      
            pbar.set_description(f"Mode: {mode}, Loss: {loss['total'].item():.4f}, Acc: {loss['accuracy'].item():.4f}")  
            self.handle_batch_loss(loss, label_info)
        self.handle_end_of_epoch(data, label, output, metadata)
        return {"acc": self.avg_user_accuracy, "loss": self.epoch_metrics[f"{mode}_total"]}
    
    def handle_start_of_epoch(self, mode):
        self.mode = mode
        if mode == "train": 
            self.model.train()
        else: 
            self.model.eval()
        self.epoch_metrics = {}
        self.cm = {"gnd": [], "pred": []}
        
    def handle_batch_loss(self, loss, label_info):
        for key in loss:
            if f"{self.mode}_"+key not in self.epoch_metrics:
                self.epoch_metrics[f"{self.mode}_"+key] = []
            self.epoch_metrics[f"{self.mode}_"+key].append(loss[key].item())
        self.cm["gnd"].extend(label_info[0])
        self.cm["pred"].extend(label_info[1])
        
    def handle_end_of_epoch(self, data, label, output, metadata, idx=0):
        self.epoch_metrics.pop(f"{self.mode}_accuracy")
        for key in self.epoch_metrics:
            self.epoch_metrics[key] = round(sum(self.epoch_metrics[key]) / len(self.epoch_metrics[key]), 4)
        logger.info(f"{self.mode} epoch: {self.epoch_metrics}")
        self.wandb_logger.log(self.epoch_metrics)   
        cm = confusion_matrix(self.cm["gnd"], self.cm["pred"], labels=[0, 1, 2, 3])
        accuracy = np.trace(cm) / np.sum(cm)
        class_wise_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        mapping = get_id_to_token_mapping(self.cfg)
        class_labels = list(mapping.values())
        class_wise_accuracy = {class_labels[i]: round(class_wise_accuracy[i]*100, 2) for i in range(len(class_wise_accuracy))}
        user_end_turn_accuracy = class_wise_accuracy[self.cfg.data.special_tokens.user_end]
        user_accuracy = class_wise_accuracy[self.cfg.data.special_tokens.user]
        avg = (user_end_turn_accuracy + user_accuracy) / 2
        self.avg_user_accuracy = avg
        self.wandb_logger.log({f"{self.mode}_accuracy_cm": round(accuracy*100, 2)})
        self.wandb_logger.log({f"{self.mode}_class_wise_accuracy": class_wise_accuracy})
        
        if self.mode != "val":
            return
        
        prediction_visualization(
            self.cfg, 
            data, 
            label, 
            output, 
            metadata,
            self.save_folder, 
            self.wandb_logger,
            idx=idx
        )
        
    def to_device(self, data):
        return [x.to(self.cfg.run_params.device) for x in data if isinstance(x, torch.Tensor)]
    
    def checkpoint_handler(self, metric):
        save_criteria = False
        if self.cfg.run_params.save_best_from_val_loss:
            if metric["loss"] < self.best_loss:
                self.best_loss = metric["loss"]
                save_criteria = True
        
        if self.cfg.run_params.save_best_from_val_acc: 
            if metric["acc"] > self.best_acc:
                logger.info(f"Accuracy updated: {self.best_acc} -> {metric['acc']}")
                self.best_acc = metric["acc"]
                save_criteria = True      
                self.not_improved = 0
            else:
                self.not_improved += 1
        if self.cfg.run_params.early_stopping:
            if self.not_improved >= self.cfg.run_params.early_stopping_patience:
                logger.info(f"Early stopping at epoch: {self.epoch}")
                exit()
            else:
                logger.info(f"Epochs without improvement: {self.not_improved}")
        
        if not save_criteria:
            return         
            
        if self.epoch % self.cfg.run_params.epoch_save_interval == 0:
            save_criteria = True
        else:
            save_criteria = False
        
        if save_criteria:
            save_checkpoint(self.cfg, self.epoch, self.model, self.save_folder)
    
    def infer_loop(self, loader, ):
        latency_list_all = {}
        for idx, data in enumerate(tqdm(loader, desc="Infer")):
            if len(data) == 3:
                data, label, metadata = data
                codec_probs, codec_entropy = None, None
            elif len(data) == 4:
                data, label, metadata, (codec_probs, codec_entropy) = data
            else:
                raise NotImplementedError(f"Data length not implemented: {len(data)}")
            data, label = self.to_device([data, label])
            if self.cfg.model.use_system_ip_embed:
                system_ids = self.get_system_ids(label)
            output = self.model.infer(data, init_hidden=True, system_ids=system_ids)

            assert output.size(-1) == len(self.cfg.data.special_tokens.keys()), f"Output size mismatch: {output.size(-1)} != {len(self.cfg.data.special_tokens.keys())}"
            audio, label_data, texts, start_time, end_time, key = metadata
            total_time = end_time - start_time
            num_output_frames = output.size(1)
            num_frames_per_second = round(num_output_frames / total_time.item())
            max_latency = int(num_frames_per_second * self.cfg.infer_params.max_latency)
            soft_output = torch.softmax(output, dim=-1)
            assert len(self.cfg.infer_params.score_turns) == 1, f"Score turns not implemented for more than 1 turn - {self.cfg.infer_params.score_turns}, {len(self.cfg.infer_params.score_turns)}"
            turn = self.cfg.infer_params.score_turns[0]
            prev_turn = self.cfg.infer_params.previous_turns[0]
            prev_turn_id = get_token_to_id_mapping(self.cfg)[prev_turn]
            turn_id = get_token_to_id_mapping(self.cfg)[turn]
            system_ids = self.get_system_ids(label)
            latency_list_parallel, latency_list_parallel_numpy, index_list_parallel = metrics.evaluate_latency_parallel(
                self.cfg, 
                soft_output[:, :, turn_id], 
                label.squeeze(), 
                turn_id, 
                prev_turn_id, 
                max_latency, 
                label_data
            )
            
            if latency_list_parallel is None:
                continue
            
            if idx < self.cfg.infer_params.num_infer_pred_imgs:
                prediction_visualization(
                    self.cfg, 
                    data, 
                    label, 
                    output, 
                    metadata,
                    self.cfg.infer_folder, 
                    None,
                    idx=0,
                    latency_index_list=index_list_parallel,
                    latency_list=latency_list_parallel[0.7],
                    fig_size=(120, 10),
                    save_name=f"infer_prediction{idx+1}.png",
                    save_data_name=f"infer_prediction_data{idx+1}.pt",
                    plot_pitch=False,
                    probs_and_entropy=(codec_probs, codec_entropy),
                    latency_list_full=latency_list_parallel,
                )
            
            if self.cfg.infer_params.generate_error_vs_silence:
                metrics.generate_error_vs_silence(
                    cfg=cfg,
                    latency_index_list=index_list_parallel,
                    latency_list=latency_list_parallel,
                    metadata=metadata,
                )
            
            if self.cfg.infer_params.asr_eval:
                self.asr_eval.eval_single(self.cfg, key, audio, index_list_parallel, latency_list_parallel_numpy, thresholds=list(latency_list_parallel.keys()))
            for threshold in latency_list_parallel:
                if threshold not in latency_list_all:
                    latency_list_all[threshold] = []
                latency_list_all[threshold].extend(latency_list_parallel[threshold])
                # print(latency_list_all)
        ep_cutoff, positive_median_latency, positive_ep_90, all_median_latency, all_ep_90  = {}, {}, {}, {}, {}
        for threshold in latency_list_all:
            ep_cutoff[threshold] = (len([x for x in latency_list_all[threshold] if x < 0])*100) / len(latency_list_all[threshold])
            # median_latency[threshold] = np.median(latency_list_all[threshold]) * (1000 / self.cfg.data.audio_params.freq)
            # ep_90[threshold] = np.percentile(latency_list_all[threshold], 90) * (1000 / self.cfg.data.audio_params.freq)
            positive_latency = [x for x in latency_list_all[threshold] if x >= 0]
            positive_median_latency[threshold] = np.median(positive_latency) * (1000 / self.cfg.data.audio_params.freq)
            positive_ep_90[threshold] = np.percentile(positive_latency, 90) * (1000 / self.cfg.data.audio_params.freq)
            # print(median_latency)
            all_median_latency[threshold] = np.median(latency_list_all[threshold]) * (1000 / self.cfg.data.audio_params.freq)
            all_ep_90[threshold] = np.percentile(latency_list_all[threshold], 90) * (1000 / self.cfg.data.audio_params.freq)
        return ep_cutoff, positive_median_latency, positive_ep_90, all_median_latency, all_ep_90 
            
    def infer(self, loader):        
        
        if not self.cfg.infer_params.asr_offline_eval_only:
            self.model = load_checkpoint(
                path=os.path.join(self.cfg.infer_folder, self.cfg.infer_params.infer_checkpoint_name),
                model=self.model
            )
            if hasattr(self.cfg.infer_params, "system_stream"):
                if not self.cfg.infer_params.system_stream:
                    self.cfg.infer_folder = self.cfg.infer_folder + "_disable_system"
                if hasattr(self.cfg.infer_params, "infer_system_ids"):
                    if not self.cfg.infer_params.infer_system_ids:
                        self.cfg.infer_folder = self.cfg.infer_folder + "_disable_system_ids"
                os.makedirs(self.cfg.infer_folder, exist_ok=True)
            self.model.eval()
            
            if self.cfg.infer_params.asr_eval:
                if self.cfg.infer_params.asr_eval_online:
                    self.asr_eval = metrics.ASR_EVAL(self.cfg)
                else:
                    self.asr_eval = metrics.ASR_EVAL_OFFLINE(self.cfg)

            with torch.no_grad():
                ep_cutoff, positive_median_latency, positive_ep_90, all_median_latency, all_ep_90 = self.infer_loop(loader)
            score_metrics = {
                "median_latency_positive": positive_median_latency,
                "ep_cutoff": ep_cutoff,
                "ep_90_positive": positive_ep_90,
                "median_latency_all": all_median_latency,
                "ep_90_all": all_ep_90
            }
            if self.cfg.infer_params.asr_eval:
                if self.cfg.infer_params.asr_eval_online:
                    true_label_wer, true_label_cer, pred_label_wer, pred_label_cer = self.asr_eval.final_score()
                    score_metrics["wer_metrics"] = pred_label_wer
                    score_metrics["cer_metrics"] = pred_label_cer
                    score_metrics["true_label_wer"] = true_label_wer
                    score_metrics["true_label_cer"] = true_label_cer
                    print("turns missed due to bad clipping:", self.asr_eval.bad_clipping)
                
            plot_and_save_metrics(
                metrics=score_metrics,
                cfg=self.cfg
            )
        else:
            logger.info("Skipping infer loop, only ASR offline evaluation..")
        if self.cfg.infer_params.asr_eval:
            asr_metrics = metrics.asr_offline_score(self.cfg)
            plot_and_save_metrics(
                    metrics=None,
                    cfg=self.cfg,
                    read_other_metrics_from_file=True
                )
    
def load_default_trainer(cfg, model, loaders, config_paths):
    
    trainer = DefaultTrainer(cfg, model, config_paths)
    
    if hasattr(cfg, "infer_params"):
        test_loader = loaders["test"]
        trainer.infer(loader=test_loader)
        exit()
        
    train_loader = loaders["train"]
    dev_loader = loaders["val"]
    
    if cfg.run_params.load_checkpoint:
        load_checkpoint(
            path=cfg.run_params.load_checkpoint_path,
            model=trainer.model,
        )
    for epoch in tqdm(range(cfg.run_params.epochs)):
        trainer.epoch = epoch

        trainer.train(mode="train", loader=train_loader)
        val_metric = trainer.train(mode="val", loader=dev_loader)
        trainer.checkpoint_handler(val_metric)
        