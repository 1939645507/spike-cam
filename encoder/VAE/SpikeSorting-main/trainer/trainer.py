# T. S. Liang @ HKU EEE, Jan. 29th 2026
# Email: sliang57@connect.hku.hk
# Training script for Waveform/Channel Autoencoder.

import argparse
import logging
import os
import shutil
import math
import random
from datetime import datetime
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from safetensors.torch import save_file, load_file
from tqdm.auto import tqdm

# Accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.tensorboard import SummaryWriter

from models.BasicBlocks import Encoder, Decoder
from utils.data_loader import prepare_dataset
from sklearn.manifold import TSNE
import torch
if not hasattr(torch, 'xpu'):
    torch.xpu = None
# ==============================================================================
# Trainer Class
# ==============================================================================

class AutoencoderTrainer:
    
    def __init__(self, args, encoder_cls, decoder_cls):
        
        self.args = args
        
        # 1. -------------------- 基础配置 --------------------
        self.logging_dir = os.path.join(self.args.output_dir, self.args.logging_dir)
        self.accelerator_project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, 
            logging_dir=self.logging_dir
        )

        # 2. -------------------- 初始化 Accelerator (必须最先执行) --------------------
        # 这会建立分布式训练环境或单机加速环境
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=self.accelerator_project_config,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        # 3. -------------------- 初始化 Logger --------------------
        # 现在 Accelerator 已经就绪，可以安全地使用 get_logger 了
        self.logger = get_logger(__name__, log_level="INFO")
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        # 4. -------------------- 设备特定优化 --------------------
        if torch.cuda.is_available():
            # RTX 4090 等显卡的 TF32 优化
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("Using CUDA with TF32 optimization", main_process_only=True)
        elif torch.backends.mps.is_available():
            # Mac M4 的加速提示
            self.logger.info("Using Apple Silicon MPS acceleration", main_process_only=True)

        # 打印详细的加速器状态
        self.logger.info(self.accelerator.state, main_process_only=True)

        # 5. -------------------- 实验追踪 (TensorBoard) --------------------
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=self.logging_dir)
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)
        else:
            self.writer = None

        if self.args.seed is not None:
            set_seed(self.args.seed)

        # 6. -------------------- 初始化模型 --------------------
        self.logger.info("Initializing Encoder and Decoder...", main_process_only=True)
        
        # 传入参数初始化模型
        self.encoder = encoder_cls(waveform_len=args.waveform_len, num_channels=args.num_channels)
        self.decoder = decoder_cls(waveform_len=args.waveform_len, num_channels=args.num_channels)

        self.encoder.train()
        self.decoder.train()

        # 7. -------------------- 优化器 --------------------
        params_to_optimize = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        self.optimizer = AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # 8. -------------------- 学习率调度器 --------------------
        from diffusers.optimization import get_scheduler
        self.lr_scheduler = get_scheduler(
            "constant", # AE 训练通常使用 constant 或 linear
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    def train(self):
        
        logging.info("Start training")
        
        global_step = 0
        first_epoch = 0

        # -------------------- Checkpointing Hooks --------------------
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                # Unwrap models
                enc = self.accelerator.unwrap_model(self.encoder)
                dec = self.accelerator.unwrap_model(self.decoder)
                
                # Save Encoder
                enc_dir = os.path.join(output_dir, "encoder")
                os.makedirs(enc_dir, exist_ok=True)
                save_file(enc.state_dict(), os.path.join(enc_dir, "model.safetensors"))

                # Save Decoder
                dec_dir = os.path.join(output_dir, "decoder")
                os.makedirs(dec_dir, exist_ok=True)
                save_file(dec.state_dict(), os.path.join(dec_dir, "model.safetensors"))

                # Clear weights to avoid OOM
                weights.clear()
                torch.cuda.empty_cache()

        def load_model_hook(models, input_dir):
            # Clear current models
            # Note: In Accelerate's load_state, 'models' list usually contains the registered models
            # We manually load state dicts here.
            
            enc_path = os.path.join(input_dir, "encoder", "model.safetensors")
            dec_path = os.path.join(input_dir, "decoder", "model.safetensors")

            if os.path.exists(enc_path):
                state_dict_enc = load_file(enc_path, device="cpu")
                self.accelerator.unwrap_model(self.encoder).load_state_dict(state_dict_enc)
                del state_dict_enc
            
            if os.path.exists(dec_path):
                state_dict_dec = load_file(dec_path, device="cpu")
                self.accelerator.unwrap_model(self.decoder).load_state_dict(state_dict_dec)
                del state_dict_dec

            torch.cuda.empty_cache()

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

        # -------------------- Data Preparation --------------------
        train_loader, val_loader = prepare_dataset(train_path = args.trainset_path, val_path = args.valset_path)

        # Prepare everything with Accelerator
        self.encoder, self.decoder, self.optimizer, train_loader, val_loader, self.lr_scheduler = self.accelerator.prepare(
            self.encoder, self.decoder, self.optimizer, train_loader, val_loader, self.lr_scheduler
        )

        # Recalculate steps
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # -------------------- Resume --------------------
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting new.")
                initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])
                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        # -------------------- Training Loop --------------------
        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, self.args.num_train_epochs):
            
            self.encoder.train()
            self.decoder.train()
            
            train_loss = 0.0
            log_wf_loss = 0.0
            log_ch_loss = 0.0

            for step, batch in enumerate(train_loader):
                
                with self.accelerator.accumulate(self.encoder, self.decoder):
                    
                    wf_loss, ch_loss, total_loss = self.train_onestep(batch)

                    # Logging accumulator
                    avg_wf = self.accelerator.gather(wf_loss).mean()
                    avg_ch = self.accelerator.gather(ch_loss).mean()
                    
                    log_wf_loss += avg_wf.item() / self.args.gradient_accumulation_steps
                    log_ch_loss += avg_ch.item() / self.args.gradient_accumulation_steps
                    train_loss += total_loss.item() / self.args.gradient_accumulation_steps

                    self.accelerator.backward(total_loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            list(self.encoder.parameters()) + list(self.decoder.parameters()),
                            self.args.max_grad_norm
                        )
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    logs = {
                        "loss": train_loss,
                        "wf_loss": log_wf_loss,
                        "ch_loss": log_ch_loss,
                        "lr": self.lr_scheduler.get_last_lr()[0]
                    }
                    self.accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)
                    
                    # Reset logs
                    train_loss = 0.0
                    log_wf_loss = 0.0
                    log_ch_loss = 0.0

                    # Checkpointing & Validation
                    if global_step % self.args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            # 1. Validation
                            self.log_validation(val_loader, global_step)

                            # 2. Saving
                            self.manage_checkpoints(global_step)

                    if global_step >= self.args.max_train_steps:
                        break

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def train_onestep(self, batch):
        # Unpack batch: [B, 32], [B, 20]
        waveform, channel_max, channel_min = batch["waveform"], batch["channel_max"], batch["channel_min"]
        
        channel_max_min = torch.concat([channel_max, channel_min], dim = 1)
        # Forward Pass
        latent = self.encoder(waveform, channel_max_min) # [B, 8]
        recon_wf, recon_ch = self.decoder(latent)        # [B, 32], [B, 20]

        # Loss Calculation
        loss_wf = F.mse_loss(recon_wf, waveform, reduction="mean")
        loss_ch = F.mse_loss(recon_ch, channel_max_min, reduction="mean")

        # Weighted loss (can adjust weights in args if needed)
        total_loss = loss_wf + loss_ch

        return loss_wf, loss_ch, total_loss

    def log_validation(self, val_loader, global_step):
        self.logger.info("Running validation...")
        
        self.encoder.eval()
        self.decoder.eval()
        
        total_val_loss = 0.0
        num_batches = 0
        all_latents = [] # 用于存储所有的潜变量
        
        viz_dir = os.path.join(self.args.output_dir, "visualizations")
        if self.accelerator.is_main_process:
            os.makedirs(viz_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch in val_loader:
                waveform, channel_max, channel_min = batch["waveform"], batch["channel_max"], batch["channel_min"]
                channel_max_min = torch.concat([channel_max, channel_min], dim=1)

                waveform = waveform.to(self.accelerator.device)
                channel_max_min = channel_max_min.to(self.accelerator.device)

                latent = self.encoder(waveform, channel_max_min)
                recon_wf, recon_ch = self.decoder(latent)

                if self.accelerator.is_main_process:
                    all_latents.append(latent.cpu())

                loss = F.mse_loss(recon_wf, waveform, reduction="mean") + F.mse_loss(recon_ch, channel_max_min)
                total_val_loss += loss.item()
                
                # --- Visualization: Waveform reconstruction, Channel Max/Min ---
                if num_batches == 0 and self.accelerator.is_main_process:
                    num_samples_to_viz = min(4, waveform.shape[0])
                    
                    fig, axes = plt.subplots(num_samples_to_viz, 2, figsize=(15, 3 * num_samples_to_viz))
                    if num_samples_to_viz == 1: axes = np.expand_dims(axes, axis=0)
                    
                    mean = 1.4784424140308792
                    std = 104.25511002217199

                    for i in range(num_samples_to_viz):
                        # 1. waveform comparison on the left
                        gt_wf = waveform[i].cpu().numpy() * std + mean
                        recon_wf_i = recon_wf[i].cpu().numpy() * std + mean
                        
                        axes[i, 0].plot(gt_wf, label="GT Waveform", color="blue", alpha=0.6)
                        axes[i, 0].plot(recon_wf_i, label="Recon Waveform", color="red", linestyle="--", alpha=0.8)
                        axes[i, 0].set_title(f"Sample {i} - Temporal")
                        axes[i, 0].legend(loc="upper right", fontsize='x-small')
                        axes[i, 0].grid(True, alpha=0.2)

                        # 2. Channel Max/Min comparison on the right
                        gt_ch = channel_max_min[i].cpu().numpy()
                        recon_ch_i = recon_ch[i].cpu().numpy()
                        
                        x_axis = np.arange(len(gt_ch))
                        axes[i, 1].bar(x_axis - 0.2, gt_ch, width=0.4, label="GT Ch Max/Min", color="gray", alpha=0.5)
                        axes[i, 1].bar(x_axis + 0.2, recon_ch_i, width=0.4, label="Recon Ch Max/Min", color="orange", alpha=0.8)
                        
                        axes[i, 1].axvline(x=9.5, color='black', linestyle=':', alpha=0.5)
                        axes[i, 1].set_title(f"Sample {i} - Spatial (Max 0-9 | Min 10-19)")
                        axes[i, 1].legend(loc="upper right", fontsize='x-small')
                        axes[i, 1].grid(axis='y', alpha=0.2)

                    plt.tight_layout()
                    save_path = os.path.join(viz_dir, f"val_recon_step_{global_step}.png")
                    plt.savefig(save_path)
                    plt.close()
                    self.logger.info(f"Saved temporal-spatial comparison to {save_path}")
                
                num_batches += 1

        # t-sne visualization ---
        #if self.accelerator.is_main_process:

        #    latents_cat = torch.cat(all_latents, dim=0).numpy()

        #    max_tsne_samples = 2000
        #    if latents_cat.shape[0] > max_tsne_samples:
        #        indices = np.random.choice(latents_cat.shape[0], max_tsne_samples, replace=False)
        #        latents_to_viz = latents_cat[indices]
        #    else:
        #        latents_to_viz = latents_cat

        #    self.logger.info(f"Computing t-SNE for {latents_to_viz.shape[0]} samples...")

        #    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        #    latents_2d = tsne.fit_transform(latents_to_viz)

        #    plt.figure(figsize=(10, 8))
        #    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c='blue', alpha=0.5, s=5)
        #    plt.title(f"t-SNE of Latent Space at Step {global_step}")
        #    plt.xlabel("t-SNE 1")
        #    plt.ylabel("t-SNE 2")
        #    plt.grid(True, alpha=0.2)
            
        #    tsne_save_path = os.path.join(viz_dir, f"val_tsne_step_{global_step}.png")
        #    plt.savefig(tsne_save_path)
        #    plt.close()
        #    self.logger.info(f"Saved t-SNE visualization to {tsne_save_path}")

        avg_val_loss = total_val_loss / num_batches
        self.accelerator.log({"val_loss": avg_val_loss}, step=global_step)
        self.logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        
        self.encoder.train()
        self.decoder.train()

    def manage_checkpoints(self, global_step):
        # Pruning old checkpoints
        if self.args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= self.args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]
                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)
                    self.logger.info(f"Removed old checkpoint: {removing_checkpoint}")

        save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
        self.accelerator.save_state(save_path)
        self.logger.info(f"Saved state to {save_path}")

# ==============================================================================
# Argument Parsing & Main
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Simple training script for Autoencoder")
    
    # Model Args
    parser.add_argument("--waveform_len", type=int, default=32)
    parser.add_argument("--num_channels", type=int, default=20)
    
    # Training Args
    parser.add_argument("--output_dir", type=str, default="output/autoencoder_test")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="4090 prefers bf16")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--trainset_path", type=str, default="/Users/zhangxinyuanmacmini/Desktop/BMI_2026/VAE/SpikeSorting-main/data/Achilles_Shank2_train_0.2.pt")
    parser.add_argument("--valset_path", type=str, default="/Users/zhangxinyuanmacmini/Desktop/BMI_2026/VAE/SpikeSorting-main/data/Achilles_Shank2_val_0.2.pt")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Either 'latest' or path to checkpoint")
    
    # Tracker
    parser.add_argument("--tracker_project_name", type=str, default="autoencoder_project")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()

    trainer = AutoencoderTrainer(args, Encoder, Decoder)
    trainer.train()

    # run python -m trainer.trainer spikeinterface_env
