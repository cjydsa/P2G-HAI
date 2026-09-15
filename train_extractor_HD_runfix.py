import argparse
import logging
import time
import itertools
import os
import sys
import signal

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL,  UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torchvision.utils import save_image
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from adapter.resampler import Resampler
from VTONHD import VTONHDDataset, collate_fn
from adapter.attention_processor import CacheAttnProcessor2_0, RefCAttnProcessor2_0, RefSAttnProcessor2_0

logger = get_logger(__name__)


# ===================================
# EMA (Exponential Moving Average) Helper
# ===================================
class EMAModel:
    """
    Exponential Moving Average of model parameters.
    Only tracks trainable modules (proj, ref_unet, cross_attn_processors).
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device
        self.shadow_params = {}

        # Only track trainable parameters
        unwrapped = model.module if hasattr(model, 'module') else model
        for name, param in unwrapped.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(device if device else param.device)

    def update(self, model):
        """Update EMA parameters"""
        unwrapped = model.module if hasattr(model, 'module') else model
        for name, param in unwrapped.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data.to(self.shadow_params[name].device), alpha=1 - self.decay
                )

    def copy_to(self, model):
        """Copy EMA parameters to model"""
        unwrapped = model.module if hasattr(model, 'module') else model
        for name, param in unwrapped.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name].to(param.device))

    def store(self, model):
        """Store current model parameters"""
        self.backup_params = {}
        unwrapped = model.module if hasattr(model, 'module') else model
        for name, param in unwrapped.named_parameters():
            if name in self.shadow_params:
                self.backup_params[name] = param.data.clone()

    def restore(self, model):
        """Restore backed-up parameters"""
        if not hasattr(self, 'backup_params'):
            return
        unwrapped = model.module if hasattr(model, 'module') else model
        for name, param in unwrapped.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        del self.backup_params

    def state_dict(self):
        """Get EMA state dict"""
        return {
            'decay': self.decay,
            'shadow_params': {k: v.cpu() for k, v in self.shadow_params.items()}
        }

    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.decay = state_dict['decay']
        self.shadow_params = {k: v.to(self.device) if self.device else v
                             for k, v in state_dict['shadow_params'].items()}


def parse_args():
    parser = argparse.ArgumentParser(description="VTON-HD Garment Extractor Training Script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained CLIP image encoder.",
    )
    parser.add_argument(
        "--pretrained_text_model_path",
        type=str,
        default=None,
        help="Optional separate text encoder/tokenizer path. If None, use base model.",
    )
    parser.add_argument(
        "--pretrained_vae_model_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model.",
    )
    parser.add_argument(
        "--pretrained_adapter_model_path",
        type=str,
        default=None,
        help="Path to pretrained adapter model (IP-Adapter weights).",
    )

    # VTON-HD dataset parameters (required)
    parser.add_argument(
        "--vtonhd_root",
        type=str,
        required=True,
        help="VTON-HD dataset root directory (zalando-hd-resized).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split: train or test.",
    )
    parser.add_argument(
        "--pairs_file",
        type=str,
        default=None,
        help="Path to pairs.txt. Required only when pairing_mode=pairs_file (auto-detect if None). Optional in same_name for saving pairs.",
    )
    parser.add_argument(
        "--pairing_mode",
        type=str,
        default="pairs_file",
        choices=["pairs_file", "same_name"],
        help="Pairing strategy: 'pairs_file' reads from pairs.txt, 'same_name' matches by filename (e.g. 00000.00.jpg <-> 00000.00.jpg).",
    )
    parser.add_argument(
        "--enforce_same_name_pairs",
        action="store_true",
        help="Validate that all pairs have matching basenames (image_file == cloth_file). Raises error if not.",
    )

    # Prompt parameters
    parser.add_argument(
        "--prompts_cache",
        type=str,
        default="",
        help="Path to prompt cache file (json or jsonl). Empty means no cache.",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="cache_or_fallback",
        choices=["cache_only", "cache_or_fallback", "fallback_only"],
        help="Prompt mode: cache_only (use cache with fallback on miss), "
             "cache_or_fallback (try cache then fallback), or fallback_only (always use fallback).",
    )
    parser.add_argument(
        "--prompt_fallback",
        type=str,
        default="",
        help="Fallback prompt when cache is missing or in fallback_only mode.",
    )
    parser.add_argument(
        "--auto_prepare_prompts",
        action="store_true",
        help="Automatically build/update prompt cache before training if coverage is insufficient.",
    )

    parser.add_argument(
        "--prepare_prompts_only",
        action="store_true",
        help="Only build/update prompts_cache (when used with --auto_prepare_prompts) and then exit.",
    )
    parser.add_argument(
        "--openai_prompt_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for prompt generation (vision-capable).",
    )
    parser.add_argument(
        "--openai_prompt_use_person",
        action="store_true",
        help="Also send the person image to OpenAI (more context, higher cost). Default: cloth only.",
    )
    parser.add_argument(
        "--openai_prompt_max_retries",
        type=int,
        default=5,
        help="Max retry attempts for OpenAI prompt generation.",
    )
    parser.add_argument(
        "--openai_prompt_sleep_base",
        type=float,
        default=1.5,
        help="Base seconds for exponential backoff on OpenAI errors.",
    )
    parser.add_argument(
        "--openai_prompt_word_min",
        type=int,
        default=20,
        help="Minimum word count for generated prompt (sanity check).",
    )
    parser.add_argument(
        "--openai_prompt_word_max",
        type=int,
        default=45,
        help="Maximum word count for generated prompt (sanity check).",
    )
    parser.add_argument(
        "--openai_prompt_max_side",
        type=int,
        default=512,
        help="Max image dimension (longest side) sent to API for prompt generation. 0=no scaling. Lower values reduce cost and speed up generation.",
    )
    parser.add_argument(
        "--openai_prompt_jpeg_quality",
        type=int,
        default=85,
        help="JPEG quality (1-100) for images sent to API. Lower values reduce cost but may affect prompt quality.",
    )
    parser.add_argument(
        "--prompt_instruction_file",
        type=str,
        default=None,
        help="Optional path to a text file containing the instruction sent to OpenAI for prompt generation.",
    )
    parser.add_argument(
        "--prompt_instruction_text",
        type=str,
        default=None,
        help="Optional instruction string sent to OpenAI for prompt generation (overrides --prompt_instruction_file).",
    )
    parser.add_argument(
        "--prompt_max_pixels",
        type=int,
        default=147456,
        help="Max pixels for vision models (DashScope). Default 147456 (384x384). Use 65536 (256x256) for faster generation.",
    )
    parser.add_argument(
        "--prompt_max_tokens",
        type=int,
        default=80,
        help="Max tokens for prompt generation response. Lower values speed up generation.",
    )
    parser.add_argument(
        "--prompt_workers",
        type=int,
        default=8,
        help="Number of concurrent workers for parallel prompt generation. Higher values speed up batch processing.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--clip_penultimate",
        action="store_true",
        help="Use penultimate CLIP layer for text embedding",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--train_image_size",
        type=int,
        default=512,
        help="Training image size passed into dataset (reduce to 384/320 to save VRAM).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Force mixed precision.",
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Enable VRAM-saving mode: batch=1, no EMA, disable preview, enable checkpointing/slicing/tiling.",
    )

    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="(Unsafe) Enable xFormers memory efficient attention. OFF by default because it can break the custom attention cache and crash on Windows/GPU setups.",
    )

    parser.add_argument(
        "--noise_offset", type=float, default=0.05, help="noise_offset."
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=0, help="noise_offset."
    )

    # Classifier-Free Guidance (CFG) dropout parameters
    parser.add_argument(
        "--text_drop_prob", type=float, default=0.15,
        help="Probability of dropping text prompt (replace with empty string) for CFG training."
    )
    parser.add_argument(
        "--image_drop_prob", type=float, default=0.10,
        help="Probability of dropping image embeddings (zero out) for CFG training."
    )

    # EMA (Exponential Moving Average) parameters
    parser.add_argument(
        "--use_ema", action="store_true", default=False,
        help="Use EMA weights for better generation quality."
    )
    parser.add_argument(
        "--no_use_ema", dest="use_ema", action="store_false",
        help="Disable EMA weights."
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999,
        help="EMA decay rate (default 0.9999)."
    )
    parser.add_argument(
        "--ema_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to store EMA shadow params (cpu saves VRAM).",
    )

    parser.add_argument("--num_train_epochs", type=int, default=100000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (prevents training instability).",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    # Advanced training strategy arguments
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--milestone_steps", type=int, default=10000, help="Save milestone every N steps")
    parser.add_argument("--validation_steps", type=int, default=1000, help="Validation interval")
    parser.add_argument("--validation_samples", type=int, default=50, help="Number of samples for validation")
    parser.add_argument("--val_preview_max", type=int, default=4, help="Maximum number of validation preview images to save")
    parser.add_argument("--cuda_cleanup_steps", type=int, default=50, help="CUDA cache cleanup interval (steps)")
    parser.add_argument("--cuda_deep_cleanup_steps", type=int, default=1000, help="Deep CUDA cleanup interval (steps)")
    parser.add_argument(
        "--cuda_ipc_collect",
        dest="cuda_ipc_collect",
        action="store_true",
        help="Enable CUDA IPC cache collection during cleanup (recommended on Windows).",
    )
    parser.add_argument(
        "--no_cuda_ipc_collect",
        dest="cuda_ipc_collect",
        action="store_false",
        help="Disable CUDA IPC cache collection during cleanup.",
    )
    parser.add_argument("--cuda_mem_log_steps", type=int, default=500, help="Log CUDA memory stats every N steps (0 disables)")
    parser.set_defaults(cuda_ipc_collect=True)

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()

    # xFormers can overwrite custom attention processors and break the attention-cache path.
    # Keep it OFF by default; also force-disable in --low_vram for stability on Windows.
    if getattr(args, 'low_vram', False) and getattr(args, 'enable_xformers', False):
        logger.warning('--low_vram is incompatible with --enable_xformers in this project; forcing xformers OFF to avoid crashes.')
        args.enable_xformers = False

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def apply_low_vram_overrides(args):
    if not args.low_vram:
        return
    if args.train_batch_size != 1:
        print(f"[WARN] --low_vram forces train_batch_size=1 (was {args.train_batch_size})")
        args.train_batch_size = 1
    if args.use_ema:
        print("[WARN] --low_vram disables EMA.")
    args.use_ema = False
    if args.val_preview_max != 0:
        print(f"[WARN] --low_vram sets val_preview_max=0 (was {args.val_preview_max})")
        args.val_preview_max = 0
    if args.validation_samples > 10:
        print(f"[WARN] --low_vram caps validation_samples to 10 (was {args.validation_samples})")
        args.validation_samples = min(args.validation_samples, 10)


def enable_memory_saving(unet, ref_unet, vae, low_vram: bool, enable_xformers: bool = False):
    status = {}

    def _try(label, fn):
        try:
            fn()
            status[label] = "on"
        except Exception as e:
            status[label] = "failed"
            print(f"[WARN] {label} failed: {e}")

    if unet is not None and hasattr(unet, "enable_gradient_checkpointing"):
        _try("unet_grad_ckpt", unet.enable_gradient_checkpointing)
    else:
        status["unet_grad_ckpt"] = "unsupported"

    if ref_unet is not None and hasattr(ref_unet, "enable_gradient_checkpointing"):
        _try("ref_unet_grad_ckpt", ref_unet.enable_gradient_checkpointing)
    else:
        status["ref_unet_grad_ckpt"] = "unsupported"

    if low_vram:
        if unet is not None and hasattr(unet, "set_attention_slice"):
            _try("unet_attn_slice", lambda: unet.set_attention_slice("max"))
        else:
            status["unet_attn_slice"] = "unsupported"

        if ref_unet is not None and hasattr(ref_unet, "set_attention_slice"):
            _try("ref_unet_attn_slice", lambda: ref_unet.set_attention_slice("max"))
        else:
            status["ref_unet_attn_slice"] = "unsupported"

        if enable_xformers:
            if unet is not None and hasattr(unet, "enable_xformers_memory_efficient_attention"):
                _try("unet_xformers", unet.enable_xformers_memory_efficient_attention)
            else:
                status["unet_xformers"] = "unsupported"

            if ref_unet is not None and hasattr(ref_unet, "enable_xformers_memory_efficient_attention"):
                _try("ref_unet_xformers", ref_unet.enable_xformers_memory_efficient_attention)
            else:
                status["ref_unet_xformers"] = "unsupported"
        else:
            status["unet_xformers"] = "disabled"
            status["ref_unet_xformers"] = "disabled"

        if vae is not None and hasattr(vae, "enable_slicing"):
            _try("vae_slicing", vae.enable_slicing)
        else:
            status["vae_slicing"] = "unsupported"

        if vae is not None and hasattr(vae, "enable_tiling"):
            _try("vae_tiling", vae.enable_tiling)
        else:
            status["vae_tiling"] = "unsupported"

    return status


def cuda_memory_cleanup(tag: str, deep: bool, do_ipc: bool):
    import gc
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if do_ipc:
            # ipc_collect helps release stale IPC handles on Windows/WDDM to reduce fragmentation.
            torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.synchronize()
        if deep:
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        # Best-effort cleanup; do not fail training due to cleanup errors.
        pass


def save_model_checkpoint(output_dir, global_steps, sd_model, epoch,
                          accelerator, ema_model=None,
                          checkpoint_type="regular", loss_value=None):
    """
    Safer checkpoint on Windows/WDDM:
      - unwrap -> 收集仅可训练模块
      - 统一搬到 CPU 后再打包
      - 保存前后强制清理/同步
      - 原子写（临时文件 -> replace）
    """
    import os, gc, tempfile, logging, torch
    os.makedirs(output_dir, exist_ok=True)

    # 只有主进程保存
    if accelerator is not None and not accelerator.is_main_process:
        return None

    unwrapped = accelerator.unwrap_model(sd_model)

    # ---- 把所有 state_dict 先搬到 CPU，避免 torch.save 期间反复 GPU 同步 ----
    def to_cpu_state_dict(sd):
        cpu_sd = {}
        for k, v in sd.items():
            if torch.is_tensor(v):
                cpu_sd[k] = v.detach().to("cpu", non_blocking=True)
            else:
                cpu_sd[k] = v
        return cpu_sd

    # 组装 CPU 版 checkpoint 字典
    ckpt = {
        "epoch": int(epoch),
        "global_steps": int(global_steps),
        "image_proj": to_cpu_state_dict(unwrapped.proj.state_dict()),
        "ref_unet":   to_cpu_state_dict(unwrapped.ref_unet.state_dict()),
    }
    # cross-attn 只存可训练的
    cross = {}
    for name, proc in unwrapped.unet.attn_processors.items():
        if isinstance(proc, RefCAttnProcessor2_0):
            cross[name] = to_cpu_state_dict(proc.state_dict())
    ckpt["cross_attn_processors"] = cross

    if loss_value is not None:
        ckpt["loss"] = float(loss_value)

    # Save EMA weights if available
    if ema_model is not None:
        # Guard against wrong positional args: ema_model must be an EMA object (has state_dict), not a string like "best".
        if not hasattr(ema_model, "state_dict"):
            raise TypeError(f"save_model_checkpoint: ema_model must have state_dict(); got {type(ema_model)}. Did you pass checkpoint_type positionally? Use keyword args: ema_model=ema_model, checkpoint_type=\"best\".")
        ckpt["ema"] = {k: v.cpu() for k, v in ema_model.state_dict()['shadow_params'].items()}
        ckpt["ema_decay"] = ema_model.decay

    # 保存前尽量把 GPU 压力降到最低
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # 生成目标文件名
    if checkpoint_type == "best":
        filename = "best.pt"
    elif checkpoint_type == "last":
        filename = "last.pt"
    elif checkpoint_type == "milestone":
        filename = f"milestone_{global_steps:06d}.pt"
    else:
        filename = f"model_{global_steps:06d}.pt"
    final_path = os.path.join(output_dir, filename)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir, prefix=".tmp_ckpt_", suffix=".pt"); os.close(tmp_fd)

    try:
        # 全是 CPU 张量 -> 纯主机 IO，几乎不动 GPU
        with open(tmp_path, "wb") as f:
            torch.save(ckpt, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        logging.info(f"Saved {checkpoint_type} checkpoint to {final_path}")
    except Exception as e:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
        logging.error(f"Failed to save checkpoint: {e}")
        raise
    finally:
        # 释放内存
        del ckpt, cross
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return final_path


def run_validation(model, validation_dataloader, vae, text_encoder, image_encoder,
                  noise_scheduler, accelerator, weight_dtype, save_samples=False,
                  step=None, output_dir=None, val_preview_max=4):
    """Run validation and return metrics"""
    # Clear cache before validation to avoid memory spike
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    model.eval()
    total_loss = 0.0
    num_samples = 0
    sample_count = 0

    if accelerator.is_main_process:
        logger.info(f"VAL: Starting validation with {len(validation_dataloader)} batches...")
        logger.info(f"VAL: Will evaluate on {len(validation_dataloader.dataset)} samples")

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_dataloader):
            # More frequent progress logging (every 5 batches)
            if accelerator.is_main_process and batch_idx % 5 == 0:
                logger.info(f"VAL: Processing batch {batch_idx+1}/{len(validation_dataloader)}")

            # Same forward pass as training
            person_latents = vae.encode(
                batch["vae_person"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
            person_latents = person_latents * 0.18215

            garment_latents = vae.encode(
                batch["vae_clothes"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
            garment_latents = garment_latents * 0.18215

            noise = torch.randn_like(garment_latents)
            bsz = garment_latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=garment_latents.device)
            timesteps = timesteps.long()

            noisy_garment_latents = noise_scheduler.add_noise(garment_latents, noise, timesteps)

            # FIX P1: Validation should NOT apply dropout - use full conditioning for stable evaluation
            # This ensures validation loss accurately reflects model performance with full information
            clip_person_images = batch["clip_image"]  # Use all images without dropout

            person_image_embeds = image_encoder(
                clip_person_images.to(accelerator.device, dtype=weight_dtype),
                output_hidden_states=True
            ).hidden_states[-2]

            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

            target = noise if noise_scheduler.prediction_type == "epsilon" else \
                     noise_scheduler.get_velocity(garment_latents, noise, timesteps)

            model_pred = model(
                encoder_hidden_states,
                noisy_garment_latents,
                person_latents,  # Use original person_latents (no masking in validation)
                person_image_embeds,
                timesteps
            )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            total_loss += loss.item()
            num_samples += bsz

            # 限制保存的预览图片数量
            if save_samples and sample_count < val_preview_max and step is not None and output_dir is not None:
                try:
                    # 多步去噪：从当前的noisy_garment_latents开始，逐步去噪到干净图片
                    # IMPORTANT: Start from the CURRENT noisy latents (from validation batch), NOT from pure noise

                    # 获取单个样本的输入（取batch中第一个）
                    single_noisy_latents = noisy_garment_latents[0:1]  # 从当前噪声latents开始
                    single_timestep = timesteps[0:1]  # 当前时间步
                    single_person_latents = person_latents[0:1]
                    single_person_image_embeds = person_image_embeds[0:1]
                    single_encoder_hidden_states = encoder_hidden_states[0:1]
                    single_garment_latent = garment_latents[0:1]  # Ground truth for reference

                    # 设置DDIM采样器
                    validation_scheduler = DDIMScheduler(
                        beta_start=0.00085,
                        beta_end=0.012,
                        beta_schedule="scaled_linear",
                        num_train_timesteps=1000,
                        rescale_betas_zero_snr=True,
                        timestep_spacing="trailing",
                        prediction_type="epsilon",
                    )

                    # 从当前时间步开始去噪到0
                    current_timesteps = list(range(single_timestep.item(), -1, -1))  # 从当前t到0
                    # 限制为最多20步（平衡质量和验证速度）
                    if len(current_timesteps) > 20:
                        step_size = len(current_timesteps) // 20
                        current_timesteps = current_timesteps[::step_size]
                    if current_timesteps[-1] != 0:
                        current_timesteps.append(0)  # 确保最后一步到0

                    # 多步去噪循环
                    latents = single_noisy_latents.clone()

                    for i in range(len(current_timesteps)-1):
                        t_current = current_timesteps[i]
                        t_next = current_timesteps[i+1]

                        # 转换为tensor
                        timestep_tensor = torch.tensor([t_current], device=latents.device, dtype=torch.long)

                        # 模型预测噪声
                        noise_pred = model(
                            single_encoder_hidden_states,
                            latents,
                            single_person_latents,
                            single_person_image_embeds,
                            timestep_tensor
                        )

                        # 手动DDIM步骤：从t_current去噪到t_next
                        alpha_prod_t = validation_scheduler.alphas_cumprod[t_current]
                        alpha_prod_t_prev = validation_scheduler.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0)

                        beta_prod_t = 1 - alpha_prod_t

                        # 计算预测的原始样本 x0
                        pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

                        # 计算前一步的样本
                        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
                        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

                        latents = prev_sample

                    # Decode predicted garment to image domain
                    vae_dtype = next(vae.parameters()).dtype
                    vae_device = next(vae.parameters()).device
                    latents_to_decode = (latents / 0.18215).to(device=vae_device, dtype=vae_dtype)

                    # Decode with autocast for type consistency
                    with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=(vae_dtype in [torch.float16, torch.bfloat16])):
                        pred_image = vae.decode(latents_to_decode).sample

                    # Decode ground truth garment
                    gt_latents_to_decode = (single_garment_latent / 0.18215).to(device=vae_device, dtype=vae_dtype)
                    with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=(vae_dtype in [torch.float16, torch.bfloat16])):
                        gt_image = vae.decode(gt_latents_to_decode).sample

                    # Get person and cloth images from batch
                    person_image = batch["vae_person"][0:1]  # [1, 3, H, W]
                    cloth_image = batch["vae_clothes"][0:1]  # [1, 3, H, W]

                    # Normalize all images to [0,1]
                    pred_image = (pred_image.float() + 1.0) / 2.0
                    pred_image = torch.clamp(pred_image, 0.0, 1.0).to("cpu")

                    gt_image = (gt_image.float() + 1.0) / 2.0
                    gt_image = torch.clamp(gt_image, 0.0, 1.0).to("cpu")

                    person_image = (person_image.float() + 1.0) / 2.0
                    person_image = torch.clamp(person_image, 0.0, 1.0).to("cpu")

                    cloth_image = (cloth_image.float() + 1.0) / 2.0
                    cloth_image = torch.clamp(cloth_image, 0.0, 1.0).to("cpu")

                    # Concatenate 4 images horizontally: [person | cloth | ground_truth | prediction]
                    grid_image = torch.cat([person_image, cloth_image, gt_image, pred_image], dim=3)  # Concat along width

                    # 创建验证图片保存目录
                    val_images_dir = os.path.join(output_dir, "validation_images")
                    os.makedirs(val_images_dir, exist_ok=True)

                    # 使用cloth_file文件名来命名（去除路径和扩展名）
                    cloth_filename = batch["cloth_files"][0]  # 取第一个样本
                    cloth_basename = os.path.splitext(os.path.basename(cloth_filename))[0]

                    # 保存4张拼接图像，文件名格式: clothname_step_XXXXXX.png
                    save_path = os.path.join(val_images_dir, f"{cloth_basename}_step_{step:06d}.png")
                    save_image(grid_image, save_path)

                    if accelerator.is_main_process:
                        logger.info(f"VAL: Saved preview {sample_count+1}/{val_preview_max} -> {save_path}")

                    sample_count += 1
                except Exception as e:
                    # 如果保存失败，不影响训练继续
                    if accelerator.is_main_process:
                        logger.warning(f"Failed to save validation sample: {e}")

            # 清理 ref_unet 缓存，避免验证时显存累积（必改 1）
            if hasattr(model, "ref_unet"):
                _clear_ref_unet_cache(model.ref_unet)

    model.train()

    # 验证结束后立刻归还显存
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        logger.info(f"VAL: Completed validation, avg loss: {total_loss / len(validation_dataloader):.4f}")

    return total_loss / len(validation_dataloader) if len(validation_dataloader) > 0 else float('inf')


def count_model_params(model):
    return sum([p.numel() for p in model.parameters()]) / 1e6


def _clear_ref_unet_cache(ref_unet):
    """
    清空每一层 CacheAttnProcessor 的 cache，避免跨 step 残留图/显存
    在 Windows + WDDM 下这很容易造成显存碎片逐步累积
    """
    for proc in ref_unet.attn_processors.values():
        if hasattr(proc, "cache"):
            c = proc.cache
            # 兼容各种写法
            if hasattr(c, "clear"):
                c.clear()
            else:
                for k in list(c.keys()):
                    c[k] = None


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


class SDModel(torch.nn.Module):
    """
    SD model for REVERSE task: Extract garment from dressed person
    Architecture follows Figure 3 from IMAGDressing paper
    """

    def __init__(self, unet, ref_unet, proj, adapter_modules) -> None:
        super().__init__()

        # Denoising UNet (frozen except hybrid attention cross-attention)
        self.unet = unet

        # Garment UNet (trainable) - now extracts features from PERSON image
        self.ref_unet = ref_unet

        # Projection layer (trainable) - maps CLIP image features
        self.proj = proj

        # Hybrid Attention adapter modules (trainable cross-attention)
        self.adapter_modules = adapter_modules

    def forward(self, encoder_hidden_states, latents, ref_latents, clip_image_embeddings, timesteps):
        """
        REVERSE task forward pass:
        - ref_latents: person latents (input to Garment UNet)
        - latents: garment latents (output target for Denoising UNet)
        - clip_image_embeddings: CLIP features from person image
        """
        # Clear cache first to avoid cross-batch contamination
        for proc in self.ref_unet.attn_processors.values():
            if hasattr(proc, "cache"):
                c = proc.cache
                if hasattr(c, "clear"):
                    c.clear()
                else:
                    for k in list(c.keys()):
                        c[k] = None

        ref_timesteps = torch.zeros_like(timesteps)

        # Project CLIP features
        person_proj_embed = self.proj(clip_image_embeddings)

        # Garment UNet extracts features from PERSON image
        _ = self.ref_unet(
            ref_latents,  # person latents
            ref_timesteps,
            encoder_hidden_states=person_proj_embed,
            return_dict=False,
        )

        # Get cached features from Garment UNet
        sa_hidden_states = {}
        for name in self.ref_unet.attn_processors.keys():
            proc = self.ref_unet.attn_processors[name]
            if hasattr(proc, "cache") and "hidden_states" in proc.cache:
                sa_hidden_states[name] = proc.cache["hidden_states"]
            else:
                raise RuntimeError(f"Attention cache missing for {name}")

        # Denoising UNet generates GARMENT using extracted features
        noise_pred = self.unet(
            latents,  # noisy garment latents
            timesteps,
            encoder_hidden_states=encoder_hidden_states,  # text embeddings
            cross_attention_kwargs={
                "sa_hidden_states": sa_hidden_states,  # features from person
            }
        ).sample

        return noise_pred



# =========================
# Prompt cache preparation
# =========================
import base64
import mimetypes
import json
import random

def _build_prompt_instruction_default() -> str:
    """Default instruction: ~30-word garment prompt."""
    return (
        "Write ONE single-line English prompt (about 25–35 words) describing the GARMENT only. "
        "Include: garment type, main color(s), pattern/print, material, sleeve length, neckline/collar, "
        "fit/silhouette, closure (buttons/zipper), and notable details (logo/embroidery/lace). "
        "Do NOT describe the person, face, pose, background, or lighting. "
        "Return only the prompt text."
    )

def _load_prompt_instruction(args) -> str:
    if getattr(args, "prompt_instruction_text", None):
        return args.prompt_instruction_text.strip()
    if getattr(args, "prompt_instruction_file", None):
        with open(args.prompt_instruction_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return _build_prompt_instruction_default()

def _encode_image_to_data_url(path: str, max_side: int = 0, jpeg_quality: int = 85) -> str:
    """
    Encode image to base64 data URL with optional resizing and compression.

    Args:
        path: Path to image file
        max_side: Maximum dimension for longest side (0 = no scaling)
        jpeg_quality: JPEG quality (1-100)

    Returns:
        Data URL string (data:image/jpeg;base64,...)
    """
    try:
        from PIL import Image
        from io import BytesIO

        # Open and convert to RGB
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if needed
        if max_side > 0:
            width, height = img.size
            max_dim = max(width, height)

            if max_dim > max_side:
                # Calculate new dimensions (maintain aspect ratio)
                scale = max_side / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)

                # Resize with high-quality BICUBIC interpolation
                img = img.resize((new_width, new_height), Image.BICUBIC)

        # Encode to JPEG in memory
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        buffer.seek(0)

        # Base64 encode
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    except Exception as e:
        # Fallback: read original file as binary (no processing)
        import warnings
        warnings.warn(f"Failed to process image with PIL ({e}), using raw binary encoding")

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        return f"data:{mime};base64,{b64}"

def _load_prompt_cache_any(path: str) -> dict:
    """Load .json or .jsonl prompt cache into dict[key] -> prompt(str) or list[str]."""
    if not path or not os.path.exists(path):
        return {}
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # default: jsonl
    cache = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            k = obj.get("key")
            if not k:
                continue
            if "prompt" in obj:
                cache[k] = obj["prompt"]
            elif "prompts" in obj:
                cache[k] = obj["prompts"]
    return cache

def _atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)

def _append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def openai_generate_garment_prompt(client, model: str, cloth_path: str, person_path: str = None,
                                  use_person: bool = False, max_retries: int = 5, sleep_base: float = 1.5,
                                  word_min: int = 20, word_max: int = 45, instruction: str = "",
                                  max_side: int = 0, jpeg_quality: int = 85, max_pixels: int = 147456,
                                  max_tokens: int = 80) -> str:
    """
    Generate a short garment prompt (~30 words) using OpenAI vision.

    Args:
        client: OpenAI client instance
        model: Model name (e.g., "gpt-4o-mini", "qwen-vl-plus")
        cloth_path: Path to cloth image
        person_path: Path to person image (optional)
        use_person: Whether to send person image to API
        max_retries: Maximum retry attempts
        sleep_base: Base sleep time for exponential backoff
        word_min: Minimum word count
        word_max: Maximum word count
        instruction: Custom instruction for prompt generation
        max_side: Maximum image dimension (0=no scaling)
        jpeg_quality: JPEG quality (1-100)
        max_pixels: Max pixels for vision models (DashScope)
        max_tokens: Max tokens for response

    Returns:
        Generated prompt string
    """
    instr = instruction or _build_prompt_instruction_default()
    cloth_data_url = _encode_image_to_data_url(cloth_path, max_side=max_side, jpeg_quality=jpeg_quality)

    content = [
        {"type": "text", "text": instr},
        {"type": "image_url", "image_url": {"url": cloth_data_url}},
    ]
    if use_person and person_path is not None:
        person_data_url = _encode_image_to_data_url(person_path, max_side=max_side, jpeg_quality=jpeg_quality)
        content.append({"type": "image_url", "image_url": {"url": person_data_url}})

    # Build extra_body for DashScope/Qwen models
    extra_params = {}
    if model.startswith("qwen"):
        extra_params["extra_body"] = {
            "enable_thinking": False,
            "max_pixels": max_pixels
        }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=max_tokens,
                **extra_params
            )
            text = (resp.choices[0].message.content or "").strip()
            text = " ".join(text.replace("\n", " ").replace("\t", " ").split())
            text = text.strip().strip('"').strip("'").strip()

            words = [w for w in text.split(" ") if w]
            wc = len(words)

            # Empty prompt check
            if wc == 0:
                raise RuntimeError("Empty prompt from model.")

            # Handle over-length: truncate to word_max
            if wc > word_max:
                truncated_words = words[:word_max]
                return " ".join(truncated_words)

            # Handle under-length: pad with filler words
            if wc < word_min:
                filler_words = ["fabric", "textile", "material", "design", "style", "detail"]
                words_to_add = word_min - wc
                for i in range(words_to_add):
                    words.append(filler_words[i % len(filler_words)])
                return " ".join(words)

            return text
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (2 ** attempt) + random.random() * 0.2)

    raise RuntimeError(f"OpenAI prompt generation failed after retries. Last error: {last_err}")

def ensure_prompts_cache_complete(args, tokenizer):
    """Build/update prompt cache so that EVERY pair has a prompt (with concurrent generation)."""
    if not args.prompts_cache:
        raise ValueError("--prompts_cache is required when using --auto_prepare_prompts")

    # Check API keys: support both OpenAI and DashScope
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY (Qwen/DashScope) or OPENAI_API_KEY (OpenAI) before --auto_prepare_prompts.")

    # Get base_url: priority to explicit env vars, fallback to DashScope default if using DASHSCOPE_API_KEY
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
    if not base_url and os.getenv("DASHSCOPE_API_KEY"):
        # Default DashScope OpenAI-compatible endpoint
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        print(f"[PromptBuilder] Using default DashScope endpoint: {base_url}")

    from openai import OpenAI
    from VTONHD import VTONHDDataset, make_pair_key
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    instruction = _load_prompt_instruction(args)

    # Lightweight dataset instance just to read pairs & directories (no image loading at init)
    # NOTE: verify_files=False for prompt generation stage (files will be verified during training)
    tmp_ds = VTONHDDataset(
        root=args.vtonhd_root,
        split=args.split,
        tokenizer=tokenizer,
        pairs_file=args.pairs_file,
        prompts_cache=args.prompts_cache if (args.prompts_cache and os.path.exists(args.prompts_cache)) else None,
        prompt_mode="fallback_only",
        prompt_fallback="",
        size=args.train_image_size,
        verify_files=False,  # Skip file verification during prompt generation (faster)
        pairing_mode=args.pairing_mode,
        enforce_same_name_pairs=args.enforce_same_name_pairs,
    )

    cache_path = args.prompts_cache
    ext = os.path.splitext(cache_path)[1].lower()
    existing = _load_prompt_cache_any(cache_path)
    existing_keys = set(existing.keys())

    # Create OpenAI client with proper base_url
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"[PromptBuilder] Using API endpoint: {base_url}")
    else:
        client = OpenAI(api_key=api_key)
        print(f"[PromptBuilder] Using default OpenAI endpoint")

    total = len(tmp_ds.data)
    failures = []

    print(f"[PromptBuilder] Model: {args.openai_prompt_model}")
    print(f"[PromptBuilder] Total pairs: {total}, cached: {len(existing_keys)}, to generate: {total - len(existing_keys)}")
    print(f"[PromptBuilder] Cache format: {ext}, path: {cache_path}")
    print(f"[PromptBuilder] Image settings: max_side={args.openai_prompt_max_side}, jpeg_quality={args.openai_prompt_jpeg_quality}")
    print(f"[PromptBuilder] Concurrency: {args.prompt_workers} workers")
    print(f"[PromptBuilder] Performance: max_pixels={args.prompt_max_pixels}, max_tokens={args.prompt_max_tokens}")

    # Thread-safe write lock for jsonl append
    write_lock = threading.Lock()
    generated_count = 0

    def generate_one_prompt(idx_item):
        """Worker function to generate one prompt"""
        idx, item = idx_item
        key = make_pair_key(item["image_file"], item["cloth_file"])

        if key in existing_keys:
            return None  # Already cached

        person_path = os.path.join(tmp_ds.person_dir, item["image_file"])
        cloth_path = os.path.join(tmp_ds.cloth_dir, item["cloth_file"])

        try:
            prompt = openai_generate_garment_prompt(
                client,
                model=args.openai_prompt_model,
                cloth_path=cloth_path,
                person_path=person_path,
                use_person=args.openai_prompt_use_person,
                max_retries=args.openai_prompt_max_retries,
                sleep_base=args.openai_prompt_sleep_base,
                word_min=args.openai_prompt_word_min,
                word_max=args.openai_prompt_word_max,
                instruction=instruction,
                max_side=args.openai_prompt_max_side,
                jpeg_quality=args.openai_prompt_jpeg_quality,
                max_pixels=args.prompt_max_pixels,
                max_tokens=args.prompt_max_tokens,
            )

            # Thread-safe write
            if ext == ".jsonl":
                with write_lock:
                    _append_jsonl(cache_path, {"key": key, "prompt": prompt})

            return {"key": key, "prompt": prompt, "error": None}

        except Exception as e:
            return {"key": key, "prompt": None, "error": str(e)}

    # Collect items to generate
    items_to_generate = []
    for idx, item in enumerate(tmp_ds.data):
        key = make_pair_key(item["image_file"], item["cloth_file"])
        if key not in existing_keys:
            items_to_generate.append((idx, item))

    if len(items_to_generate) == 0:
        print(f"[PromptBuilder] ✓ All prompts already cached!")
        return

    print(f"[PromptBuilder] Starting concurrent generation of {len(items_to_generate)} prompts...")

    # Concurrent execution
    with ThreadPoolExecutor(max_workers=args.prompt_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(generate_one_prompt, item): item for item in items_to_generate}

        # Process as they complete
        for future in as_completed(future_to_item):
            result = future.result()

            if result is None:
                continue  # Already cached

            if result["error"] is None:
                # Success
                generated_count += 1
                existing[result["key"]] = result["prompt"]
                existing_keys.add(result["key"])

                # Progress logging every 50 items
                if generated_count % 50 == 0:
                    print(f"[PromptBuilder] Progress: {generated_count}/{len(items_to_generate)} generated")

                # For JSON format, save periodically to avoid data loss
                if ext == ".json" and (generated_count % 200 == 0):
                    with write_lock:
                        _atomic_write_json(cache_path, existing)
                    print(f"[PromptBuilder] Checkpoint saved at {generated_count} items")
            else:
                # Failure
                failures.append({"key": result["key"], "error": result["error"]})
                print(f"[PromptBuilder] ERROR generating prompt for {result['key']}: {result['error']}")

    # Final save for JSON format
    if ext == ".json" and generated_count > 0:
        with write_lock:
            _atomic_write_json(cache_path, existing)

    print(f"[PromptBuilder] ✓ Completed. Generated: {generated_count}, Total cached: {len(existing_keys)}, Failures: {len(failures)}")

    # Handle failures
    if failures:
        fail_path = cache_path + ".failures.json"
        try:
            with open(fail_path, "w", encoding="utf-8") as f:
                json.dump(failures, f, ensure_ascii=False, indent=2)
            print(f"[PromptBuilder] Failure log saved to: {fail_path}")
        except Exception as _e:
            print(f"[PromptBuilder] WARNING: could not write failures file: {_e}")
        raise RuntimeError(
            f"[PromptBuilder] {len(failures)} prompts failed; cache incomplete. "
            f"See {fail_path}. Fix errors and re-run --auto_prepare_prompts."
        )

def main():
    args = parse_args()
    apply_low_vram_overrides(args)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    accelerator.print("VRAM CONFIG")
    accelerator.print(
        f"  train_batch_size={args.train_batch_size}, grad_accum={args.gradient_accumulation_steps}, "
        f"mixed_precision={args.mixed_precision}, low_vram={args.low_vram}, use_ema={args.use_ema}, "
        f"ema_device={args.ema_device}, train_image_size={args.train_image_size}"
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    text_model_path = args.pretrained_text_model_path or args.pretrained_model_name_or_path
    tokenizer = CLIPTokenizer.from_pretrained(text_model_path, subfolder="tokenizer")
    # Auto-prepare prompt cache (main process only)
    if args.auto_prepare_prompts and accelerator.is_main_process:
        ensure_prompts_cache_complete(args, tokenizer)
    accelerator.wait_for_everyone()

    if args.prepare_prompts_only:
        if accelerator.is_main_process:
            print("[PromptBuilder] prepare_prompts_only enabled - exiting after cache preparation.")
        return

    text_encoder = CLIPTextModel.from_pretrained(text_model_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    # load ipa weight
    ipa_weight = torch.load(args.pretrained_adapter_model_path, map_location="cpu")
    image_proj = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=16,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )
    # 兼容不同格式的权重文件
    try:
        image_proj.load_state_dict(ipa_weight['image_proj'])
    except KeyError:
        # 有些权重直接是整个state_dict
        image_proj.load_state_dict(ipa_weight)

    # set attention processor
    attn_procs = {}
    st = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        # lora_rank = hidden_size // 2 # args.lora_rank
        if cross_attention_dim is None:
            # Self-attention: use RefSAttnProcessor2_0 (will be frozen)
            attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ref.weight": st[layer_name + ".to_k.weight"],
                "to_v_ref.weight": st[layer_name + ".to_v.weight"],
            }
            attn_procs[name].load_state_dict(weights)
        else:
            # Cross-attention: use RefCAttnProcessor2_0 (trainable for person features)
            attn_procs[name] = RefCAttnProcessor2_0(name, hidden_size=hidden_size,
                                                     cross_attention_dim=cross_attention_dim)
    unet.set_attn_processor(attn_procs)

    # Freeze self-attention processors (RefSAttnProcessor2_0)
    # Only train cross-attention processors (RefCAttnProcessor2_0)
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, RefSAttnProcessor2_0):
            # Freeze self-attention ref params
            proc.requires_grad_(False)
            proc.eval()  # Set to eval mode to freeze any normalization/dropout

    # Do NOT call adapter_modules.requires_grad_(True) - it would unfreeze self-attention!
    # We'll collect only trainable cross-attention processors into optimizer
    del st

    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    ref_unet.set_attn_processor(
        {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_proj.requires_grad_(True)
    ref_unet.requires_grad_(True)
    # adapter_modules are handled individually - self-attn frozen, cross-attn trainable

    # Collect trainable cross-attention processors only
    trainable_cross_attn_params = []
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, RefCAttnProcessor2_0):
            # Explicitly enable gradients for cross-attention parameters
            proc.requires_grad_(True)
            trainable_cross_attn_params.extend(proc.parameters())

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())  # for saving only
    sd_model = SDModel(unet, ref_unet, image_proj, adapter_modules)

    mem_status = enable_memory_saving(unet=unet, ref_unet=ref_unet, vae=vae, low_vram=args.low_vram, enable_xformers=args.enable_xformers)
    if args.low_vram:
        accelerator.print("LOW_VRAM FEATURES")
        accelerator.print(
            f"  grad_ckpt: unet={mem_status.get('unet_grad_ckpt')}, ref_unet={mem_status.get('ref_unet_grad_ckpt')}"
        )
        accelerator.print(
            f"  attn_slice: unet={mem_status.get('unet_attn_slice')}, ref_unet={mem_status.get('ref_unet_attn_slice')}"
        )
        accelerator.print(
            f"  xformers: unet={mem_status.get('unet_xformers')}, ref_unet={mem_status.get('ref_unet_xformers')}"
        )
        accelerator.print(
            f"  vae: slicing={mem_status.get('vae_slicing')}, tiling={mem_status.get('vae_tiling')}"
        )

    # Collect parameters for optimization: proj + ref_unet + cross-attention only
    params_to_opt = itertools.chain(
        sd_model.proj.parameters(),
        sd_model.ref_unet.parameters(),
        trainable_cross_attn_params
    )

    # Count trainable cross-attention parameters
    cross_attn_param_count = sum([p.numel() for p in trainable_cross_attn_params]) / 1e6

    accelerator.print("Trainable parameters: proj:{:.2f}M, ref_unet:{:.2f}M, cross_attn:{:.2f}M".format(
        count_model_params(sd_model.proj), count_model_params(sd_model.ref_unet),
        cross_attn_param_count))
    # accelerator.print("Trainable parameters: {:.2f}M".format(len(params_to_opt)))
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    if (
            accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        optimizer = torch.optim.AdamW(
            params_to_opt,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
            foreach=False,   # Disable foreach to reduce peak memory usage
            fused=False      # Ensure not using fused variant (unstable on Windows)
        )
    else:
        # use deepspeed config
        optimizer = DummyOptim(
            params_to_opt,
            lr=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"],
            weight_decay=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["weight_decay"]
        )

    # TODO (patil-suraj): load scheduler using args
    noise_scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing", prediction_type="epsilon",
    )

    # Load VTON-HD dataset
    dataset = VTONHDDataset(
        root=args.vtonhd_root,
        split=args.split,
        tokenizer=tokenizer,
        pairs_file=args.pairs_file,
        prompts_cache=args.prompts_cache,
        prompt_mode=args.prompt_mode,
        prompt_fallback=args.prompt_fallback,
        size=args.train_image_size,
        verify_files=True,  # STRICT: Verify all files exist before training
        pairing_mode=args.pairing_mode,
        enforce_same_name_pairs=args.enforce_same_name_pairs,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.train_batch_size,
        num_workers=0,  # Use main process only for minimal memory footprint
        pin_memory=False,  # Disable pinned memory to save RAM
        persistent_workers=False
    )

    if accelerator.state.deepspeed_plugin is not None:
        # here we use agrs.gradient_accumulation_steps
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        # use deepspeed scheduler
        lr_scheduler = DummyScheduler(
            optimizer,
            warmup_num_steps=accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "warmup_num_steps"]
        )

    if (
            accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto"
    ):
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    sd_model, optimizer, lr_scheduler = accelerator.prepare(sd_model, optimizer, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin is None:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    else:
        if accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]:
            weight_dtype = torch.float16
        elif accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]:
            weight_dtype = torch.bfloat16
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # 设置 CUDNN benchmark 为 False，避免卷积算法切换带来的额外显存峰值
    torch.backends.cudnn.benchmark = False

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  CFG Training: text_drop_prob={args.text_drop_prob}, image_drop_prob={args.image_drop_prob}")
    logger.info(
        f"  EMA: enabled={args.use_ema}, decay={args.ema_decay if args.use_ema else 'N/A'}, "
        f"device={args.ema_device if args.use_ema else 'N/A'}"
    )
    logger.info(f"  Prediction Type: {noise_scheduler.config.prediction_type}")

    # Initialize EMA model if enabled
    ema_model = None
    if args.use_ema:
        ema_device = torch.device("cpu") if args.ema_device == "cpu" else accelerator.device
        ema_model = EMAModel(sd_model, decay=args.ema_decay, device=ema_device)
        logger.info(f"✓ EMA initialized with decay={args.ema_decay} on {ema_device}")

    global_steps = 0
    starting_epoch = 0
    resumed_step = -1  # Track the step we resumed from to avoid re-saving

    best_val_loss = float('inf')

    # Create validation dataset (same configuration as training)
    val_dataset = VTONHDDataset(
        root=args.vtonhd_root,
        split=args.split,
        tokenizer=tokenizer,
        pairs_file=args.pairs_file,
        prompts_cache=args.prompts_cache,
        prompt_mode=args.prompt_mode,
        prompt_fallback=args.prompt_fallback,
        size=args.train_image_size,
        verify_files=True,  # STRICT: Verify all files exist before validation
        pairing_mode=args.pairing_mode,
        enforce_same_name_pairs=args.enforce_same_name_pairs,
    )

    # Use a subset for validation
    # FIX: Use fixed seed for validation set to ensure consistency across training runs
    # This allows best_val_loss comparison to be meaningful when resuming from checkpoint
    val_size = min(args.validation_samples, len(val_dataset))
    val_generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducible validation set
    val_indices = torch.randperm(len(val_dataset), generator=val_generator)[:val_size]
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    val_dataloader = torch.utils.data.DataLoader(
        val_subset, collate_fn=collate_fn, batch_size=1, num_workers=0, shuffle=False,  # Use main process for validation
        pin_memory=True, persistent_workers=False
    )

    accelerator.print(f"✓ Validation dataset: {val_size} samples")

    # 输出选中的验证图片路径（用于调试和监控）
    if accelerator.is_main_process:
        accelerator.print("=" * 60)
        accelerator.print("VALIDATION SAMPLES DETAILS")
        accelerator.print("=" * 60)
        for i, idx in enumerate(val_indices[:min(10, len(val_indices))]):  # 最多显示前10个
            item = val_dataset.data[idx]
            accelerator.print(f"  Sample {i+1:2d} (idx={idx:4d}): Person={item['image_file']}")
            accelerator.print(f"             {' '*17} Cloth={item['cloth_file']}")
        if len(val_indices) > 10:
            accelerator.print(f"  ... and {len(val_indices)-10} more samples")
        accelerator.print("=" * 60)

    # Advanced training strategy initialization
    accelerator.print("=" * 60)
    accelerator.print("TRAINING STRATEGY")
    accelerator.print("=" * 60)
    accelerator.print(f"Save Checkpoints: Every {args.save_steps} steps")
    accelerator.print(f"Milestones: Every {args.milestone_steps} steps")
    accelerator.print(f"Validation: Every {args.validation_steps} steps")
    accelerator.print("=" * 60)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")

        # 添加 unwrap 正确访问 accelerator 包装的模型
        unwrapped = accelerator.unwrap_model(sd_model)
        unwrapped.ref_unet.load_state_dict(ckpt["ref_unet"], strict=True)
        unwrapped.proj.load_state_dict(ckpt["image_proj"], strict=True)
        for name, proc in unwrapped.unet.attn_processors.items():
            if isinstance(proc, RefCAttnProcessor2_0) and name in ckpt["cross_attn_processors"]:
                proc.load_state_dict(ckpt["cross_attn_processors"][name], strict=True)

        # Optional: restore epoch and global_steps
        starting_epoch = int(ckpt.get("epoch", 0))
        global_steps = int(ckpt.get("global_steps", 0))
        resumed_step = global_steps  # Remember the resumed step
        if "loss" in ckpt:
            best_val_loss = ckpt["loss"]
        accelerator.print(f"Resumed from {args.resume_from_checkpoint} at step {global_steps}")

    def should_save_checkpoint(step):
        """Check if should save checkpoint at current step"""
        return step % args.save_steps == 0

    def should_save_milestone(step):
        """Check if should save milestone at current step"""
        return step % args.milestone_steps == 0 and step > 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        train_sampler.set_epoch(epoch)  # 确保每个epoch的采样顺序不同
        sd_model.train()
        train_loss = 0.0
        step = 0
        begin = time.perf_counter()
        for batch in train_dataloader:
            load_data_time = time.perf_counter() - begin

            # REVERSE TASK: Extract garment from dressed person
            # Convert images to latent space
            with torch.no_grad():
                # Person latents (input to Garment UNet)
                person_latents = vae.encode(
                    batch["vae_person"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                person_latents = person_latents * 0.18215

                # Garment latents (target output)
                garment_latents = vae.encode(
                    batch["vae_clothes"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                garment_latents = garment_latents * 0.18215

            # Sample noise for garment latents (target)
            noise = torch.randn_like(garment_latents)
            if args.noise_offset > 0:
                noise += args.noise_offset * torch.randn(
                    (garment_latents.shape[0], garment_latents.shape[1], 1, 1),
                    device=garment_latents.device,
                )
            bsz = garment_latents.shape[0]

            # Sample random timesteps
            # FIX: Avoid timestep 999 which causes numerical instability with noise offset
            # Use range [0, 998] instead of [0, 999] to prevent extreme noise ratios
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps - 1, (bsz,), device=garment_latents.device)
            timesteps = timesteps.long()

            # Add noise to garment latents
            noisy_garment_latents = noise_scheduler.add_noise(garment_latents, noise, timesteps)

            # ===============================================
            # CFG Dropout: Apply text and image dropout
            # ===============================================
            # Apply text dropout (CFG for text prompts)
            if args.text_drop_prob > 0 and torch.rand(1).item() < args.text_drop_prob:
                # Use empty text embedding for CFG
                empty_input_ids = tokenizer(
                    [""] * bsz,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(empty_input_ids)[0]
            else:
                # Normal text embedding
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

            # Apply image dropout (CFG for image embeddings)
            apply_image_dropout = (args.image_drop_prob > 0 and torch.rand(1).item() < args.image_drop_prob)

            if apply_image_dropout:
                # Zero out image embeddings for CFG
                with torch.no_grad():
                    # Create zero embeddings with correct shape
                    person_image_embeds = torch.zeros(
                        bsz,
                        257,  # CLIP image encoder output size
                        image_encoder.config.hidden_size,
                        device=accelerator.device,
                        dtype=weight_dtype
                    )
            else:
                # Normal image embedding
                with torch.no_grad():
                    person_image_embeds = image_encoder(
                        batch["clip_image"].to(accelerator.device, dtype=weight_dtype),
                        output_hidden_states=True
                    ).hidden_states[-2]

            # Compute target
            if noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(garment_latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")


            try:
                # Forward pass: person latents -> garment latents
                model_pred = sd_model(
                    encoder_hidden_states,
                    noisy_garment_latents,  # target: garment
                    person_latents,  # reference: person (always use full person latents for ref_unet feature extraction)
                    person_image_embeds,  # CLIP from person (dropout applied above)
                    timesteps
                )

                # Compute loss
                if args.snr_gamma == 0:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = (loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights).mean()

                # NaN检测：在backward前检查loss
                if torch.isnan(loss) or torch.isinf(loss):
                    if accelerator.is_main_process:
                        logger.error(f"❌ NaN/Inf detected at step {global_steps}!")
                        logger.error(f"   Loss value: {loss.item()}")
                        logger.error(f"   Timestep: {timesteps.tolist()}")
                        # 紧急保存checkpoint
                        #logger.error("   Saving emergency checkpoint...")
                        #save_model_checkpoint(args.output_dir, global_steps, sd_model, epoch,
                                            #accelerator, ema_model, "emergency_nan")
                    # 跳过这个batch，继续训练
                    logger.warning(f"   Skipping batch {global_steps} due to NaN/Inf")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    global_steps += 1
                    step += 1
                    begin = time.perf_counter()
                    continue

                # Gather losses
                avg_loss = accelerator.gather(loss.detach()).mean()
                train_loss += avg_loss.item()

                # Backpropagate
                accelerator.backward(loss)

                # Add gradient clipping to prevent training explosion
                do_opt_step = ((step + 1) % args.gradient_accumulation_steps == 0)
                if do_opt_step and args.max_grad_norm and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, sd_model.parameters()),
                        max_norm=float(args.max_grad_norm),
                    )

                # Clear ref_unet cache after each mini-step to reduce peak memory usage
                _clear_ref_unet_cache(sd_model.ref_unet)

                # Log training progress
                if accelerator.is_main_process:
                    logging.info(
                        "Epoch {}, step {}, loss: {:.4f}, lr: {:.6f}, time: {:.2f}s, data_time: {:.2f}s".format(
                            epoch, global_steps, loss.detach().item(), optimizer.param_groups[0]["lr"],
                            time.perf_counter() - begin, load_data_time)
                    )

                # Gradient check (first step only)
                if global_steps == 0 and accelerator.is_main_process:
                    logger.info("=" * 60)
                    logger.info("GRADIENT CHECK (First Step)")
                    logger.info("=" * 60)

                    # Check ref_unet gradients (should have gradients)
                    ref_unet_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                            for p in sd_model.ref_unet.parameters() if p.requires_grad)
                    logger.info(f"✓ ref_unet has gradients: {ref_unet_has_grad}")

                    # Check image_proj gradients (should have gradients)
                    proj_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                       for p in sd_model.proj.parameters() if p.requires_grad)
                    logger.info(f"✓ image_proj has gradients: {proj_has_grad}")

                    # Check RefCAttnProcessor2_0 gradients (should have gradients)
                    cross_attn_has_grad = False
                    for name, proc in sd_model.unet.attn_processors.items():
                        if isinstance(proc, RefCAttnProcessor2_0):
                            proc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                           for p in proc.parameters() if p.requires_grad)
                            cross_attn_has_grad = cross_attn_has_grad or proc_grad
                    logger.info(f"✓ RefCAttnProcessor2_0 (cross-attn) has gradients: {cross_attn_has_grad}")

                    # Check RefSAttnProcessor2_0 gradients (should NOT have gradients - frozen)
                    self_attn_has_grad = False
                    for name, proc in sd_model.unet.attn_processors.items():
                        if isinstance(proc, RefSAttnProcessor2_0):
                            proc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                           for p in proc.parameters())
                            self_attn_has_grad = self_attn_has_grad or proc_grad
                    logger.info(f"✓ RefSAttnProcessor2_0 (self-attn) has gradients: {self_attn_has_grad} (should be False)")

                    # Check frozen modules
                    unet_has_grad = any(p.grad is not None for p in sd_model.unet.parameters()
                                       if p.requires_grad and not any(p is proc_p
                                       for proc in sd_model.unet.attn_processors.values()
                                       for proc_p in proc.parameters()))
                    logger.info(f"✓ Main UNet (non-adapter) has gradients: {unet_has_grad} (should be False)")

                    logger.info("=" * 60)

                    if not ref_unet_has_grad or not proj_has_grad or not cross_attn_has_grad:
                        logger.warning("⚠ WARNING: Expected gradients missing! Check model setup.")
                    if self_attn_has_grad:
                        logger.warning("⚠ WARNING: Self-attention has gradients but should be frozen!")

                # DEBUG: Monitor person_latents and model predictions every 500 steps
                if global_steps % 500 == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        logger.info("=" * 60)
                        logger.info(f"DEBUG INFO (Step {global_steps})")
                        logger.info("=" * 60)
                        logger.info(f"person_latents stats: min={person_latents.min():.4f}, max={person_latents.max():.4f}, mean={person_latents.mean():.4f}")
                        logger.info(f"model_pred stats: min={model_pred.min():.4f}, max={model_pred.max():.4f}, mean={model_pred.mean():.4f}")
                        logger.info(f"target stats: min={target.min():.4f}, max={target.max():.4f}, mean={target.mean():.4f}")
                        logger.info(f"CFG dropouts - text: {args.text_drop_prob*100:.1f}%, image: {args.image_drop_prob*100:.1f}%")
                        logger.info("=" * 60)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # Free gradient memory immediately

                    # Update EMA after optimizer step
                    if args.use_ema and ema_model is not None:
                        ema_model.update(sd_model)

                    # Clear cache after optimizer step to reduce fragmentation
                    cuda_memory_cleanup(tag="post_step", deep=False, do_ipc=args.cuda_ipc_collect)

                    # Proactively clean up on interval to prevent fragmentation
                    if args.cuda_cleanup_steps > 0 and global_steps % args.cuda_cleanup_steps == 0:
                        for name in [
                            "person_latents",
                            "garment_latents",
                            "noisy_garment_latents",
                            "noise",
                            "model_pred",
                            "loss",
                        ]:
                            if name in locals():
                                del locals()[name]
                        cuda_memory_cleanup(tag="periodic", deep=False, do_ipc=args.cuda_ipc_collect)

                    if accelerator.sync_gradients:
                        accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps}, step=global_steps)
                        train_loss = 0.0
            except RuntimeError as e:
                err_msg = str(e)
                if "out of memory" in err_msg.lower() or "cuda error" in err_msg.lower():
                    if accelerator.is_main_process:
                        logger.error(f"[CUDA] RuntimeError at step {global_steps}: {err_msg}")
                    cuda_memory_cleanup(tag="exception", deep=True, do_ipc=args.cuda_ipc_collect)
                    optimizer.zero_grad(set_to_none=True)
                    global_steps += 1
                    step += 1
                    begin = time.perf_counter()
                    continue
                raise

            if accelerator.is_main_process and args.cuda_mem_log_steps > 0 and torch.cuda.is_available():
                if global_steps % args.cuda_mem_log_steps == 0:
                    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                    rsv = torch.cuda.memory_reserved() / (1024 ** 3)
                    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    logger.info(f"[CUDA] step {global_steps}: alloc={alloc:.2f} GB, reserved={rsv:.2f} GB, peak={peak:.2f} GB")

            # Deep memory cleanup on interval to prevent Windows WDDM fragmentation
            if args.cuda_deep_cleanup_steps > 0 and global_steps % args.cuda_deep_cleanup_steps == 0 and global_steps > 0:
                if accelerator.is_main_process:
                    logger.info(f"Deep memory cleanup at step {global_steps}")
                cuda_memory_cleanup(tag="deep", deep=True, do_ipc=args.cuda_ipc_collect)
            # Checkpoint strategy
            if accelerator.is_main_process and global_steps > 0:
                # Regular checkpoint saving
                # Skip saving if this is exactly the resumed step (to avoid overwriting the checkpoint we just loaded)
                if should_save_checkpoint(global_steps) and global_steps != resumed_step:
                    save_model_checkpoint(args.output_dir, global_steps, sd_model, epoch, accelerator, ema_model=ema_model, checkpoint_type="regular")

                # Milestone checkpoint saving
                if should_save_milestone(global_steps) and global_steps != resumed_step:
                    save_model_checkpoint(args.output_dir, global_steps, sd_model, epoch, accelerator, ema_model=ema_model, checkpoint_type="milestone")

                # Validation and best model saving
                if global_steps % args.validation_steps == 0:
                    logger.info("=" * 60)
                    logger.info("RUNNING VALIDATION")
                    logger.info("=" * 60)

                    # Force memory cleanup before validation to prevent OOM
                    cuda_memory_cleanup(tag="pre_validation", deep=True, do_ipc=args.cuda_ipc_collect)

                    # Run validation with current model
                    logger.info("VAL: Evaluating regular model...")
                    # Save preview images every validation (controlled by val_preview_max)
                    save_preview = (args.val_preview_max > 0)
                    val_loss = run_validation(
                        sd_model, val_dataloader, vae, text_encoder, image_encoder,
                        noise_scheduler, accelerator, weight_dtype,
                        save_samples=save_preview, step=global_steps, output_dir=args.output_dir,
                        val_preview_max=args.val_preview_max if save_preview else 0
                    )

                    # Force memory cleanup after validation
                    cuda_memory_cleanup(tag="post_validation", deep=True, do_ipc=args.cuda_ipc_collect)

                    logger.info(f"✓ Validation Loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_model_checkpoint(args.output_dir, global_steps, sd_model, epoch, accelerator, ema_model=ema_model, checkpoint_type="best", loss_value=val_loss)
                        logger.info(f"🎯 NEW BEST MODEL! Val Loss: {val_loss:.4f}")

                    # Log metrics
                    accelerator.log({
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss
                    }, step=global_steps)

                    logger.info("=" * 60)

            global_steps += 1
            step += 1

            # Stop training
            if global_steps >= args.max_train_steps:
                break
            begin = time.perf_counter()

        # Check if we should stop training after breaking from batch loop
        if global_steps >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    # Save last model
    if accelerator.is_main_process:
        save_model_checkpoint(args.output_dir, global_steps, sd_model, epoch, accelerator, ema_model=ema_model, checkpoint_type="last")

        # Final summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total steps: {global_steps}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info("=" * 60)

    accelerator.end_training()


if __name__ == "__main__":
    main()