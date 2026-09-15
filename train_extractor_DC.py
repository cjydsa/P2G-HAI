import argparse
import logging
import time
import itertools
import os
import sys
import signal
import gc
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path

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
from Dresscode import DressCodeDataset, collate_fn, make_pair_key, normalize_prompt_cache_key
from adapter.attention_processor import (
    CacheAttnProcessor2_0,
    RefCAttnProcessor2_0,
    RefSAttnProcessor2_0,
    SAttnProcessor2_0,
    CAttnProcessor2_0,
)

logger = get_logger(__name__)
PROMPT_KEY_FORMAT = "dresscode_pair_key_v1"


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


DEFAULT_FALLBACK_PROMPT = (
    "a studio-style product shot of the garment only: specify garment type, main colors, fabric material, "
    "neckline or collar style, sleeve length, closure type, fit or silhouette, and notable details like buttons, "
    "embroidery, or trim"
)


def parse_args():
    parser = argparse.ArgumentParser(description="DressCode Garment Extractor Training Script.")
    default_cuda_cleanup_steps = 50
    default_cuda_deep_cleanup_steps = 1000
    default_cuda_ipc_collect = True
    default_dataloader_pin_memory = False
    default_dataloader_persistent_workers = False
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

    # DressCode dataset parameters (required)
    parser.add_argument(
        "--dresscode_root",
        type=str,
        required=True,
        help="DressCode dataset root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split: train or test.",
    )
    parser.add_argument(
        "--dresscode_category",
        type=str,
        default="all",
        choices=["all", "upper_body", "lower_body", "dresses"],
        help="DressCode category to use.",
    )
    parser.add_argument(
        "--dresscode_test_order",
        type=str,
        default="paired",
        choices=["paired", "unpaired"],
        help="DressCode test order (only used when split=test).",
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
        help="Prompt mode: cache_only (strict cache-only, error on miss), "
             "cache_or_fallback (use cache, fallback on miss), or fallback_only (always use fallback).",
    )
    parser.add_argument(
        "--prompt_fallback",
        type=str,
        default=DEFAULT_FALLBACK_PROMPT,
        help="Fallback prompt when cache is missing or in fallback_only mode.",
    )
    parser.add_argument(
        "--strict_prompts_cache",
        action="store_true",
        help="If set, raise on cache miss even in cache_or_fallback mode.",
    )
    parser.add_argument(
        "--min_cache_coverage",
        type=float,
        default=None,
        help="Minimum prompt cache coverage (percent) required before training. "
             "Default: 100 for cache_only, 0 otherwise.",
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
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. On Windows, 0 is most stable (try 2/4 for speed).",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        dest="dataloader_pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory (can increase host RAM usage).",
    )
    parser.add_argument(
        "--no_dataloader_pin_memory",
        dest="dataloader_pin_memory",
        action="store_false",
        help="Disable DataLoader pin_memory.",
    )
    parser.add_argument(
        "--dataloader_persistent_workers",
        dest="dataloader_persistent_workers",
        action="store_true",
        help="Enable DataLoader persistent_workers (requires num_workers > 0).",
    )
    parser.add_argument(
        "--no_dataloader_persistent_workers",
        dest="dataloader_persistent_workers",
        action="store_false",
        help="Disable DataLoader persistent_workers.",
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
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers memory efficient attention if available (disabled by default).",
    )
    default_sdpa_mode = "math" if os.name == "nt" else "auto"
    parser.add_argument(
        "--sdpa_mode",
        type=str,
        default=default_sdpa_mode,
        choices=["auto", "math", "flash"],
        help=(
            "SDPA backend preference: auto=enable flash/mem_efficient if available, "
            "math=disable flash/mem_efficient, flash=prefer flash. "
            "Windows defaults to math for stability."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for UNet/ref_UNet.",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
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

    # IMAGDressing ablation modes (A0/A1/A2)
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="A2",
        choices=["A0", "A1", "A2"],
        help="IMAGDressing ablation: A0=Base (no IEB/HA), A1=Base+IEB (direct IEB injection), A2=Full (IEB+HA).",
    )
    parser.add_argument(
        "--ablation_s",
        type=str,
        default="",
        choices=["S0", "S1", "S2"],
        help="Work-1 structure ablation preset: S0->A0, S1->A1, S2->A2.",
    )
    parser.add_argument(
        "--ablation_p",
        type=str,
        default="",
        choices=["P1", "P4"],
        help="Work-1 prompt preset: P1=fallback_only, P4=cache_only (strict).",
    )
    parser.add_argument(
        "--ablation_spec",
        type=str,
        default="",
        help="Optional ablation spec label (e.g., S2_P4). Overrides ablation_s/ablation_p.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Run a single validation pass and exit (requires --checkpoint or --resume_from_checkpoint).",
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path used for eval-only runs (alias of --resume_from_checkpoint).",
    )

    # Advanced training strategy arguments
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--milestone_steps", type=int, default=10000, help="Save milestone every N steps")
    parser.add_argument("--validation_steps", type=int, default=1000, help="Validation interval")
    parser.add_argument("--validation_samples", type=int, default=50, help="Number of samples for validation")
    parser.add_argument("--val_preview_max", type=int, default=4, help="Maximum number of validation preview images to save")
    parser.add_argument(
        "--validation_subset_json",
        type=str,
        default="",
        help="Optional JSON file listing validation pairs (category/image_file/cloth_file). Overrides random sampling when set.",
    )
    parser.add_argument(
        "--test_set_json",
        type=str,
        default="",
        help="Alias of --validation_subset_json for eval-only fixed test subset.",
    )
    parser.add_argument(
        "--save_val_pairs",
        action="store_true",
        default=False,
        help="Eval-only: save per-pair visual_metrics/real and visual_metrics/fake images.",
    )
    parser.add_argument(
        "--eval_denoise_steps",
        type=int,
        default=12,
        help="Eval-only: maximum denoise steps per saved sample (lower values reduce VRAM usage).",
    )
    parser.add_argument(
        "--eval_cleanup_every",
        type=int,
        default=8,
        help="Eval-only: run deep CUDA cleanup every N saved samples (0 disables).",
    )
    parser.add_argument(
        "--eval_oom_fallback_steps",
        type=int,
        default=6,
        help="Eval-only: fallback denoise steps on CUDA OOM during pair image saving (0 disables retry).",
    )
    parser.add_argument(
        "--eval_skip_existing_pairs",
        dest="eval_skip_existing_pairs",
        action="store_true",
        help="Eval-only: skip pair generation when both output images already exist (resume-friendly).",
    )
    parser.add_argument(
        "--no_eval_skip_existing_pairs",
        dest="eval_skip_existing_pairs",
        action="store_false",
        help="Disable skipping of existing eval pair outputs.",
    )
    parser.add_argument(
        "--compute_visual_metrics",
        action="store_true",
        default=False,
        help="Eval-only: compute PSNR/SSIM/LPIPS/FID/KID.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="eval_metrics.json",
        help="Eval-only: output metrics JSON filename under output_dir.",
    )
    parser.add_argument(
        "--fid_backend",
        type=str,
        default="auto",
        choices=["auto", "standard", "cleanfid"],
        help="FID backend preference. Eval-only uses standard (calculate_fid_kid_standard.py); Windows always forces standard.",
    )
    parser.add_argument(
        "--kid_backend",
        type=str,
        default="auto",
        choices=["auto", "standard", "cleanfid"],
        help="KID backend preference. Eval-only uses standard (calculate_fid_kid_standard.py); Windows always forces standard.",
    )
    parser.add_argument(
        "--strict_visual_metrics",
        action="store_true",
        default=False,
        help="Fail eval-only if FID/KID cannot be computed.",
    )
    parser.add_argument(
        "--cuda_cleanup_steps",
        type=int,
        default=default_cuda_cleanup_steps,
        help="CUDA cache cleanup interval (steps)",
    )
    parser.add_argument(
        "--cuda_deep_cleanup_steps",
        type=int,
        default=default_cuda_deep_cleanup_steps,
        help="Deep CUDA cleanup interval (steps)",
    )
    parser.add_argument(
        "--cuda_ipc_collect",
        dest="cuda_ipc_collect",
        action="store_true",
        help="Enable CUDA IPC cache collection during cleanup.",
    )
    parser.add_argument(
        "--no_cuda_ipc_collect",
        dest="cuda_ipc_collect",
        action="store_false",
        help="Disable CUDA IPC cache collection during cleanup.",
    )
    parser.add_argument("--cuda_mem_log_steps", type=int, default=500, help="Log CUDA memory stats every N steps (0 disables)")
    parser.add_argument(
        "--mem_debug",
        action="store_true",
        help="Enable per-step memory debug logging and gc.collect(); log frequency uses --cuda_mem_log_steps (or 50 if 0).",
    )
    parser.add_argument(
        "--vram_soft_cap_gb",
        type=float,
        default=0.0,
        help="Soft cap GPU memory per process in GB (0 disables). Exits gracefully on OOM when set.",
    )
    parser.set_defaults(cuda_ipc_collect=default_cuda_ipc_collect)
    parser.set_defaults(gradient_checkpointing=True)
    parser.set_defaults(dataloader_pin_memory=default_dataloader_pin_memory)
    parser.set_defaults(dataloader_persistent_workers=default_dataloader_persistent_workers)
    parser.set_defaults(eval_skip_existing_pairs=True)

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.min_cache_coverage is None:
        args.min_cache_coverage = 100.0 if args.prompt_mode == "cache_only" else 0.0
    if args.min_cache_coverage < 0 or args.min_cache_coverage > 100:
        raise ValueError("--min_cache_coverage must be between 0 and 100")
    args.eval_denoise_steps = max(1, int(args.eval_denoise_steps))
    args.eval_cleanup_every = max(0, int(args.eval_cleanup_every))
    args.eval_oom_fallback_steps = max(0, int(args.eval_oom_fallback_steps))

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
    if args.eval_denoise_steps > 12:
        print(f"[WARN] --low_vram caps eval_denoise_steps to 12 (was {args.eval_denoise_steps})")
        args.eval_denoise_steps = 12


def _resolve_effective_ablation_mode(args):
    if not args.ablation_s:
        return args.ablation_mode
    mapping = {"S0": "A0", "S1": "A1", "S2": "A2"}
    return mapping[args.ablation_s]


def resolve_ablation(args):
    spec = (args.ablation_spec or "").strip()
    if spec:
        parts = [p.strip().upper() for p in spec.replace("-", "_").split("_") if p.strip()]
        for part in parts:
            if part in ("S0", "S1", "S2"):
                args.ablation_s = part
            elif part in ("P1", "P4"):
                args.ablation_p = part

    if args.ablation_s:
        mapping = {"S0": "A0", "S1": "A1", "S2": "A2"}
        args.ablation_mode = mapping.get(args.ablation_s, args.ablation_mode)

    if args.ablation_p:
        if args.ablation_p == "P1":
            args.prompt_mode = "fallback_only"
            args.prompt_fallback = DEFAULT_FALLBACK_PROMPT
            args.strict_prompts_cache = False
            args.min_cache_coverage = 0.0
        elif args.ablation_p == "P4":
            args.prompt_mode = "cache_only"
            args.strict_prompts_cache = True
            args.min_cache_coverage = 100.0
            args.prompt_fallback = ""
            if not args.prompts_cache or not os.path.exists(args.prompts_cache):
                raise RuntimeError("--ablation_p P4 requires --prompts_cache to be set to an existing file.")

    if args.min_cache_coverage is None:
        args.min_cache_coverage = 100.0 if args.prompt_mode == "cache_only" else 0.0
    if args.min_cache_coverage < 0 or args.min_cache_coverage > 100:
        raise ValueError("--min_cache_coverage must be between 0 and 100")

    print(
        f"[AblationResolved] ablation_mode={args.ablation_mode}, "
        f"prompt_mode={args.prompt_mode}, min_cache_coverage={args.min_cache_coverage}"
    )


def _build_ablation_spec(args, effective_ablation_mode):
    if args.ablation_spec:
        return args.ablation_spec
    parts = []
    if args.ablation_s:
        parts.append(args.ablation_s)
    else:
        parts.append(effective_ablation_mode)
    if args.ablation_p:
        parts.append(args.ablation_p)
    return "_".join(parts)


def _build_ablation_config(mode):
    mode = (mode or "A2").upper()
    if mode == "A0":
        return SimpleNamespace(
            mode="A0",
            enable_ieb=False,
            enable_ha=False,
            direct_ieb=False,
            unet_train="cross_attn",
        )
    if mode == "A1":
        return SimpleNamespace(
            mode="A1",
            enable_ieb=True,
            enable_ha=False,
            direct_ieb=True,
            unet_train="frozen",
        )
    if mode == "A2":
        return SimpleNamespace(
            mode="A2",
            enable_ieb=True,
            enable_ha=True,
            direct_ieb=False,
            unet_train="frozen",
        )
    raise ValueError(f"Unknown ablation_mode: {mode}")


def _enable_unet_cross_attn_training(unet):
    trainable_params = []
    for name, module in unet.named_modules():
        if ".attn2" in name:
            module.requires_grad_(True)
            trainable_params.extend([p for p in module.parameters() if p.requires_grad])
    return trainable_params


def _get_autocast_context(device, dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _resolve_weight_dtype(accelerator):
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
    return weight_dtype


def _build_dataloader_kwargs(num_workers, pin_memory, persistent_workers, is_windows):
    num_workers = max(int(num_workers), 0)
    use_persistent = bool(persistent_workers) and num_workers > 0
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": use_persistent,
    }

    if is_windows:
        if num_workers > 0:
            print("[WARN] Windows DataLoader: num_workers > 0 can be unstable; consider 0 for stability.")
            if use_persistent:
                print("[WARN] Windows DataLoader: persistent_workers may be unstable; consider disabling it.")
            try:
                import multiprocessing as mp
                kwargs["multiprocessing_context"] = mp.get_context("spawn")
            except Exception as e:
                print(f"[WARN] Failed to set multiprocessing_context='spawn': {e}")
        else:
            kwargs["persistent_workers"] = False

    if persistent_workers and num_workers == 0:
        print("[WARN] DataLoader persistent_workers requested but num_workers=0; disabling persistent_workers.")
        kwargs["persistent_workers"] = False

    return kwargs


def _log_prompt_cache_stats(logger_obj, dataset, label):
    if dataset is None:
        return
    total = dataset.cache_hit_count + dataset.cache_miss_count
    if total <= 0:
        return
    hit_rate = (dataset.cache_hit_count / total) * 100.0
    logger_obj.info(
        f"[PromptCache] {label}: hits={dataset.cache_hit_count}, "
        f"misses={dataset.cache_miss_count}, hit_rate={hit_rate:.1f}%"
    )


def _quote_cmd_arg(value):
    if value is None:
        return ""
    text = str(value)
    if text == "":
        return "\"\""
    if any(ch in text for ch in (" ", "\t", "\"")):
        return "\"" + text.replace("\"", "\\\"") + "\""
    return text


def _build_prompt_regen_command(args):
    script_path = sys.argv[0] if sys.argv and sys.argv[0] else "train_extractor_DC.py"
    cmd = [
        "python",
        _quote_cmd_arg(script_path),
        "--pretrained_model_name_or_path", _quote_cmd_arg(args.pretrained_model_name_or_path),
        "--image_encoder_path", _quote_cmd_arg(args.image_encoder_path),
        "--dresscode_root", _quote_cmd_arg(args.dresscode_root),
        "--dresscode_category", _quote_cmd_arg(args.dresscode_category),
        "--dresscode_test_order", _quote_cmd_arg(args.dresscode_test_order),
        "--split", _quote_cmd_arg(args.split),
        "--prompts_cache", _quote_cmd_arg(args.prompts_cache),
    ]

    if getattr(args, "pretrained_vae_model_path", None):
        cmd += ["--pretrained_vae_model_path", _quote_cmd_arg(args.pretrained_vae_model_path)]
    if getattr(args, "pretrained_adapter_model_path", None):
        cmd += ["--pretrained_adapter_model_path", _quote_cmd_arg(args.pretrained_adapter_model_path)]
    if getattr(args, "pretrained_text_model_path", None):
        cmd += ["--pretrained_text_model_path", _quote_cmd_arg(args.pretrained_text_model_path)]
    if getattr(args, "prompt_instruction_file", None):
        cmd += ["--prompt_instruction_file", _quote_cmd_arg(args.prompt_instruction_file)]
    if getattr(args, "prompt_instruction_text", None):
        cmd += ["--prompt_instruction_text", _quote_cmd_arg(args.prompt_instruction_text)]
    if getattr(args, "openai_prompt_use_person", False):
        cmd.append("--openai_prompt_use_person")

    cmd += [
        "--openai_prompt_model", _quote_cmd_arg(args.openai_prompt_model),
        "--openai_prompt_max_retries", str(args.openai_prompt_max_retries),
        "--openai_prompt_sleep_base", str(args.openai_prompt_sleep_base),
        "--openai_prompt_word_min", str(args.openai_prompt_word_min),
        "--openai_prompt_word_max", str(args.openai_prompt_word_max),
        "--openai_prompt_max_side", str(args.openai_prompt_max_side),
        "--openai_prompt_jpeg_quality", str(args.openai_prompt_jpeg_quality),
        "--prompt_max_pixels", str(args.prompt_max_pixels),
        "--prompt_max_tokens", str(args.prompt_max_tokens),
        "--prompt_workers", str(args.prompt_workers),
        "--auto_prepare_prompts",
        "--prepare_prompts_only",
    ]

    return " ".join(part for part in cmd if part != "")


def enable_memory_saving(unet=None, ref_unet=None, vae=None, low_vram=False,
                         enable_xformers=False, gradient_checkpointing=True, sdpa_mode="auto"):
    status = {}

    def _try(label, fn):
        try:
            fn()
            status[label] = "on"
        except Exception as e:
            status[label] = "failed"
            print(f"[WARN] {label} failed: {e}")

    def _try_xformers(label, fn):
        try:
            fn()
            status[label] = "on"
        except (ImportError, AttributeError) as e:
            status[label] = "unavailable"
            print(f"[WARN] {label} unavailable: {e}")
        except Exception as e:
            status[label] = "failed"
            print(f"[WARN] {label} failed: {e}")

    # Enable gradient checkpointing for main unet (compatible with custom processors)
    if gradient_checkpointing:
        if unet is not None and hasattr(unet, "enable_gradient_checkpointing"):
            _try("unet_grad_ckpt", unet.enable_gradient_checkpointing)
        else:
            status["unet_grad_ckpt"] = "unsupported"
    else:
        status["unet_grad_ckpt"] = "off"

    # Enable gradient checkpointing for ref_unet (compatible with CacheAttnProcessor2_0)
    # Note: ref_unet uses CacheAttnProcessor2_0 which stores features in cache.
    # Gradient checkpointing may cause multiple forward passes, but since we clear cache
    # at the start of each forward pass, this is safe.
    if gradient_checkpointing:
        if ref_unet is not None and hasattr(ref_unet, "enable_gradient_checkpointing"):
            _try("ref_unet_grad_ckpt", ref_unet.enable_gradient_checkpointing)
        else:
            status["ref_unet_grad_ckpt"] = "unsupported"
    else:
        status["ref_unet_grad_ckpt"] = "off"

    if low_vram:
        # IMPORTANT: Attention slicing replaces ALL processors with SlicedAttnProcessor.
        # We must save custom processors first, then restore them after enabling slicing.
        saved_unet_processors = None
        saved_ref_unet_processors = None

        if unet is not None and hasattr(unet, "set_attention_slice"):
            # Save custom processors before enabling slicing
            saved_unet_processors = unet.attn_processors.copy()
            _try("unet_attn_slice", lambda: unet.set_attention_slice("max"))
            # Restore custom processors after slicing is enabled
            if saved_unet_processors is not None:
                unet.set_attn_processor(saved_unet_processors)
                status["unet_attn_slice"] = "on (processors restored)"
        else:
            status["unet_attn_slice"] = "unsupported"

        if ref_unet is not None and hasattr(ref_unet, "set_attention_slice"):
            # Save cache processors before enabling slicing
            saved_ref_unet_processors = ref_unet.attn_processors.copy()
            _try("ref_unet_attn_slice", lambda: ref_unet.set_attention_slice("max"))
            # Restore cache processors after slicing is enabled
            if saved_ref_unet_processors is not None:
                ref_unet.set_attn_processor(saved_ref_unet_processors)
                status["ref_unet_attn_slice"] = "on (processors restored)"
        else:
            status["ref_unet_attn_slice"] = "unsupported"

        if vae is not None and hasattr(vae, "enable_slicing"):
            _try("vae_slicing", vae.enable_slicing)
        else:
            status["vae_slicing"] = "unsupported"

        if vae is not None and hasattr(vae, "enable_tiling"):
            _try("vae_tiling", vae.enable_tiling)
        else:
            status["vae_tiling"] = "unsupported"

    if enable_xformers:
        if unet is not None and hasattr(unet, "enable_xformers_memory_efficient_attention"):
            _try_xformers("unet_xformers", unet.enable_xformers_memory_efficient_attention)
        else:
            status["unet_xformers"] = "unsupported"

        if ref_unet is not None and hasattr(ref_unet, "enable_xformers_memory_efficient_attention"):
            _try_xformers("ref_unet_xformers", ref_unet.enable_xformers_memory_efficient_attention)
        else:
            status["ref_unet_xformers"] = "unsupported"
    else:
        status["unet_xformers"] = "off"
        status["ref_unet_xformers"] = "off"

    if not enable_xformers or status.get("unet_xformers") != "on" or status.get("ref_unet_xformers") != "on":
        status.update(_configure_sdpa_backends(sdpa_mode))
    else:
        status["flash_sdp"] = "skipped"
        status["mem_efficient_sdp"] = "skipped"
        status["math_sdp"] = "skipped"

    return status


def _configure_sdpa_backends(mode="auto"):
    status = {
        "flash_sdp": "unsupported",
        "mem_efficient_sdp": "unsupported",
        "math_sdp": "unsupported",
    }
    if not torch.cuda.is_available():
        return status

    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return status

    flash_available = False
    mem_available = False

    if hasattr(cuda_backends, "is_flash_sdp_available"):
        try:
            flash_available = bool(cuda_backends.is_flash_sdp_available())
        except Exception:
            flash_available = False
    if hasattr(cuda_backends, "is_mem_efficient_sdp_available"):
        try:
            mem_available = bool(cuda_backends.is_mem_efficient_sdp_available())
        except Exception:
            mem_available = False

    def _try_set(label, fn, value):
        try:
            fn(value)
            status[label] = "on" if value else "off"
            return True
        except Exception as e:
            status[label] = "failed"
            return False

    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "math", "flash"):
        mode = "auto"

    if mode == "math":
        desired_flash = False
        desired_mem = False
        desired_math = True
    elif mode == "flash":
        if flash_available:
            desired_flash = True
            desired_mem = False
            desired_math = False
        else:
            desired_flash = False
            desired_mem = False
            desired_math = True
    else:
        desired_flash = True if flash_available else False
        desired_mem = True if mem_available else False
        desired_math = not (flash_available or mem_available)

    if hasattr(cuda_backends, "enable_flash_sdp"):
        _try_set("flash_sdp", cuda_backends.enable_flash_sdp, desired_flash)
    if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
        _try_set("mem_efficient_sdp", cuda_backends.enable_mem_efficient_sdp, desired_mem)
    if hasattr(cuda_backends, "enable_math_sdp"):
        _try_set("math_sdp", cuda_backends.enable_math_sdp, desired_math)

    return status


def _safe_cuda_call(fn, *args, **kwargs):
    if not torch.cuda.is_available():
        return
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


def _safe_cuda_empty_cache():
    _safe_cuda_call(torch.cuda.empty_cache)


def _safe_cuda_synchronize():
    _safe_cuda_call(torch.cuda.synchronize)


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


def _log_cuda_memory_stats(logger_obj, tag: str, step: int):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    rsv = torch.cuda.memory_reserved() / (1024 ** 3)
    max_rsv = torch.cuda.max_memory_reserved() / (1024 ** 3)
    logger_obj.info(
        f"[CUDA] {tag} step {step}: alloc={alloc:.2f} GB, reserved={rsv:.2f} GB, max_reserved={max_rsv:.2f} GB"
    )


def _apply_vram_soft_cap(device, cap_gb: float):
    if cap_gb is None or cap_gb <= 0:
        return None
    if not torch.cuda.is_available():
        return None
    if not hasattr(torch.cuda, "set_per_process_memory_fraction"):
        return None
    try:
        device_index = device.index if device is not None and device.index is not None else torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        if total_gb <= 0:
            return None
        fraction = min(cap_gb / total_gb, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        return fraction, total_gb
    except Exception:
        return None


def save_model_checkpoint(output_dir, global_steps, sd_model, epoch,
                          accelerator, ema_model=None, prompt_config=None,
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
        "ablation_mode": getattr(unwrapped, "ablation_mode", "A2"),
        "ablation_flags": {
            "enable_ieb": getattr(unwrapped, "enable_ieb", True),
            "enable_ha": getattr(unwrapped, "enable_ha", True),
            "direct_ieb": getattr(unwrapped, "direct_ieb", False),
            "unet_train": getattr(unwrapped, "unet_train_mode", "frozen"),
        },
    }
    unet_has_trainable = any(p.requires_grad for p in unwrapped.unet.parameters())
    if unet_has_trainable:
        ckpt["unet"] = to_cpu_state_dict(unwrapped.unet.state_dict())
    # cross-attn 只存可训练的
    cross = {}
    for name, proc in unwrapped.unet.attn_processors.items():
        if isinstance(proc, RefCAttnProcessor2_0):
            cross[name] = to_cpu_state_dict(proc.state_dict())
    ckpt["cross_attn_processors"] = cross

    if loss_value is not None:
        ckpt["loss"] = float(loss_value)

    if prompt_config is not None:
        ckpt["prompt_config"] = dict(prompt_config)

    # Save EMA weights if available
    if ema_model is not None:
        ckpt["ema"] = {k: v.cpu() for k, v in ema_model.state_dict()['shadow_params'].items()}
        ckpt["ema_decay"] = ema_model.decay

    # 保存前尽量把 GPU 压力降到最低
    _safe_cuda_synchronize()
    _safe_cuda_empty_cache()
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
        _safe_cuda_empty_cache()
        _safe_cuda_synchronize()

    return final_path


def _load_checkpoint_safely(checkpoint_path, accelerator, sd_model):
    if not checkpoint_path:
        return None
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    unwrapped = accelerator.unwrap_model(sd_model)

    def _warn(msg):
        if accelerator is None or accelerator.is_main_process:
            logger.warning(msg)

    def _load_state(module, key, name):
        if module is None:
            _warn(f"[CKPT] Skip {name}: module not available.")
            return
        state = ckpt.get(key)
        if state is None:
            _warn(f"[CKPT] Missing {key}; keeping {name} initialization.")
            return
        try:
            result = module.load_state_dict(state, strict=False)
            if result.missing_keys:
                _warn(f"[CKPT] {name} missing_keys={len(result.missing_keys)}")
            if result.unexpected_keys:
                _warn(f"[CKPT] {name} unexpected_keys={len(result.unexpected_keys)}")
        except Exception as exc:
            _warn(f"[CKPT] Failed to load {name}: {exc}")

    _load_state(unwrapped.proj, "image_proj", "image_proj")
    _load_state(unwrapped.ref_unet, "ref_unet", "ref_unet")

    if "unet" in ckpt:
        _load_state(unwrapped.unet, "unet", "unet")
    else:
        _warn("[CKPT] Missing unet weights; skipping UNet load.")

    cross = ckpt.get("cross_attn_processors")
    expected_refc = [
        name for name, proc in unwrapped.unet.attn_processors.items()
        if isinstance(proc, RefCAttnProcessor2_0)
    ]
    has_refc = bool(expected_refc)
    if cross is None:
        if has_refc:
            _warn("[CKPT] Missing cross_attn_processors; keeping cross-attn initialization.")
    else:
        loaded = 0
        errors = 0
        skipped = 0
        for name, state in cross.items():
            proc = unwrapped.unet.attn_processors.get(name)
            if proc is None or not isinstance(proc, RefCAttnProcessor2_0):
                skipped += 1
                continue
            try:
                result = proc.load_state_dict(state, strict=False)
                loaded += 1
                if result.missing_keys or result.unexpected_keys:
                    _warn(
                        f"[CKPT] cross_attn {name} missing={len(result.missing_keys)} "
                        f"unexpected={len(result.unexpected_keys)}"
                    )
            except Exception as exc:
                errors += 1
                _warn(f"[CKPT] Failed to load cross_attn {name}: {exc}")
        missing = sum(1 for name in expected_refc if name not in cross)
        if has_refc and loaded == 0:
            _warn("[CKPT] No RefCAttnProcessor2_0 weights loaded (mode mismatch?)")
        if missing > 0:
            _warn(f"[CKPT] Missing {missing} cross-attn processor weights; keeping init.")
        if skipped > 0:
            _warn(f"[CKPT] Skipped {skipped} cross-attn processor weights (name/type mismatch).")
        if errors > 0:
            _warn(f"[CKPT] Failed to load {errors} cross-attn processor weights.")

    return ckpt


def run_validation(model, validation_dataloader, vae, text_encoder, image_encoder,
                  noise_scheduler, accelerator, weight_dtype, save_samples=False,
                  step=None, output_dir=None, val_preview_max=4):
    """Run validation and return metrics"""
    # Clear cache before validation to avoid memory spike
    import gc
    gc.collect()
    _safe_cuda_empty_cache()

    model.eval()
    total_loss = 0.0
    num_samples = 0
    sample_count = 0
    use_ieb = getattr(model, "enable_ieb", True)

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
            person_image_embeds = None
            if use_ieb:
                clip_person_images = batch["clip_image"]  # Use all images without dropout
                with _get_autocast_context(accelerator.device, weight_dtype):
                    person_image_embeds = image_encoder(
                        clip_person_images.to(accelerator.device, dtype=weight_dtype),
                        output_hidden_states=True
                    ).hidden_states[-2]
                if person_image_embeds.dtype != weight_dtype:
                    person_image_embeds = person_image_embeds.to(dtype=weight_dtype)

            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
            if encoder_hidden_states.dtype != weight_dtype:
                encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)

            target = noise if noise_scheduler.prediction_type == "epsilon" else \
                     noise_scheduler.get_velocity(garment_latents, noise, timesteps)

            if noisy_garment_latents.dtype != weight_dtype:
                noisy_garment_latents = noisy_garment_latents.to(dtype=weight_dtype)
            if person_latents.dtype != weight_dtype:
                person_latents = person_latents.to(dtype=weight_dtype)
            with _get_autocast_context(accelerator.device, weight_dtype):
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
                    single_person_image_embeds = None
                    if person_image_embeds is not None:
                        single_person_image_embeds = person_image_embeds[0:1]
                    single_encoder_hidden_states = encoder_hidden_states[0:1]
                    single_garment_latent = garment_latents[0:1]  # Ground truth for reference
                    if single_person_latents.dtype != weight_dtype:
                        single_person_latents = single_person_latents.to(dtype=weight_dtype)
                    if single_person_image_embeds is not None and single_person_image_embeds.dtype != weight_dtype:
                        single_person_image_embeds = single_person_image_embeds.to(dtype=weight_dtype)
                    if single_encoder_hidden_states.dtype != weight_dtype:
                        single_encoder_hidden_states = single_encoder_hidden_states.to(dtype=weight_dtype)

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
                        latents_for_model = latents
                        if latents_for_model.dtype != weight_dtype:
                            latents_for_model = latents_for_model.to(dtype=weight_dtype)
                        with _get_autocast_context(accelerator.device, weight_dtype):
                            noise_pred = model(
                                single_encoder_hidden_states,
                                latents_for_model,
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
            if hasattr(model, "unet"):
                _clear_unet_extra_state(model.unet)

    model.train()

    # 验证结束后立刻归还显存
    _safe_cuda_empty_cache()

    if accelerator.is_main_process:
        logger.info(f"VAL: Completed validation, avg loss: {total_loss / len(validation_dataloader):.4f}")

    return total_loss / len(validation_dataloader) if len(validation_dataloader) > 0 else float('inf')


def _build_pair_id(category, image_file, cloth_file):
    pair_id = f"{category}___{image_file}___{cloth_file}"
    for ch in ("/", "\\", ":", "|"):
        pair_id = pair_id.replace(ch, "_")
    pair_id = pair_id.replace("|||", "__")
    return pair_id


def _resolve_pair_meta(pair_items, pair_cursor, batch_idx, sample_idx, image_files, cloth_files):
    if pair_items is not None and pair_cursor + sample_idx < len(pair_items):
        meta = pair_items[pair_cursor + sample_idx]
        category = meta.get("category", "unknown")
        image_file = meta.get("image_file", f"{batch_idx:06d}.jpg")
        cloth_file = meta.get("cloth_file", f"{batch_idx:06d}_cloth.jpg")
        return category, image_file, cloth_file
    category = "unknown"
    image_file = image_files[sample_idx] if sample_idx < len(image_files) else f"{batch_idx:06d}.jpg"
    cloth_file = cloth_files[sample_idx] if sample_idx < len(cloth_files) else f"{batch_idx:06d}_cloth.jpg"
    return category, image_file, cloth_file


def _resolve_visual_metrics_dirs(output_dir):
    if not output_dir:
        return None, None, None
    base_dir = os.path.join(output_dir, "visual_metrics")
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")
    return base_dir, real_dir, fake_dir


def run_eval_only_with_saving(model, validation_dataloader, vae, text_encoder, image_encoder,
                              noise_scheduler, accelerator, weight_dtype,
                              output_dir, save_pairs=False, pair_items=None,
                              eval_denoise_steps=12, eval_cleanup_every=8,
                              eval_skip_existing_pairs=True, eval_oom_fallback_steps=6):
    import gc
    import traceback

    gc.collect()
    _safe_cuda_empty_cache()

    model.eval()
    total_loss = 0.0
    num_batches = 0
    use_ieb = getattr(model, "enable_ieb", True)

    skipped_count = 0
    processed_count = 0
    resumed_existing_count = 0
    oom_retry_count = 0
    progress_every = 10

    gt_dir = None
    gen_dir = None
    if save_pairs and output_dir:
        _, gt_dir, gen_dir = _resolve_visual_metrics_dirs(output_dir)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)

    eval_scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    eval_alphas_cumprod = eval_scheduler.alphas_cumprod.to(accelerator.device)

    attempt_steps = []
    primary_steps = max(1, int(eval_denoise_steps))
    attempt_steps.append(primary_steps)
    fallback_steps = max(0, int(eval_oom_fallback_steps))
    if 0 < fallback_steps < primary_steps:
        attempt_steps.append(fallback_steps)
    half_fallback = max(1, fallback_steps // 2) if fallback_steps > 1 else 0
    if 0 < half_fallback < primary_steps and half_fallback not in attempt_steps:
        attempt_steps.append(half_fallback)

    pair_cursor = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(validation_dataloader):
            person_latents = None
            garment_latents = None
            noise = None
            timesteps = None
            noisy_garment_latents = None
            person_image_embeds = None
            encoder_hidden_states = None
            target = None
            model_pred = None
            loss = None
            bsz = 1
            if isinstance(batch, dict):
                for key in ("vae_clothes", "input_ids", "image_files", "cloth_files"):
                    if key in batch:
                        value = batch[key]
                        if torch.is_tensor(value):
                            bsz = value.shape[0]
                        elif isinstance(value, (list, tuple)):
                            bsz = len(value)
                        break

            try:
                image_files = batch.get("image_files", [])
                cloth_files = batch.get("cloth_files", [])

                # Resume fast-path: when all outputs in this batch already exist, skip compute entirely.
                if save_pairs and gt_dir and gen_dir and eval_skip_existing_pairs and accelerator.is_main_process:
                    existing_in_batch = 0
                    for i in range(bsz):
                        category, image_file, cloth_file = _resolve_pair_meta(
                            pair_items, pair_cursor, batch_idx, i, image_files, cloth_files
                        )
                        pair_id = _build_pair_id(category, image_file, cloth_file)
                        gt_path = os.path.join(gt_dir, f"{pair_id}_groundtruth.png")
                        gen_path = os.path.join(gen_dir, f"{pair_id}_generated.png")
                        if os.path.isfile(gt_path) and os.path.isfile(gen_path):
                            existing_in_batch += 1
                    if existing_in_batch == bsz:
                        resumed_existing_count += existing_in_batch
                        processed_count += bsz
                        if progress_every > 0 and processed_count % progress_every == 0:
                            max_mem_mb = 0.0
                            if torch.cuda.is_available():
                                try:
                                    max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                                except Exception:
                                    max_mem_mb = 0.0
                            accelerator.print(
                                f"[EvalOnly] Processed {processed_count} pairs (skipped={skipped_count}, resumed={resumed_existing_count}) "
                                f"max_mem={max_mem_mb:.1f} MB"
                            )
                        continue

                person_latents = vae.encode(
                    batch["vae_person"].to(accelerator.device, dtype=weight_dtype)
                ).latent_dist.sample()
                person_latents = person_latents * 0.18215

                garment_latents = vae.encode(
                    batch["vae_clothes"].to(accelerator.device, dtype=weight_dtype)
                ).latent_dist.sample()
                garment_latents = garment_latents * 0.18215

                noise = torch.randn_like(garment_latents)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=garment_latents.device)
                timesteps = timesteps.long()

                noisy_garment_latents = noise_scheduler.add_noise(garment_latents, noise, timesteps)

                person_image_embeds = None
                if use_ieb:
                    clip_person_images = batch["clip_image"]
                    with _get_autocast_context(accelerator.device, weight_dtype):
                        person_image_embeds = image_encoder(
                            clip_person_images.to(accelerator.device, dtype=weight_dtype),
                            output_hidden_states=True
                        ).hidden_states[-2]
                    if person_image_embeds.dtype != weight_dtype:
                        person_image_embeds = person_image_embeds.to(dtype=weight_dtype)

                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                if encoder_hidden_states.dtype != weight_dtype:
                    encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)

                target = noise if noise_scheduler.prediction_type == "epsilon" else \
                    noise_scheduler.get_velocity(garment_latents, noise, timesteps)

                if noisy_garment_latents.dtype != weight_dtype:
                    noisy_garment_latents = noisy_garment_latents.to(dtype=weight_dtype)
                if person_latents.dtype != weight_dtype:
                    person_latents = person_latents.to(dtype=weight_dtype)
                with _get_autocast_context(accelerator.device, weight_dtype):
                    model_pred = model(
                        encoder_hidden_states,
                        noisy_garment_latents,
                        person_latents,
                        person_image_embeds,
                        timesteps
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                total_loss += loss.item()
                num_batches += 1

                if save_pairs and gt_dir and gen_dir and accelerator.is_main_process:
                    for i in range(bsz):
                        single_noisy_latents = None
                        single_timestep = None
                        single_person_latents = None
                        single_person_image_embeds = None
                        single_encoder_hidden_states = None
                        latents = None
                        noise_pred = None
                        pred_image = None
                        gt_image = None
                        latents_to_decode = None
                        try:
                            category, image_file, cloth_file = _resolve_pair_meta(
                                pair_items, pair_cursor, batch_idx, i, image_files, cloth_files
                            )
                            pair_id = _build_pair_id(category, image_file, cloth_file)
                            gt_path = os.path.join(gt_dir, f"{pair_id}_groundtruth.png")
                            gen_path = os.path.join(gen_dir, f"{pair_id}_generated.png")

                            if eval_skip_existing_pairs and os.path.isfile(gt_path) and os.path.isfile(gen_path):
                                resumed_existing_count += 1
                                continue

                            single_noisy_latents = noisy_garment_latents[i:i + 1]
                            single_timestep = timesteps[i:i + 1]
                            single_person_latents = person_latents[i:i + 1]
                            single_person_image_embeds = None
                            if person_image_embeds is not None:
                                single_person_image_embeds = person_image_embeds[i:i + 1]
                            single_encoder_hidden_states = encoder_hidden_states[i:i + 1]

                            if single_person_latents.dtype != weight_dtype:
                                single_person_latents = single_person_latents.to(dtype=weight_dtype)
                            if single_person_image_embeds is not None and single_person_image_embeds.dtype != weight_dtype:
                                single_person_image_embeds = single_person_image_embeds.to(dtype=weight_dtype)
                            if single_encoder_hidden_states.dtype != weight_dtype:
                                single_encoder_hidden_states = single_encoder_hidden_states.to(dtype=weight_dtype)

                            for attempt_idx, denoise_steps in enumerate(attempt_steps):
                                try:
                                    current_timesteps = _build_eval_timesteps(single_timestep.item(), denoise_steps)
                                    latents = single_noisy_latents.clone()

                                    for j in range(len(current_timesteps) - 1):
                                        t_current = current_timesteps[j]
                                        t_next = current_timesteps[j + 1]
                                        timestep_tensor = torch.tensor([t_current], device=latents.device, dtype=torch.long)

                                        latents_for_model = latents
                                        if latents_for_model.dtype != weight_dtype:
                                            latents_for_model = latents_for_model.to(dtype=weight_dtype)
                                        with _get_autocast_context(accelerator.device, weight_dtype):
                                            noise_pred = model(
                                                single_encoder_hidden_states,
                                                latents_for_model,
                                                single_person_latents,
                                                single_person_image_embeds,
                                                timestep_tensor
                                            )

                                        alpha_prod_t = eval_alphas_cumprod[t_current]
                                        alpha_prod_t_prev = eval_alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=latents.device)
                                        beta_prod_t = 1 - alpha_prod_t

                                        pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                                        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
                                        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

                                        latents = prev_sample

                                    vae_dtype = next(vae.parameters()).dtype
                                    vae_device = next(vae.parameters()).device
                                    latents_to_decode = (latents / 0.18215).to(device=vae_device, dtype=vae_dtype)
                                    with _get_autocast_context(vae_device, vae_dtype):
                                        pred_image = vae.decode(latents_to_decode).sample

                                    pred_image = (pred_image.float() + 1.0) / 2.0
                                    pred_image = torch.clamp(pred_image, 0.0, 1.0).to("cpu")

                                    gt_source = batch["vae_clothes"][i:i + 1]
                                    if not torch.is_tensor(gt_source):
                                        raise RuntimeError("vae_clothes is not a tensor; cannot save eval groundtruth image.")
                                    gt_image = torch.clamp((gt_source.float() + 1.0) / 2.0, 0.0, 1.0).to("cpu")

                                    save_image(gt_image, gt_path)
                                    save_image(pred_image, gen_path)
                                    break
                                except RuntimeError as inner_exc:
                                    if _is_cuda_oom_error(inner_exc) and attempt_idx + 1 < len(attempt_steps):
                                        oom_retry_count += 1
                                        if accelerator.is_main_process:
                                            logger.warning(
                                                "[EvalOnly] CUDA OOM at batch %s sample %s; retrying with %s denoise steps.",
                                                batch_idx,
                                                i,
                                                attempt_steps[attempt_idx + 1],
                                            )
                                        latents = None
                                        noise_pred = None
                                        pred_image = None
                                        gt_image = None
                                        latents_to_decode = None
                                        cuda_memory_cleanup(tag="eval_save_retry", deep=True, do_ipc=True)
                                        continue
                                    raise
                        except Exception as save_exc:
                            skipped_count += 1
                            if _is_cuda_oom_error(save_exc):
                                cuda_memory_cleanup(tag="eval_save_exception", deep=True, do_ipc=True)
                            if accelerator.is_main_process:
                                logger.warning(
                                    f"[EvalOnly] Save failed for batch {batch_idx} sample {i}; skipping."
                                )
                                traceback.print_exc()
                        finally:
                            single_noisy_latents = None
                            single_timestep = None
                            single_person_latents = None
                            single_person_image_embeds = None
                            single_encoder_hidden_states = None
                            latents = None
                            noise_pred = None
                            pred_image = None
                            gt_image = None
                            latents_to_decode = None
                            pair_index = pair_cursor + i + 1
                            if eval_cleanup_every > 0 and pair_index % eval_cleanup_every == 0:
                                cuda_memory_cleanup(tag="eval_periodic", deep=True, do_ipc=True)

                processed_count += bsz
                if accelerator.is_main_process and progress_every > 0 and processed_count % progress_every == 0:
                    max_mem_mb = 0.0
                    if torch.cuda.is_available():
                        try:
                            max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        except Exception:
                            max_mem_mb = 0.0
                    accelerator.print(
                        f"[EvalOnly] Processed {processed_count} pairs (skipped={skipped_count}, resumed={resumed_existing_count}) "
                        f"max_mem={max_mem_mb:.1f} MB"
                    )

            except Exception as outer_exc:
                skipped = bsz
                skipped_count += skipped
                if _is_cuda_oom_error(outer_exc):
                    cuda_memory_cleanup(tag="eval_batch_exception", deep=True, do_ipc=True)
                if accelerator.is_main_process:
                    logger.error(f"[EvalOnly] Batch {batch_idx} failed; skipped {skipped} samples.")
                    traceback.print_exc()

            finally:
                if hasattr(model, "ref_unet"):
                    _clear_ref_unet_cache(model.ref_unet)
                if hasattr(model, "unet"):
                    _clear_unet_extra_state(model.unet)

                person_latents = None
                garment_latents = None
                noise = None
                timesteps = None
                noisy_garment_latents = None
                person_image_embeds = None
                encoder_hidden_states = None
                target = None
                model_pred = None
                loss = None

                _safe_cuda_empty_cache()
                pair_cursor += bsz

    model.train()
    _safe_cuda_empty_cache()

    if accelerator.is_main_process and resumed_existing_count > 0:
        accelerator.print(f"[EvalOnly] Reused existing pair images: {resumed_existing_count}")
    if accelerator.is_main_process and oom_retry_count > 0:
        accelerator.print(f"[EvalOnly] OOM retries succeeded: {oom_retry_count}")

    if num_batches == 0:
        return float("inf"), skipped_count
    return total_loss / num_batches, skipped_count


def _calculate_pixel_metrics(real_dir, generated_dir, device):
    import glob
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import lpips

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    real_paths = sorted(glob.glob(os.path.join(real_dir, "*_groundtruth.png")))
    gen_paths = sorted(glob.glob(os.path.join(generated_dir, "*_generated.png")))
    if len(real_paths) == 0 or len(gen_paths) == 0:
        print("[WARN] Pixel metrics skipped: no images found.")
        return None

    if len(real_paths) != len(gen_paths):
        min_count = min(len(real_paths), len(gen_paths))
        real_paths = real_paths[:min_count]
        gen_paths = gen_paths[:min_count]

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    psnr_values = []
    ssim_values = []
    lpips_values = []
    resize_count = 0
    skipped_count = 0
    warned_resize = False
    num_pairs = len(real_paths)

    iterator = zip(real_paths, gen_paths)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(real_paths), desc="Pixel metrics")

    for real_path, gen_path in iterator:
        try:
            real_img = np.array(Image.open(real_path).convert("RGB"))
            gen_img = np.array(Image.open(gen_path).convert("RGB"))

            if real_img.shape[:2] != gen_img.shape[:2]:
                if not warned_resize:
                    print("[WARN] Resizing generated images to match groundtruth size for pixel metrics.")
                    warned_resize = True
                resize_count += 1
                gen_img = np.array(
                    Image.fromarray(gen_img).resize(
                        (real_img.shape[1], real_img.shape[0]),
                        resample=Image.BICUBIC,
                    )
                )

            psnr_values.append(psnr(real_img, gen_img, data_range=255))
            try:
                ssim_val = ssim(real_img, gen_img, channel_axis=2, data_range=255)
            except TypeError:
                ssim_val = ssim(real_img, gen_img, multichannel=True, data_range=255)
            ssim_values.append(ssim_val)

            real_tensor = torch.from_numpy(real_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            gen_tensor = torch.from_numpy(gen_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            real_tensor = real_tensor.to(device)
            gen_tensor = gen_tensor.to(device)

            with torch.no_grad():
                lpips_val = lpips_model(real_tensor, gen_tensor).item()
            lpips_values.append(lpips_val)
        except Exception as exc:
            skipped_count += 1
            print(f"[WARN] Skipping pixel metrics for {real_path}: {type(exc).__name__}: {exc}")
            continue

    psnr_mean = float(np.mean(psnr_values)) if psnr_values else None
    psnr_std = float(np.std(psnr_values)) if psnr_values else None
    ssim_mean = float(np.mean(ssim_values)) if ssim_values else None
    ssim_std = float(np.std(ssim_values)) if ssim_values else None
    lpips_mean = float(np.mean(lpips_values)) if lpips_values else None
    lpips_std = float(np.std(lpips_values)) if lpips_values else None

    del lpips_model
    _safe_cuda_empty_cache()

    return {
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "resize_count": resize_count,
        "num_pairs": num_pairs,
        "skipped_count": skipped_count,
    }


def _compute_fid_kid_standard(real_dir, fake_dir):
    import glob
    import os

    from calculate_fid_kid_standard import (
        InceptionV3FeatureExtractor,
        calculate_fid_standard,
        calculate_kid_standard,
    )

    def _collect_image_paths(folder):
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(folder, pattern)))
        return sorted(paths)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_paths = _collect_image_paths(real_dir)
    fake_paths = _collect_image_paths(fake_dir)

    if not real_paths or not fake_paths:
        raise RuntimeError("FID/KID standard: no images found in real/fake directories.")

    inception = InceptionV3FeatureExtractor(device=device)
    real_feats = inception.extract_features(real_paths, batch_size=32)
    fake_feats = inception.extract_features(fake_paths, batch_size=32)

    fid = calculate_fid_standard(real_feats, fake_feats)
    kid_mean, kid_std = calculate_kid_standard(real_feats, fake_feats)
    if kid_mean is None or kid_std is None:
        raise RuntimeError("KID returned None (insufficient samples).")

    _safe_cuda_empty_cache()
    return float(fid), float(kid_mean), float(kid_std)


def count_model_params_raw(model, trainable_only=False):
    if model is None:
        return 0
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_model_params(model):
    return count_model_params_raw(model) / 1e6


def _clear_nested_refs(obj):
    if obj is None:
        return
    if torch.is_tensor(obj):
        return
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            _clear_nested_refs(obj[k])
            obj[k] = None
        try:
            obj.clear()
        except Exception:
            pass
        return
    if isinstance(obj, (list, set, tuple)):
        for item in obj:
            _clear_nested_refs(item)
        if isinstance(obj, (list, set)):
            try:
                obj.clear()
            except Exception:
                pass


def _clear_ref_unet_cache(ref_unet):
    """
    Clear CacheAttnProcessor cache to avoid cross-step residual refs/VRAM fragmentation.
    """
    if ref_unet is None:
        return
    for proc in ref_unet.attn_processors.values():
        if hasattr(proc, "clear_cache") and callable(proc.clear_cache):
            try:
                proc.clear_cache()
                continue
            except Exception:
                pass
        if hasattr(proc, "cache"):
            _clear_nested_refs(proc.cache)


def _clear_unet_extra_state(unet):
    if unet is None:
        return
    for proc in unet.attn_processors.values():
        for attr in (
            "sa_hidden_states",
            "_sa_hidden_states",
            "_last_sa_hidden_states",
            "last_sa_hidden_states",
            "cached_sa_hidden_states",
        ):
            if hasattr(proc, attr):
                try:
                    setattr(proc, attr, None)
                except Exception:
                    pass
        if hasattr(proc, "cache"):
            _clear_nested_refs(proc.cache)


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or "cuda error: cublas_status_alloc_failed" in msg
    )


def _build_eval_timesteps(start_timestep: int, max_steps: int):
    start = max(int(start_timestep), 0)
    max_steps = max(int(max_steps), 1)
    timesteps = list(range(start, -1, -1))
    if len(timesteps) > max_steps:
        step_size = max(1, len(timesteps) // max_steps)
        timesteps = timesteps[::step_size]
        if len(timesteps) > max_steps:
            timesteps = timesteps[:max_steps]
    if not timesteps:
        timesteps = [0]
    if timesteps[-1] != 0:
        timesteps.append(0)
    return timesteps


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

    def __init__(self, unet, ref_unet, proj, adapter_modules, ablation_config=None) -> None:
        super().__init__()

        # Denoising UNet (frozen except hybrid attention cross-attention)
        self.unet = unet

        # Garment UNet (trainable) - now extracts features from PERSON image
        self.ref_unet = ref_unet

        # Projection layer (trainable) - maps CLIP image features
        self.proj = proj

        # Hybrid Attention adapter modules (trainable cross-attention)
        self.adapter_modules = adapter_modules

        self.ablation_mode = getattr(ablation_config, "mode", "A2")
        self.enable_ieb = getattr(ablation_config, "enable_ieb", True)
        self.enable_ha = getattr(ablation_config, "enable_ha", True)
        self.direct_ieb = getattr(ablation_config, "direct_ieb", False)
        self.unet_train_mode = getattr(ablation_config, "unet_train", "frozen")

    def forward(self, encoder_hidden_states, latents, ref_latents, clip_image_embeddings, timesteps):
        """
        REVERSE task forward pass:
        - ref_latents: person latents (input to Garment UNet)
        - latents: garment latents (output target for Denoising UNet)
        - clip_image_embeddings: CLIP features from person image
        """
        amp_dtype = latents.dtype
        if encoder_hidden_states.dtype != amp_dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype=amp_dtype)
        if clip_image_embeddings is not None and clip_image_embeddings.dtype != amp_dtype:
            clip_image_embeddings = clip_image_embeddings.to(dtype=amp_dtype)
        if ref_latents.dtype != amp_dtype:
            ref_latents = ref_latents.to(dtype=amp_dtype)

        person_proj_embed = None
        if self.enable_ieb and clip_image_embeddings is not None:
            with _get_autocast_context(latents.device, amp_dtype):
                person_proj_embed = self.proj(clip_image_embeddings)

        sa_hidden_states = None
        if self.enable_ha:
            if self.ref_unet is None:
                raise RuntimeError("ref_unet is required when enable_ha is True.")
            _clear_ref_unet_cache(self.ref_unet)
            ref_timesteps = torch.zeros_like(timesteps)

            with _get_autocast_context(latents.device, amp_dtype):
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

                # Drop cache refs early to reduce peak memory
                _clear_ref_unet_cache(self.ref_unet)

        if self.direct_ieb and person_proj_embed is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states, person_proj_embed], dim=1)

        with _get_autocast_context(latents.device, amp_dtype):
            # Denoising UNet generates GARMENT using extracted features
            cross_kwargs = {"sa_hidden_states": sa_hidden_states} if self.enable_ha else None
            noise_pred = self.unet(
                latents,  # noisy garment latents
                timesteps,
                encoder_hidden_states=encoder_hidden_states,  # text embeddings
                cross_attention_kwargs=cross_kwargs,
            ).sample

            if not torch.is_grad_enabled() and isinstance(sa_hidden_states, dict):
                sa_hidden_states.clear()

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
            cache = json.load(f)
            if not isinstance(cache, dict):
                return {}
            normalized = {}
            for k, v in cache.items():
                nk = normalize_prompt_cache_key(k)
                normalized[nk] = v
            return normalized
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
            nk = normalize_prompt_cache_key(k)
            if "text" in obj:
                cache[nk] = obj["text"]
            elif "prompt" in obj:
                cache[nk] = obj["prompt"]
            elif "prompts" in obj:
                cache[nk] = obj["prompts"]
    return cache


def _extract_prompt_value(obj):
    if "text" in obj:
        return obj["text"]
    if "prompt" in obj:
        return obj["prompt"]
    if "prompts" in obj:
        return obj["prompts"]
    return None


def _upgrade_prompts_cache_keys(cache_path: str) -> int:
    """Append normalized keys for legacy cache entries that used paths instead of basenames."""
    if not cache_path or not os.path.exists(cache_path):
        return 0

    ext = os.path.splitext(cache_path)[1].lower()
    upgraded = 0

    if ext == ".json":
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if not isinstance(cache, dict):
            return 0

        updated = False
        for key, value in list(cache.items()):
            normalized_key = normalize_prompt_cache_key(key)
            if normalized_key != key and normalized_key not in cache:
                cache[normalized_key] = value
                upgraded += 1
                updated = True

        if updated:
            _atomic_write_json(cache_path, cache)
    else:
        existing_keys = set()
        items = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("key")
                if not key:
                    continue
                existing_keys.add(key)
                items.append(obj)

        new_records = []
        for obj in items:
            key = obj.get("key")
            if not key:
                continue
            normalized_key = normalize_prompt_cache_key(key)
            if normalized_key == key or normalized_key in existing_keys:
                continue
            prompt_value = _extract_prompt_value(obj)
            if prompt_value is None:
                continue
            record = {"key": normalized_key}
            if isinstance(prompt_value, list):
                record["prompts"] = prompt_value
            else:
                record["text"] = prompt_value
                record["prompt"] = prompt_value
            new_records.append(record)
            existing_keys.add(normalized_key)
            upgraded += 1

        if new_records:
            for record in new_records:
                _append_jsonl(cache_path, record)

    if upgraded > 0:
        print(f"[PromptCache] Added {upgraded} normalized keys (basename format) for legacy cache entries.")

    return upgraded

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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    instruction = _load_prompt_instruction(args)

    # Lightweight dataset instance just to read pairs & directories (no image loading at init)
    # NOTE: verify_files=False for prompt generation stage (files will be verified during training)
    tmp_ds = DressCodeDataset(
        root=args.dresscode_root,
        split=args.split,
        tokenizer=tokenizer,
        category=args.dresscode_category,
        test_order=args.dresscode_test_order,
        prompts_cache=args.prompts_cache if (args.prompts_cache and os.path.exists(args.prompts_cache)) else None,
        prompt_mode="fallback_only",
        prompt_fallback=DEFAULT_FALLBACK_PROMPT,
        size=args.train_image_size,
        verify_files=False,  # Skip file verification during prompt generation (faster)
    )

    cache_path = args.prompts_cache
    _upgrade_prompts_cache_keys(cache_path)
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
        key = make_pair_key(item["category"], item["image_file"], item["cloth_file"])

        if key in existing_keys:
            return None  # Already cached

        images_dir = tmp_ds.images_dir_map[item["category"]]
        person_path = os.path.join(images_dir, item["image_file"])
        cloth_path = os.path.join(images_dir, item["cloth_file"])

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

            record = {
                "key": key,
                "text": prompt,
                "prompt": prompt,
                "image_file": item["image_file"],
                "cloth_file": item["cloth_file"],
                "category": item["category"],
            }

            # Thread-safe write
            if ext == ".jsonl":
                with write_lock:
                    _append_jsonl(cache_path, record)

            return {"key": key, "record": record, "error": None}

        except Exception as e:
            return {"key": key, "record": None, "error": str(e)}

    # Collect items to generate
    items_to_generate = []
    for idx, item in enumerate(tmp_ds.data):
        key = make_pair_key(item["category"], item["image_file"], item["cloth_file"])
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
                record = result["record"]
                if ext == ".json":
                    existing[result["key"]] = record
                else:
                    existing[result["key"]] = record["text"]
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
    if "PYTHONFAULTHANDLER" not in os.environ:
        os.environ["PYTHONFAULTHANDLER"] = "1"
    try:
        import faulthandler
        faulthandler.enable()
    except Exception as e:
        print(f"[WARN] faulthandler.enable() failed: {e}")

    args = parse_args()
    apply_low_vram_overrides(args)
    resolve_ablation(args)
    if args.prompt_mode in ("cache_or_fallback", "fallback_only"):
        if args.prompt_fallback is None or str(args.prompt_fallback).strip() == "":
            raise RuntimeError(
                "[PromptFallbackEmpty] --prompt_fallback is empty or whitespace. "
                "Provide a non-empty fallback prompt via --prompt_fallback \"...\", "
                "or use --prompt_mode cache_only and ensure 100% prompt cache coverage."
            )
    if args.eval_only and not args.compute_visual_metrics:
        args.compute_visual_metrics = True
    if args.test_set_json and not args.validation_subset_json:
        args.validation_subset_json = args.test_set_json
    elif args.test_set_json and args.validation_subset_json and args.test_set_json != args.validation_subset_json:
        raise RuntimeError("--test_set_json and --validation_subset_json must match when both are set.")
    effective_ablation_mode = _resolve_effective_ablation_mode(args)
    ablation_spec = _build_ablation_spec(args, effective_ablation_mode)
    build_mode = effective_ablation_mode
    runtime_mode = effective_ablation_mode
    ablation = _build_ablation_config(build_mode)
    runtime_ablation = _build_ablation_config(runtime_mode)
    args.ablation_mode = ablation.mode
    ablation_s_value = args.ablation_s or ""
    ablation_p_value = args.ablation_p or ""
    prompt_config = {
        "prompts_cache_path": args.prompts_cache,
        "prompt_mode": args.prompt_mode,
        "prompt_fallback": args.prompt_fallback or "",
        "prompt_key_format": PROMPT_KEY_FORMAT,
        "ablation_spec": ablation_spec,
        "ablation_s": ablation_s_value,
        "ablation_p": ablation_p_value,
        "effective_ablation_mode": effective_ablation_mode,
        "ablation_mode": ablation.mode,
        "ablation_flags": {
            "enable_ieb": ablation.enable_ieb,
            "enable_ha": ablation.enable_ha,
            "direct_ieb": ablation.direct_ieb,
            "unet_train": ablation.unet_train,
        },
    }
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
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
    accelerator.print(
        "ABLAT_RESOLVED "
        f"spec={ablation_spec}, ablation_s={ablation_s_value}, ablation_p={ablation_p_value}, "
        f"effective_ablation_mode={effective_ablation_mode}, prompt_mode={args.prompt_mode}, "
        f"strict_prompts_cache={args.strict_prompts_cache}, eval_only={args.eval_only}"
    )
    accelerator.print("VRAM CONFIG")
    accelerator.print(
        f"  train_batch_size={args.train_batch_size}, grad_accum={args.gradient_accumulation_steps}, "
        f"mixed_precision={args.mixed_precision}, low_vram={args.low_vram}, use_ema={args.use_ema}, "
        f"ema_device={args.ema_device}, train_image_size={args.train_image_size}"
    )
    accelerator.print("RUNTIME CONFIG")
    if os.name == "nt":
        accelerator.print(f"  sdpa_mode={args.sdpa_mode} (Windows default: math)")
    else:
        accelerator.print(f"  sdpa_mode={args.sdpa_mode}")
    accelerator.print(f"  seed={args.seed}")
    accelerator.print(
        f"  dataloader_num_workers={args.dataloader_num_workers}, "
        f"pin_memory={args.dataloader_pin_memory}, "
        f"persistent_workers={args.dataloader_persistent_workers}"
    )
    accelerator.print(
        f"  mem_debug={args.mem_debug}, cuda_mem_log_steps={args.cuda_mem_log_steps}, "
        f"vram_soft_cap_gb={args.vram_soft_cap_gb}"
    )
    accelerator.print(
        f"  eval_denoise_steps={args.eval_denoise_steps}, eval_cleanup_every={args.eval_cleanup_every}, "
        f"eval_skip_existing_pairs={args.eval_skip_existing_pairs}, eval_oom_fallback_steps={args.eval_oom_fallback_steps}"
    )
    accelerator.print(
        f"  prompts_cache={args.prompts_cache if args.prompts_cache else 'NONE'}, "
        f"prompt_mode={args.prompt_mode}, strict_prompts_cache={args.strict_prompts_cache}, "
        f"min_cache_coverage={args.min_cache_coverage}"
    )
    accelerator.print(
        f"  gradient_checkpointing={args.gradient_checkpointing}, "
        f"xformers_enabled={args.enable_xformers_memory_efficient_attention}"
    )
    accelerator.print("ABLATION CONFIG")
    accelerator.print(
        f"  mode={ablation.mode}, IEB={ablation.enable_ieb}, HA={ablation.enable_ha}, "
        f"direct_IEB={ablation.direct_ieb}, unet_train={ablation.unet_train}"
    )
    if os.name == "nt" and args.enable_xformers_memory_efficient_attention:
        accelerator.print("[WARN] xformers enabled on Windows; if unstable, disable it.")
    if os.name == "nt" and args.dataloader_num_workers > 0:
        accelerator.print("[WARN] Windows DataLoader workers > 0 can be unstable; 0 is recommended.")
    if args.strict_prompts_cache and args.prompt_mode == "fallback_only":
        accelerator.print("[WARN] strict_prompts_cache has no effect with prompt_mode=fallback_only.")
    soft_cap = _apply_vram_soft_cap(accelerator.device, args.vram_soft_cap_gb)
    if args.vram_soft_cap_gb and args.vram_soft_cap_gb > 0:
        if soft_cap:
            fraction, total_gb = soft_cap
            accelerator.print(
                f"  vram_soft_cap_gb={args.vram_soft_cap_gb} (device {total_gb:.1f} GB, fraction {fraction:.3f})"
            )
        else:
            accelerator.print("[WARN] vram_soft_cap_gb requested but not applied (unsupported).")
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
        if args.prompts_cache:
            _upgrade_prompts_cache_keys(args.prompts_cache)
    accelerator.wait_for_everyone()

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
    st = unet.state_dict() if ablation.enable_ha else None
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
            if ablation.enable_ha:
                # Self-attention: use RefSAttnProcessor2_0 (will be frozen)
                attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ref.weight": st[layer_name + ".to_k.weight"],
                    "to_v_ref.weight": st[layer_name + ".to_v.weight"],
                }
                attn_procs[name].load_state_dict(weights)
            else:
                attn_procs[name] = SAttnProcessor2_0(name, hidden_size)
        else:
            if ablation.enable_ha:
                # Cross-attention: use RefCAttnProcessor2_0 (trainable for person features)
                attn_procs[name] = RefCAttnProcessor2_0(
                    name,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
            else:
                attn_procs[name] = CAttnProcessor2_0(
                    name,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
    unet.set_attn_processor(attn_procs)

    if ablation.enable_ha:
        # Freeze self-attention processors (RefSAttnProcessor2_0)
        # Only train cross-attention processors (RefCAttnProcessor2_0)
        for name, proc in unet.attn_processors.items():
            if isinstance(proc, RefSAttnProcessor2_0):
                # Freeze self-attention ref params
                proc.requires_grad_(False)
                proc.eval()  # Set to eval mode to freeze any normalization/dropout

        # Do NOT call adapter_modules.requires_grad_(True) - it would unfreeze self-attention!
        # We'll collect only trainable cross-attention processors into optimizer
    if st is not None:
        del st

    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    ref_unet.set_attn_processor(
        {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_proj.requires_grad_(ablation.enable_ieb)
    ref_unet.requires_grad_(ablation.enable_ha)
    # adapter_modules are handled individually - self-attn frozen, cross-attn trainable
    vae.eval()
    text_encoder.eval()
    image_encoder.eval()

    # Collect trainable cross-attention processors only
    trainable_cross_attn_params = []
    if ablation.enable_ha:
        for name, proc in unet.attn_processors.items():
            if isinstance(proc, RefCAttnProcessor2_0):
                # Explicitly enable gradients for cross-attention parameters
                proc.requires_grad_(True)
                trainable_cross_attn_params.extend(proc.parameters())

    unet_trainable_params = []
    if ablation.unet_train == "cross_attn":
        unet_trainable_params = _enable_unet_cross_attn_training(unet)

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())  # for saving only
    sd_model = SDModel(unet, ref_unet, image_proj, adapter_modules, ablation_config=ablation)
    if runtime_mode != build_mode:
        sd_model.enable_ieb = runtime_ablation.enable_ieb
        sd_model.enable_ha = runtime_ablation.enable_ha
        sd_model.direct_ieb = runtime_ablation.direct_ieb
        sd_model.ablation_config = runtime_ablation
        accelerator.print(
            "[AblationOverride] "
            f"build_mode={build_mode}, runtime_mode={runtime_mode} (no shape changes)"
        )

    mem_status = enable_memory_saving(
        unet=unet,
        ref_unet=ref_unet,
        vae=vae,
        low_vram=args.low_vram,
        enable_xformers=args.enable_xformers_memory_efficient_attention,
        gradient_checkpointing=args.gradient_checkpointing,
        sdpa_mode=args.sdpa_mode,
    )
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
    else:
        accelerator.print(
            "Memory backends: "
            f"grad_ckpt unet={mem_status.get('unet_grad_ckpt')}, ref_unet={mem_status.get('ref_unet_grad_ckpt')}; "
            f"xformers unet={mem_status.get('unet_xformers')}, ref_unet={mem_status.get('ref_unet_xformers')}; "
            f"sdpa flash={mem_status.get('flash_sdp')}, mem_efficient={mem_status.get('mem_efficient_sdp')}, "
            f"math={mem_status.get('math_sdp')}"
        )
    if ablation.enable_ha and args.gradient_checkpointing and mem_status.get("ref_unet_grad_ckpt") != "on":
        accelerator.print("[WARN] ref_unet gradient checkpointing requested but not enabled.")

    if args.eval_only:
        checkpoint_path = args.checkpoint or args.resume_from_checkpoint
        if not checkpoint_path:
            raise RuntimeError("--eval_only requires --checkpoint or --resume_from_checkpoint to be set.")

        noise_scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing", prediction_type="epsilon",
        )

        sd_model = accelerator.prepare(sd_model)
        weight_dtype = _resolve_weight_dtype(accelerator)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        image_encoder.to(accelerator.device, dtype=weight_dtype)

        torch.backends.cudnn.benchmark = False

        dataset = DressCodeDataset(
            root=args.dresscode_root,
            split=args.split,
            tokenizer=tokenizer,
            category=args.dresscode_category,
            test_order=args.dresscode_test_order,
            prompts_cache=args.prompts_cache,
            prompt_mode=args.prompt_mode,
            prompt_fallback=args.prompt_fallback,
            size=args.train_image_size,
            verify_files=True,
            strict_prompts_cache=args.strict_prompts_cache,
            enforce_cache_coverage=False,
        )

        # Build key->idx mapping for subset selection
        key_to_idx = {}
        for idx, item in enumerate(dataset.data):
            key = normalize_prompt_cache_key(
                make_pair_key(item["category"], item["image_file"], item["cloth_file"])
            )
            if key not in key_to_idx:
                key_to_idx[key] = idx

        if args.validation_subset_json:
            with open(args.validation_subset_json, "r", encoding="utf-8") as f:
                subset_items = json.load(f)
            if not isinstance(subset_items, list) or not subset_items:
                raise RuntimeError("[SubsetJSON] validation_subset_json is empty or invalid.")
            subset_pair_keys = []
            missing_subset = []
            for idx, item in enumerate(subset_items):
                if not isinstance(item, dict):
                    raise RuntimeError(f"[SubsetJSON] item {idx} is not a dict.")
                category = item.get("category")
                image_file = item.get("image_file")
                cloth_file = item.get("cloth_file")
                if not category or not image_file or not cloth_file:
                    raise RuntimeError(f"[SubsetJSON] Missing fields at index {idx}: {item}")
                pair_key = normalize_prompt_cache_key(make_pair_key(category, image_file, cloth_file))
                subset_pair_keys.append(pair_key)
                if pair_key not in key_to_idx:
                    missing_subset.append(pair_key)
            if missing_subset:
                show_count = min(10, len(missing_subset))
                error_msg = (
                    f"[SubsetMissing] {len(missing_subset)} pair keys not found in dataset.\n"
                    "First missing keys:\n"
                )
                for i in range(show_count):
                    error_msg += f"  - {missing_subset[i]}\n"
                if len(missing_subset) > show_count:
                    error_msg += f"  ... and {len(missing_subset) - show_count} more\n"
                raise RuntimeError(error_msg)
            val_indices = [key_to_idx[key] for key in subset_pair_keys]
        else:
            val_size = min(args.validation_samples, len(dataset))
            val_generator = torch.Generator().manual_seed(42)
            val_indices = torch.randperm(len(dataset), generator=val_generator)[:val_size].tolist()
            subset_items = [dataset.data[idx] for idx in val_indices]
            subset_pair_keys = [
                normalize_prompt_cache_key(
                    make_pair_key(item["category"], item["image_file"], item["cloth_file"])
                )
                for item in subset_items
            ]

        val_size = len(val_indices)
        if val_size == 0:
            raise RuntimeError("[EvalOnly] No validation samples selected.")

        # Map fallback cache keys (image|||cloth) to full keys for eval-only
        fallback_injected = 0
        for item in subset_items:
            key = normalize_prompt_cache_key(
                make_pair_key(item["category"], item["image_file"], item["cloth_file"])
            )
            fallback_key = f"{Path(item['image_file']).name}|||{Path(item['cloth_file']).name}"
            if key not in dataset.prompt_cache and fallback_key in dataset.prompt_cache:
                dataset.prompt_cache[key] = dataset.prompt_cache[fallback_key]
                fallback_injected += 1

        missing_keys = []
        cached_pairs = 0
        primary_hits = 0
        fallback_hits = 0
        for item in subset_items:
            key = normalize_prompt_cache_key(
                make_pair_key(item["category"], item["image_file"], item["cloth_file"])
            )
            fallback_key = f"{Path(item['image_file']).name}|||{Path(item['cloth_file']).name}"
            if key in dataset.prompt_cache:
                primary_hits += 1
                cached_pairs += 1
            elif fallback_key in dataset.prompt_cache:
                fallback_hits += 1
                cached_pairs += 1
            else:
                missing_keys.append(key)
        coverage = (cached_pairs / len(subset_pair_keys)) * 100.0 if subset_pair_keys else 0.0
        if args.prompt_mode == "fallback_only":
            prompt_cache_hits = 0
            fallback_count = len(subset_pair_keys)
        elif args.prompt_mode == "cache_only":
            prompt_cache_hits = cached_pairs
            fallback_count = len(missing_keys)
        else:
            prompt_cache_hits = cached_pairs
            fallback_count = len(missing_keys)
        accelerator.print(
            f"[PromptCache] Coverage (subset): {coverage:.1f}% ({cached_pairs}/{len(subset_pair_keys)}); "
            "cache hit/miss counters reset."
        )
        accelerator.print(
            f"[PromptCache] key_format=category|||image|||cloth, fallback_key_used={fallback_hits}, "
            f"fallback_injected={fallback_injected}"
        )
        dataset.cache_hit_count = 0
        dataset.cache_miss_count = 0

        if args.prompt_mode == "cache_only" and coverage < args.min_cache_coverage:
            show_count = min(20, len(missing_keys))
            error_msg = (
                "[PromptCacheIncomplete] P4(cache_only) requires 100% cache hit on the eval subset.\n"
                f"  Coverage (subset): {coverage:.1f}% ({cached_pairs}/{len(subset_pair_keys)})\n"
                f"  Required: {args.min_cache_coverage:.1f}% (--min_cache_coverage)\n"
                f"  Missing prompts: {len(missing_keys)}\n"
            )
            if show_count > 0:
                error_msg += f"\nFirst {show_count} missing cache keys:\n"
                for i in range(show_count):
                    error_msg += f"  - {missing_keys[i]}\n"
                if len(missing_keys) > show_count:
                    error_msg += f"  ... and {len(missing_keys) - show_count} more\n"
            error_msg += (
                "\nRegenerate prompts with:\n"
                f"  {_build_prompt_regen_command(args)}\n"
            )
            raise RuntimeError(error_msg)

        accelerator.print(
            f"[Dataset] DressCode root={args.dresscode_root}, "
            f"category={args.dresscode_category}, split={args.split}, size={len(dataset)}"
        )
        sample = dataset[val_indices[0]]
        required_keys = {
            "vae_person",
            "vae_clothes",
            "clip_image",
            "drop_image_embed",
            "text",
            "text_input_ids",
            "null_text_input_ids",
            "image_file",
            "cloth_file",
        }
        missing_keys = required_keys.difference(sample.keys())
        if missing_keys:
            raise RuntimeError(f"[Dataset] Missing keys in sample: {sorted(missing_keys)}")
        prompt_preview = sample["text"].replace("\n", " ").replace("\t", " ").strip()
        if len(prompt_preview) > 80:
            prompt_preview = prompt_preview[:80]
        accelerator.print(
            f"[Dataset] sample image_file={sample['image_file']} cloth_file={sample['cloth_file']} "
            f"category={sample['category']} prompt='{prompt_preview}'"
        )
        dataset.cache_hit_count = 0
        dataset.cache_miss_count = 0

        val_subset = torch.utils.data.Subset(dataset, val_indices)

        val_loader_kwargs = _build_dataloader_kwargs(
            args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            persistent_workers=args.dataloader_persistent_workers,
            is_windows=(os.name == "nt"),
        )
        try:
            val_dataloader = torch.utils.data.DataLoader(
                val_subset, collate_fn=collate_fn, batch_size=1, shuffle=False,
                **val_loader_kwargs
            )
        except TypeError as e:
            if "multiprocessing_context" in str(e):
                val_loader_kwargs.pop("multiprocessing_context", None)
                print(f"[WARN] DataLoader does not accept multiprocessing_context: {e}")
                val_dataloader = torch.utils.data.DataLoader(
                    val_subset, collate_fn=collate_fn, batch_size=1, shuffle=False,
                    **val_loader_kwargs
                )
            else:
                raise

        accelerator.print(f"✓ Validation dataset: {val_size} samples")

        ckpt = _load_checkpoint_safely(checkpoint_path, accelerator, sd_model)
        global_steps = int(ckpt.get("global_steps", 0)) if ckpt else 0

        save_pairs = bool(args.save_val_pairs or args.compute_visual_metrics)
        val_loss, eval_save_skipped = run_eval_only_with_saving(
            sd_model, val_dataloader, vae, text_encoder, image_encoder,
            noise_scheduler, accelerator, weight_dtype,
            output_dir=args.output_dir,
            save_pairs=save_pairs,
            pair_items=subset_items,
            eval_denoise_steps=args.eval_denoise_steps,
            eval_cleanup_every=args.eval_cleanup_every,
            eval_skip_existing_pairs=args.eval_skip_existing_pairs,
            eval_oom_fallback_steps=args.eval_oom_fallback_steps,
        )
        metrics_spec = ablation_spec
        eval_errors = []
        if eval_save_skipped > 0:
            eval_errors.append(f"eval-only image saving skipped {eval_save_skipped} samples")
        metrics = {
            "ablation_s": args.ablation_s or "",
            "ablation_p": args.ablation_p or "",
            "ablation_spec": metrics_spec,
            "ablation_mode_resolved": effective_ablation_mode,
            "prompt_mode_resolved": args.prompt_mode,
            "min_cache_coverage": args.min_cache_coverage,
            "strict_prompts_cache": bool(args.strict_prompts_cache),
            "checkpoint_path": checkpoint_path,
            "eval_denoise_steps": int(args.eval_denoise_steps),
            "eval_cleanup_every": int(args.eval_cleanup_every),
            "eval_skip_existing_pairs": bool(args.eval_skip_existing_pairs),
            "eval_oom_fallback_steps": int(args.eval_oom_fallback_steps),
            "val_loss": float(val_loss),
            "save_skipped_count": int(eval_save_skipped),
            "prompt_cache_hits": prompt_cache_hits,
            "prompt_cache_misses": fallback_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if accelerator.is_main_process:
            metrics_path = os.path.join(args.output_dir, "val_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=True)
            accelerator.print(f"Eval-only metrics saved to {metrics_path}")
        accelerator.wait_for_everyone()

        if args.compute_visual_metrics:
            if accelerator.is_main_process:
                import traceback
                fid_backend = "standard"
                kid_backend = "standard"
                pixel = None
                pixel_error = None
                fid_value = None
                kid_mean = None
                kid_std = None
                fid_error = None
                kid_error = None
                metrics_error = None
                try:
                    _, gt_dir, gen_dir = _resolve_visual_metrics_dirs(args.output_dir)
                    if not gt_dir or not gen_dir:
                        raise RuntimeError("visual_metrics directories not resolved.")
                    if not (os.path.isdir(gt_dir) and os.path.isdir(gen_dir)):
                        legacy_gt = os.path.join(args.output_dir, "eval_images", "real")
                        legacy_gen = os.path.join(args.output_dir, "eval_images", "fake")
                        if os.path.isdir(legacy_gt) and os.path.isdir(legacy_gen):
                            accelerator.print("[WARN] visual_metrics dirs missing; using legacy eval_images dirs.")
                            gt_dir = legacy_gt
                            gen_dir = legacy_gen
                        else:
                            raise RuntimeError(f"visual_metrics dirs missing: {gt_dir} / {gen_dir}")
                    try:
                        pixel = _calculate_pixel_metrics(gt_dir, gen_dir, accelerator.device)
                    except Exception as exc:
                        pixel_error = f"{type(exc).__name__}: {exc}"
                        traceback.print_exc()
                        accelerator.print(f"[WARN] Pixel metrics failed: {pixel_error}")

                    if args.fid_backend != "standard" or args.kid_backend != "standard":
                        accelerator.print(
                            "[WARN] FID/KID backend forced to standard (calculate_fid_kid_standard.py)."
                        )
                    try:
                        fid_value, kid_mean, kid_std = _compute_fid_kid_standard(gt_dir, gen_dir)
                    except Exception as exc:
                        err = f"{type(exc).__name__}: {exc}"
                        fid_error = err
                        kid_error = err
                        traceback.print_exc()
                        accelerator.print(f"[WARN] FID/KID failed (standard): {err}")
                except Exception as exc:
                    metrics_error = f"{type(exc).__name__}: {exc}"
                    fid_error = fid_error or metrics_error
                    kid_error = kid_error or metrics_error
                    traceback.print_exc()
                    accelerator.print(f"[WARN] Visual metrics failed: {metrics_error}")

                if fid_value is None:
                    fid_value = -1.0
                if kid_mean is None:
                    kid_mean = -1.0
                if kid_std is None:
                    kid_std = -1.0
                if args.strict_visual_metrics and (fid_error is not None or kid_error is not None):
                    eval_errors.append("strict_visual_metrics enabled: FID/KID missing")
                metrics_out_path = os.path.join(args.output_dir, args.metrics_out)
                pixel_skipped = None if not pixel else pixel.get("skipped_count")
                resize_count = None if not pixel else pixel.get("resize_count")
                combined_skipped = eval_save_skipped
                if pixel_skipped is not None:
                    combined_skipped = max(eval_save_skipped, pixel_skipped)
                eval_metrics = {
                    "psnr_mean": None if not pixel else pixel.get("psnr_mean"),
                    "ssim_mean": None if not pixel else pixel.get("ssim_mean"),
                    "lpips_mean": None if not pixel else pixel.get("lpips_mean"),
                    "fid": float(fid_value),
                    "kid": float(kid_mean),
                    "kid_std": float(kid_std),
                    "fid_backend": fid_backend,
                    "kid_backend": kid_backend,
                    "fid_error": fid_error,
                    "kid_error": kid_error,
                    "pixel_error": pixel_error,
                    "metrics_error": metrics_error,
                    "prompt_cache_hit": prompt_cache_hits,
                    "fallback_count": fallback_count,
                    "num_pairs": len(subset_pair_keys),
                    "empty_count": 0,
                    "resize_count": resize_count,
                    "skipped_count": combined_skipped,
                    "save_skipped_count": int(eval_save_skipped),
                    "validation_subset_json": args.validation_subset_json or "",
                    "ablation_s": args.ablation_s or "",
                    "ablation_p": args.ablation_p or "",
                    "ablation_spec": metrics_spec,
                }
                with open(metrics_out_path, "w", encoding="utf-8") as f:
                    json.dump(eval_metrics, f, indent=2, ensure_ascii=True)
                accelerator.print(f"Eval metrics saved to {metrics_out_path}")
        accelerator.wait_for_everyone()
        if eval_errors and accelerator.is_main_process:
            accelerator.print("[WARN] Eval-only completed with warnings:")
            for msg in eval_errors:
                accelerator.print(f"  - {msg}")
        if args.strict_visual_metrics:
            if eval_errors:
                raise RuntimeError("Eval-only strict_visual_metrics failed:\n" + "\n".join(eval_errors))
        return

    # Collect parameters for optimization based on ablation mode
    params_to_opt_groups = []
    if ablation.enable_ieb:
        params_to_opt_groups.append(sd_model.proj.parameters())
    if ablation.enable_ha:
        params_to_opt_groups.append(sd_model.ref_unet.parameters())
        if trainable_cross_attn_params:
            params_to_opt_groups.append(trainable_cross_attn_params)
    if unet_trainable_params:
        params_to_opt_groups.append(unet_trainable_params)
    if not params_to_opt_groups:
        raise RuntimeError(f"No trainable parameters for ablation_mode={ablation.mode}")

    params_to_opt = itertools.chain(*params_to_opt_groups)

    # Count trainable parameters per group (for logging)
    cross_attn_param_count = sum(p.numel() for p in trainable_cross_attn_params) / 1e6
    unet_train_param_count = sum(p.numel() for p in unet_trainable_params) / 1e6

    if ablation.enable_ha:
        accelerator.print("Trainable parameters: proj:{:.2f}M, ref_unet:{:.2f}M, cross_attn:{:.2f}M".format(
            count_model_params(sd_model.proj), count_model_params(sd_model.ref_unet),
            cross_attn_param_count))
    elif ablation.direct_ieb:
        accelerator.print("Trainable parameters: proj:{:.2f}M".format(count_model_params(sd_model.proj)))
    else:
        accelerator.print("Trainable parameters: unet_cross_attn:{:.2f}M".format(unet_train_param_count))
    total_trainable = 0
    total_params = 0
    for name, module in [
        ("unet", unet),
        ("ref_unet", ref_unet),
        ("image_proj", image_proj),
        ("image_encoder", image_encoder),
        ("text_encoder", text_encoder),
        ("vae", vae),
    ]:
        trainable = count_model_params_raw(module, trainable_only=True)
        total = count_model_params_raw(module)
        total_trainable += trainable
        total_params += total
        accelerator.print(f"  {name}: trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M")
    accelerator.print(
        f"Trainable params (all modules): {total_trainable/1e6:.2f}M / {total_params/1e6:.2f}M"
    )
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

    # Load DressCode dataset
    dataset = DressCodeDataset(
        root=args.dresscode_root,
        split=args.split,
        tokenizer=tokenizer,
        category=args.dresscode_category,
        test_order=args.dresscode_test_order,
        prompts_cache=args.prompts_cache,
        prompt_mode=args.prompt_mode,
        prompt_fallback=args.prompt_fallback,
        size=args.train_image_size,
        verify_files=True,  # STRICT: Verify all files exist before training
        strict_prompts_cache=args.strict_prompts_cache,
        enforce_cache_coverage=False,
    )

    total_pairs = len(dataset)
    missing_keys = dataset._get_missing_cache_keys() if total_pairs > 0 else []
    cached_pairs = total_pairs - len(missing_keys)
    coverage = (cached_pairs / total_pairs) * 100.0 if total_pairs > 0 else 0.0
    accelerator.print(
        f"[PromptCache] Coverage: {coverage:.1f}% ({cached_pairs}/{total_pairs}); "
        "cache hit/miss counters reset."
    )
    dataset.cache_hit_count = 0
    dataset.cache_miss_count = 0

    if args.prompt_mode == "cache_only" and coverage < args.min_cache_coverage:
        show_count = min(20, len(missing_keys))
        error_msg = (
            "[PromptCacheIncomplete] prompt_mode=cache_only requires sufficient cache coverage.\n"
            f"  Coverage: {coverage:.1f}% ({cached_pairs}/{total_pairs})\n"
            f"  Required: {args.min_cache_coverage:.1f}% (--min_cache_coverage)\n"
            f"  Missing prompts: {len(missing_keys)}\n"
        )
        if show_count > 0:
            error_msg += f"\nFirst {show_count} missing cache keys:\n"
            for i in range(show_count):
                error_msg += f"  - {missing_keys[i]}\n"
            if len(missing_keys) > show_count:
                error_msg += f"  ... and {len(missing_keys) - show_count} more\n"
        error_msg += (
            "\nRegenerate prompts with:\n"
            f"  {_build_prompt_regen_command(args)}\n"
        )
        raise RuntimeError(error_msg)

    accelerator.print(
        f"[Dataset] DressCode root={args.dresscode_root}, "
        f"category={args.dresscode_category}, split={args.split}, size={len(dataset)}"
    )
    sample = dataset[0]
    required_keys = {
        "vae_person",
        "vae_clothes",
        "clip_image",
        "drop_image_embed",
        "text",
        "text_input_ids",
        "null_text_input_ids",
        "image_file",
        "cloth_file",
    }
    missing_keys = required_keys.difference(sample.keys())
    if missing_keys:
        raise RuntimeError(f"[Dataset] Missing keys in sample: {sorted(missing_keys)}")
    prompt_preview = sample["text"].replace("\n", " ").replace("\t", " ").strip()
    if len(prompt_preview) > 80:
        prompt_preview = prompt_preview[:80]
    accelerator.print(
        f"[Dataset] sample image_file={sample['image_file']} cloth_file={sample['cloth_file']} "
        f"category={sample['category']} prompt='{prompt_preview}'"
    )
    dataset.cache_hit_count = 0
    dataset.cache_miss_count = 0

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True
    )
    train_loader_kwargs = _build_dataloader_kwargs(
        args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        is_windows=(os.name == "nt"),
    )
    try:
        train_dataloader = torch.utils.data.DataLoader(
            dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.train_batch_size,
            **train_loader_kwargs
        )
    except TypeError as e:
        if "multiprocessing_context" in str(e):
            train_loader_kwargs.pop("multiprocessing_context", None)
            print(f"[WARN] DataLoader does not accept multiprocessing_context: {e}")
            train_dataloader = torch.utils.data.DataLoader(
                dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.train_batch_size,
                **train_loader_kwargs
            )
        else:
            raise

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

    weight_dtype = _resolve_weight_dtype(accelerator)
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
    mem_log_steps = args.cuda_mem_log_steps
    if args.mem_debug and mem_log_steps <= 0:
        mem_log_steps = 50

    best_val_loss = float('inf')
    abort_training = False
    abort_reason = ""

    # Create validation dataset (same configuration as training)
    val_dataset = DressCodeDataset(
        root=args.dresscode_root,
        split=args.split,
        tokenizer=tokenizer,
        category=args.dresscode_category,
        test_order=args.dresscode_test_order,
        prompts_cache=args.prompts_cache,
        prompt_mode=args.prompt_mode,
        prompt_fallback=args.prompt_fallback,
        size=args.train_image_size,
        verify_files=True,  # STRICT: Verify all files exist before validation
        strict_prompts_cache=args.strict_prompts_cache,
        enforce_cache_coverage=False,
    )

    # Use a subset for validation
    # FIX: Use fixed seed for validation set to ensure consistency across training runs
    # This allows best_val_loss comparison to be meaningful when resuming from checkpoint
    val_size = min(args.validation_samples, len(val_dataset))
    val_generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducible validation set
    val_indices = torch.randperm(len(val_dataset), generator=val_generator)[:val_size]
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    val_loader_kwargs = _build_dataloader_kwargs(
        args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        is_windows=(os.name == "nt"),
    )
    try:
        val_dataloader = torch.utils.data.DataLoader(
            val_subset, collate_fn=collate_fn, batch_size=1, shuffle=False,
            **val_loader_kwargs
        )
    except TypeError as e:
        if "multiprocessing_context" in str(e):
            val_loader_kwargs.pop("multiprocessing_context", None)
            print(f"[WARN] DataLoader does not accept multiprocessing_context: {e}")
            val_dataloader = torch.utils.data.DataLoader(
                val_subset, collate_fn=collate_fn, batch_size=1, shuffle=False,
                **val_loader_kwargs
            )
        else:
            raise

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
        ckpt = _load_checkpoint_safely(args.resume_from_checkpoint, accelerator, sd_model)

        # Optional: restore epoch and global_steps
        if ckpt:
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
            if encoder_hidden_states.dtype != weight_dtype:
                encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)

            # Apply image dropout (CFG for image embeddings)
            person_image_embeds = None
            if sd_model.enable_ieb:
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
                        with _get_autocast_context(accelerator.device, weight_dtype):
                            person_image_embeds = image_encoder(
                                batch["clip_image"].to(accelerator.device, dtype=weight_dtype),
                                output_hidden_states=True
                            ).hidden_states[-2]
                    if person_image_embeds.dtype != weight_dtype:
                        person_image_embeds = person_image_embeds.to(dtype=weight_dtype)
            else:
                apply_image_dropout = False

            # Compute target
            if noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(garment_latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")


            try:
                if noisy_garment_latents.dtype != weight_dtype:
                    noisy_garment_latents = noisy_garment_latents.to(dtype=weight_dtype)
                if person_latents.dtype != weight_dtype:
                    person_latents = person_latents.to(dtype=weight_dtype)

                # Forward pass: person latents -> garment latents
                with _get_autocast_context(accelerator.device, weight_dtype):
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
                    model_pred = None
                    loss = None
                    target = None
                    _clear_ref_unet_cache(sd_model.ref_unet)
                    _clear_unet_extra_state(sd_model.unet)
                    _safe_cuda_empty_cache()
                    if args.mem_debug:
                        gc.collect()
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
                _clear_unet_extra_state(sd_model.unet)

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
                    if sd_model.enable_ha:
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

                        if not ref_unet_has_grad or not proj_has_grad or not cross_attn_has_grad:
                            logger.warning("⚠ WARNING: Expected gradients missing! Check model setup.")
                        if self_attn_has_grad:
                            logger.warning("⚠ WARNING: Self-attention has gradients but should be frozen!")
                    elif sd_model.direct_ieb:
                        proj_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                           for p in sd_model.proj.parameters() if p.requires_grad)
                        logger.info(f"✓ image_proj has gradients: {proj_has_grad}")
                        unet_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                           for p in sd_model.unet.parameters() if p.requires_grad)
                        logger.info(f"✓ Main UNet has gradients: {unet_has_grad} (should be False)")
                        if not proj_has_grad:
                            logger.warning("⚠ WARNING: Expected gradients missing! Check model setup.")
                    else:
                        unet_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                           for p in sd_model.unet.parameters() if p.requires_grad)
                        logger.info(f"✓ Main UNet trainable gradients: {unet_has_grad}")
                        if not unet_has_grad:
                            logger.warning("⚠ WARNING: Expected gradients missing! Check model setup.")

                    logger.info("=" * 60)

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
                            "target",
                            "encoder_hidden_states",
                            "person_image_embeds",
                            "timesteps",
                            "batch",
                        ]:
                            if name in locals():
                                del locals()[name]
                        cuda_memory_cleanup(tag="periodic", deep=False, do_ipc=args.cuda_ipc_collect)

                    if accelerator.sync_gradients:
                        accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps}, step=global_steps)
                        train_loss = 0.0

                # Release large tensors and clear any lingering cache refs
                for name in [
                    "batch",
                    "person_latents",
                    "garment_latents",
                    "noisy_garment_latents",
                    "noise",
                    "model_pred",
                    "loss",
                    "target",
                    "encoder_hidden_states",
                    "person_image_embeds",
                    "timesteps",
                ]:
                    if name in locals():
                        del locals()[name]
                _clear_ref_unet_cache(sd_model.ref_unet)
                _clear_unet_extra_state(sd_model.unet)
                _safe_cuda_empty_cache()
                if args.mem_debug:
                    gc.collect()
            except RuntimeError as e:
                err_msg = str(e)
                if "out of memory" in err_msg.lower() or "cuda error" in err_msg.lower():
                    if accelerator.is_main_process:
                        logger.error(f"[CUDA] RuntimeError at step {global_steps}: {err_msg}")
                    cuda_memory_cleanup(tag="exception", deep=True, do_ipc=args.cuda_ipc_collect)
                    optimizer.zero_grad(set_to_none=True)
                    for name in [
                        "batch",
                        "person_latents",
                        "garment_latents",
                        "noisy_garment_latents",
                        "noise",
                        "model_pred",
                        "loss",
                        "target",
                        "encoder_hidden_states",
                        "person_image_embeds",
                        "timesteps",
                    ]:
                        if name in locals():
                            del locals()[name]
                    _clear_ref_unet_cache(sd_model.ref_unet)
                    _clear_unet_extra_state(sd_model.unet)
                    _safe_cuda_empty_cache()
                    if args.vram_soft_cap_gb and args.vram_soft_cap_gb > 0:
                        abort_training = True
                        abort_reason = f"CUDA OOM under vram_soft_cap_gb={args.vram_soft_cap_gb}"
                        break
                    global_steps += 1
                    step += 1
                    begin = time.perf_counter()
                    continue
                raise

            if accelerator.is_main_process and torch.cuda.is_available():
                if args.mem_debug and mem_log_steps > 0:
                    if global_steps % mem_log_steps == 0:
                        _log_cuda_memory_stats(logger, "mem_debug", global_steps)
                elif args.cuda_mem_log_steps > 0 and global_steps % args.cuda_mem_log_steps == 0:
                    _log_cuda_memory_stats(logger, "periodic", global_steps)

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
                    save_model_checkpoint(
                        args.output_dir,
                        global_steps,
                        sd_model,
                        epoch,
                        accelerator,
                        ema_model=ema_model,
                        prompt_config=prompt_config,
                    )
                    cuda_memory_cleanup(tag="post_checkpoint", deep=False, do_ipc=args.cuda_ipc_collect)

                # Milestone checkpoint saving
                if should_save_milestone(global_steps) and global_steps != resumed_step:
                    save_model_checkpoint(
                        args.output_dir,
                        global_steps,
                        sd_model,
                        epoch,
                        accelerator,
                        ema_model=ema_model,
                        prompt_config=prompt_config,
                        checkpoint_type="milestone",
                    )
                    cuda_memory_cleanup(tag="post_checkpoint", deep=False, do_ipc=args.cuda_ipc_collect)

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
                        save_model_checkpoint(
                            args.output_dir,
                            global_steps,
                            sd_model,
                            epoch,
                            accelerator,
                            ema_model=ema_model,
                            prompt_config=prompt_config,
                            checkpoint_type="best",
                            loss_value=val_loss,
                        )
                        cuda_memory_cleanup(tag="post_checkpoint", deep=False, do_ipc=args.cuda_ipc_collect)
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
        if accelerator.is_main_process:
            _log_prompt_cache_stats(logger, dataset, "train")
        if abort_training:
            break
        if global_steps >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    # Save last model
    if accelerator.is_main_process:
        save_model_checkpoint(
            args.output_dir,
            global_steps,
            sd_model,
            epoch,
            accelerator,
            ema_model=ema_model,
            prompt_config=prompt_config,
            checkpoint_type="last",
        )
        cuda_memory_cleanup(tag="post_checkpoint", deep=False, do_ipc=args.cuda_ipc_collect)

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
