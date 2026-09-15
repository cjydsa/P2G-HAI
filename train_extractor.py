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
from IGPair import VDDataset, collate_fn
from adapter.attention_processor import CacheAttnProcessor2_0, RefCAttnProcessor2_0, RefSAttnProcessor2_0

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--pretrained_adapter_model_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default=None,
        help="Path to dataset json file.",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        default="",
        help="Root path for images (if JSON contains relative paths).",
    )

    parser.add_argument(
        "--region_filter",
        type=str,
        default="upper",
        choices=["upper", "lower", "all"],
        help="Which garment region to use: upper-body only, lower-body only, or all garments.",
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
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0.05, help="noise_offset."
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=0, help="noise_offset."
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

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def save_model_checkpoint(output_dir, global_steps, sd_model, epoch,
                          accelerator,
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


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
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

    # Enable gradient checkpointing for memory efficiency (24GB GPU friendly)
    unet.enable_gradient_checkpointing()
    ref_unet.enable_gradient_checkpointing()

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

    dataset = VDDataset(
        [
            args.dataset_json_path,
        ],
        tokenizer,
        image_root_path=args.image_root_path,
        region_filter=args.region_filter,
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

    accelerator.print(f"Using region_filter = {args.region_filter}")

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_steps = 0
    starting_epoch = 0
    resumed_step = -1  # Track the step we resumed from to avoid re-saving

    best_val_loss = float('inf')

    # Create validation dataset (subset for efficiency)
    val_dataset = VDDataset(
        [args.dataset_json_path],
        tokenizer,
        image_root_path=args.image_root_path,
        region_filter=args.region_filter,
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
            person_path = os.path.join(val_dataset.image_root_path, item["image_file"]) if val_dataset.image_root_path else item["image_file"]
            cloth_path = os.path.join(val_dataset.image_root_path, item["cloth_file"]) if val_dataset.image_root_path else item["cloth_file"]
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

            # Get CLIP embeddings from PERSON image (not garment!)
            # FIX: Only dropout CLIP features, KEEP person_latents for ref_unet to extract features
            # Reason: When person_latents is zeroed, ref_unet outputs near-zero features,
            #         causing the model to learn "no person info → generate black"
            clip_person_images = []
            person_latents_masked = []
            for idx, (clip_image, drop_image_embed) in enumerate(zip(batch["clip_image"], batch["drop_image_embed"])):
                if drop_image_embed == 1:
                    # Dropout: zero out CLIP features only (for CFG training)
                    clip_person_images.append(torch.zeros_like(clip_image))
                    person_latents_masked.append(person_latents[idx:idx+1])  # KEEP person_latents!
                else:
                    # Use person image for CLIP encoding
                    clip_person_images.append(clip_image)
                    person_latents_masked.append(person_latents[idx:idx+1])
            clip_person_images = torch.stack(clip_person_images, dim=0)
            person_latents_masked = torch.cat(person_latents_masked, dim=0)

            with torch.no_grad():
                # Extract CLIP features from PERSON image
                person_image_embeds = image_encoder(
                    clip_person_images.to(accelerator.device, dtype=weight_dtype),
                    output_hidden_states=True
                ).hidden_states[-2]

                # Immediately delete clip_person_images to free 224x224x3 batch tensor memory
                del clip_person_images

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

            # Compute target
            if noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(garment_latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

            # Forward pass: person latents → garment latents
            # Use masked person_latents instead of original to ensure proper CFG training
            model_pred = sd_model(
                encoder_hidden_states,
                noisy_garment_latents,  # target: garment
                person_latents_masked,   # reference: person (WITH DROPOUT APPLIED)
                person_image_embeds,     # CLIP from person
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
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, sd_model.parameters()), max_norm=1.0)

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
                    logger.info(f"person_latents_masked stats: min={person_latents_masked.min():.4f}, max={person_latents_masked.max():.4f}, mean={person_latents_masked.mean():.4f}")
                    logger.info(f"model_pred stats: min={model_pred.min():.4f}, max={model_pred.max():.4f}, mean={model_pred.mean():.4f}")
                    logger.info(f"target stats: min={target.min():.4f}, max={target.max():.4f}, mean={target.mean():.4f}")
                    logger.info(f"dropout rate in batch: {sum(batch['drop_image_embed'])/len(batch['drop_image_embed'])*100:.1f}%")
                    logger.info("=" * 60)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # Free gradient memory immediately

                # Clear cache after optimizer step to reduce fragmentation
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                # Proactively clean up every 50 steps to prevent fragmentation (more frequent for stability)
                if global_steps % 50 == 0:
                    # Synchronize GPU to ensure all operations complete
                    torch.cuda.synchronize()
                    # Delete large tensors that might still be in scope
                    if 'person_latents' in locals():
                        del person_latents, garment_latents, noisy_garment_latents, noise, model_pred, loss
                    torch.cuda.empty_cache()
                    gc.collect()

                if accelerator.sync_gradients:
                    accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps}, step=global_steps)
                    train_loss = 0.0

            # Deep memory cleanup every 1000 steps to prevent Windows WDDM fragmentation
            if global_steps % 1000 == 0 and global_steps > 0:
                if accelerator.is_main_process:
                    logger.info(f"🧹 Deep memory cleanup at step {global_steps}")
                import gc
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()

            # Checkpoint strategy
            if accelerator.is_main_process and global_steps > 0:
                # Regular checkpoint saving
                # Skip saving if this is exactly the resumed step (to avoid overwriting the checkpoint we just loaded)
                if should_save_checkpoint(global_steps) and global_steps != resumed_step:
                    save_model_checkpoint(args.output_dir, global_steps, sd_model, epoch, accelerator)

                # Milestone checkpoint saving
                if should_save_milestone(global_steps) and global_steps != resumed_step:
                    save_model_checkpoint(args.output_dir, global_steps, sd_model, epoch, accelerator, "milestone")

                # Validation and best model saving
                if global_steps % args.validation_steps == 0:
                    logger.info("=" * 60)
                    logger.info("RUNNING VALIDATION")
                    logger.info("=" * 60)

                    # Force memory cleanup before validation to prevent OOM
                    import gc
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()

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
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()

                    logger.info(f"✓ Validation Loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_model_checkpoint(args.output_dir, global_steps, sd_model, epoch, accelerator, "best", val_loss)
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
        save_model_checkpoint(args.output_dir, global_steps, sd_model, epoch, accelerator, "last")

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