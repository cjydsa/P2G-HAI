#!/usr/bin/env python3
"""
评估 Extractor 模型的生成质量
使用测试集JSON，生成garment图片，并与Ground Truth计算FID和KID指标
"""

import os
import sys
import json
import argparse
import random
import gc
import contextlib
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from adapter.resampler import Resampler
from adapter.attention_processor import CacheAttnProcessor2_0, RefCAttnProcessor2_0, RefSAttnProcessor2_0
from torchvision import transforms
try:
    from calculate_fid_kid_standard import (
        InceptionV3FeatureExtractor,
        calculate_fid_standard,
        calculate_kid_standard,
    )
    FID_KID_AVAILABLE = True
    FID_KID_IMPORT_ERROR = None
except Exception as exc:
    InceptionV3FeatureExtractor = None
    calculate_fid_standard = None
    calculate_kid_standard = None
    FID_KID_AVAILABLE = False
    FID_KID_IMPORT_ERROR = exc
try:
    from cleanfid import fid as cleanfid_fid
    CLEANFID_AVAILABLE = True
    CLEANFID_IMPORT_ERROR = None
except Exception as exc:
    cleanfid_fid = None
    CLEANFID_AVAILABLE = False
    CLEANFID_IMPORT_ERROR = exc
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

CUDA_POISONED = False


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("vae_deterministic must be True/False")
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y", "t"):
        return True
    if normalized in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("vae_deterministic must be True/False")


def _log_xformers_status():
    try:
        import xformers  # noqa: F401
    except Exception as exc:
        print(f"[WARN] xformers not available; using PyTorch attention. ({exc})")
        return
    print("[INFO] xformers available; custom attention processors are active (PyTorch attention).")


def safe_cuda_cleanup(tag: str):
    accel_error = getattr(torch, "AcceleratorError", None)
    error_types = (Exception, accel_error) if accel_error is not None else (Exception,)

    if CUDA_POISONED:
        try:
            gc.collect()
        except error_types as exc:
            print(f"[WARN] GC collect failed ({tag}): {exc}")
        return

    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except error_types as exc:
                print(f"[WARN] CUDA synchronize failed ({tag}): {exc}")
            try:
                torch.cuda.empty_cache()
            except error_types as exc:
                print(f"[WARN] CUDA empty_cache failed ({tag}): {exc}")
    except error_types as exc:
        print(f"[WARN] CUDA cleanup failed ({tag}): {exc}")

    try:
        gc.collect()
    except error_types as exc:
        print(f"[WARN] GC collect failed ({tag}): {exc}")


def _is_cuda_failure(exc: Exception) -> bool:
    message = str(exc)
    if not message:
        return False
    lowered = message.lower()
    return (
        "cudnn_status_execution_failed" in lowered
        or "cuda error" in lowered
        or "device-side assert" in lowered
    )


def _coerce_prompt_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return None
        for item in value:
            if isinstance(item, str):
                return item
        return None
    if isinstance(value, dict):
        for field in ("prompt", "text", "caption", "prompts"):
            if field in value:
                return _coerce_prompt_text(value[field])
    return None


def _extract_cache_key(entry):
    if not isinstance(entry, dict):
        return None
    for field in ("key", "pair_key", "id"):
        key = entry.get(field)
        if key is not None:
            return str(key)
    return None


def _extract_prompt_from_entry(entry):
    if not isinstance(entry, dict):
        return None
    for field in ("prompt", "text", "caption", "prompts"):
        if field in entry:
            return _coerce_prompt_text(entry[field])
    return None


def _cache_key_candidates(image_file, cloth_file, cache_key_mode):
    image_file = str(image_file or "")
    cloth_file = str(cloth_file or "")
    base_image = os.path.basename(image_file)
    base_cloth = os.path.basename(cloth_file)
    if cache_key_mode == "basename_pair":
        return [f"{base_image}|||{base_cloth}"]
    if cache_key_mode == "path_pair":
        return [f"{image_file}|||{cloth_file}"]
    if cache_key_mode == "auto":
        base_key = f"{base_image}|||{base_image}"
        return [
            f"{base_image}|||{base_cloth}",
            f"{image_file}|||{cloth_file}",
            base_key,
        ]
    return [f"{base_image}|||{base_cloth}"]


def _load_prompts_cache(cache_path):
    cache = {}
    if not cache_path:
        return cache
    if not os.path.exists(cache_path):
        print(f"[WARN] prompts_cache not found: {cache_path}")
        return cache
    ext = os.path.splitext(cache_path)[1].lower()
    if ext == ".json":
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key, value in data.items():
                prompt = _coerce_prompt_text(value)
                if prompt is not None:
                    cache[str(key)] = prompt
        elif isinstance(data, list):
            for entry in data:
                key = _extract_cache_key(entry)
                prompt = _extract_prompt_from_entry(entry)
                if key is not None and prompt is not None:
                    cache[str(key)] = prompt
        else:
            print("[WARN] Unsupported JSON cache structure")
    elif ext == ".jsonl":
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = _extract_cache_key(obj)
                prompt = _extract_prompt_from_entry(obj)
                if key is not None and prompt is not None:
                    cache[str(key)] = prompt
    else:
        raise ValueError(f"Unsupported cache format: {ext} (expected .json or .jsonl)")
    return cache


def _apply_prompt_cache(
    test_data,
    prompts_cache,
    prompt_fallback,
    log_first_n_prompts,
    strict_empty_prompt,
    allow_zero_prompt_cache_hit,
    cache_requested,
    cache_key_mode,
):
    prompt_cache_hit = 0
    fallback_count = 0
    empty_count = 0
    logs = []

    for idx, item in enumerate(test_data):
        prompt = None
        source = None
        key_used = None

        if cache_requested:
            image_file = item.get("image_file", "")
            cloth_file = item.get("cloth_file", "")
            for key in _cache_key_candidates(image_file, cloth_file, cache_key_mode):
                if key in prompts_cache:
                    prompt = prompts_cache[key]
                    source = "cache"
                    key_used = key
                    break

            if source == "cache":
                prompt_cache_hit += 1
            else:
                fallback_count += 1
                prompt = prompt_fallback or ""
                if prompt == "":
                    empty_count += 1
                    source = "empty"
                else:
                    source = "fallback"
        else:
            prompt = item.get("text", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            if prompt is None:
                prompt = ""
            if prompt == "":
                fallback_count += 1
                prompt = prompt_fallback or ""
                if prompt == "":
                    empty_count += 1
                    source = "empty"
                else:
                    source = "fallback"
            else:
                source = "existing"

        item["item_text"] = prompt

        if log_first_n_prompts and len(logs) < log_first_n_prompts:
            logs.append((idx, source, key_used, prompt))

    print(f"[PromptCache] hit={prompt_cache_hit} fallback={fallback_count} empty={empty_count}")
    if log_first_n_prompts:
        print(f"[PromptCache] first {len(logs)} prompts:")
        for idx, source, key_used, prompt in logs:
            key_msg = key_used if key_used else "-"
            print(f"  [{idx}] source={source} key={key_msg} prompt={prompt}")

    if cache_requested:
        if strict_empty_prompt and empty_count > 0:
            raise RuntimeError(
                "Empty prompt encountered with strict mode enabled. "
                "Please generate the test prompts cache."
            )

        if prompt_cache_hit == 0 and not allow_zero_prompt_cache_hit:
            print("[WARN] Prompt cache hit is 0; using fallback prompts.")

    return prompt_cache_hit, fallback_count, empty_count

class ExtractorModel(torch.nn.Module):
    """Extractor model wrapper for garment extraction"""
    def __init__(self, unet, ref_unet, proj):
        super().__init__()
        self.unet = unet
        self.ref_unet = ref_unet
        self.proj = proj

    def forward(self, encoder_hidden_states, latents, ref_latents, clip_image_embeddings, timesteps):
        """
        Forward pass for garment extraction
        Args:
            encoder_hidden_states: Text embeddings
            latents: Noisy garment latents (to denoise)
            ref_latents: Person latents (reference input)
            clip_image_embeddings: CLIP features from person image
            timesteps: Diffusion timesteps
        """
        # Clear cache
        for proc in self.ref_unet.attn_processors.values():
            if hasattr(proc, "cache"):
                c = proc.cache
                if hasattr(c, "clear"):
                    c.clear()
                else:
                    for k in list(c.keys()):
                        c[k] = None

        ref_timesteps = torch.zeros_like(timesteps)
        person_proj_embed = self.proj(clip_image_embeddings)

        # Extract features from person image
        _ = self.ref_unet(
            ref_latents,
            ref_timesteps,
            encoder_hidden_states=person_proj_embed,
            return_dict=False,
        )

        # Get cached features
        sa_hidden_states = {}
        for name in self.ref_unet.attn_processors.keys():
            proc = self.ref_unet.attn_processors[name]
            if hasattr(proc, "cache") and "hidden_states" in proc.cache:
                sa_hidden_states[name] = proc.cache["hidden_states"]

        # Generate garment
        noise_pred = self.unet(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={"sa_hidden_states": sa_hidden_states}
        ).sample

        return noise_pred


def load_model(checkpoint_path, pretrained_model_path, image_encoder_path, vae_path, device):
    """加载 Extractor 模型"""
    print(f"\n{'='*60}")
    print("加载模型")
    print(f"{'='*60}")
    print(f"检查点: {checkpoint_path}")

    # 加载基础模型
    print("加载基础组件...")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(vae_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)

    # 创建 projection layer
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

    # 设置 attention processors
    print("设置attention processors...")
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

        if cross_attention_dim is None:
            attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ref.weight": st[layer_name + ".to_k.weight"],
                "to_v_ref.weight": st[layer_name + ".to_v.weight"],
            }
            attn_procs[name].load_state_dict(weights)
        else:
            attn_procs[name] = RefCAttnProcessor2_0(name, hidden_size=hidden_size,
                                                     cross_attention_dim=cross_attention_dim)
    unet.set_attn_processor(attn_procs)
    del st

    # 创建 ref_unet
    print("创建reference UNet...")
    ref_unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    ref_unet.set_attn_processor({name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})
    _log_xformers_status()

    # 加载训练的权重
    print(f"加载检查点权重...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    image_proj.load_state_dict(ckpt["image_proj"])
    ref_unet.load_state_dict(ckpt["ref_unet"])

    # 加载 cross-attention 权重
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, RefCAttnProcessor2_0) and name in ckpt["cross_attn_processors"]:
            proc.load_state_dict(ckpt["cross_attn_processors"][name])

    print(f"检查点信息:")
    print(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  - Global Steps: {ckpt.get('global_steps', 'N/A')}")
    if 'loss' in ckpt:
        print(f"  - Loss: {ckpt['loss']:.4f}")

    # 创建模型
    model = ExtractorModel(unet, ref_unet, image_proj)

    # 移动到设备
    print(f"\n移动模型到设备: {device}")
    model.to(device)
    vae.to(device)
    text_encoder.to(device)
    image_encoder.to(device)

    # 设置为评估模式
    model.eval()
    vae.eval()
    text_encoder.eval()
    image_encoder.eval()

    print("[OK] Model loaded successfully\n")

    return model, vae, text_encoder, image_encoder, tokenizer


def _build_scheduler(scheduler_name):
    ddim_scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    if scheduler_name == "ddim":
        return ddim_scheduler
    if scheduler_name == "unipc":
        return UniPCMultistepScheduler.from_config(ddim_scheduler.config)
    if scheduler_name == "dpmpp_2m_karras":
        return DPMSolverMultistepScheduler.from_config(
            ddim_scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _set_scheduler_timesteps(scheduler, num_inference_steps, device):
    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(num_inference_steps)


def _timesteps_to_list(timesteps):
    if timesteps is None:
        return []
    if isinstance(timesteps, torch.Tensor):
        ts_tensor = timesteps.detach().cpu()
    else:
        ts_tensor = torch.tensor(timesteps)
    if ts_tensor.ndim == 0:
        ts_tensor = ts_tensor.unsqueeze(0)
    ts_list = ts_tensor.flatten().tolist()
    return [int(x) for x in ts_list]


def _find_reconstruct_start_index(timesteps_list, reconstruct_t):
    if reconstruct_t is None or not timesteps_list:
        return 0
    descending = timesteps_list[0] > timesteps_list[-1]
    if descending:
        for i, t in enumerate(timesteps_list):
            if t <= reconstruct_t:
                return i
    else:
        for i, t in enumerate(timesteps_list):
            if t >= reconstruct_t:
                return i
    return 0


def _prepare_timesteps_list(
    scheduler,
    num_inference_steps,
    device,
    init_mode,
    reconstruct_t,
    min_effective_steps,
    label,
):
    _set_scheduler_timesteps(scheduler, num_inference_steps, device)
    full_timesteps = _timesteps_to_list(scheduler.timesteps)
    if not full_timesteps:
        raise ValueError("scheduler.timesteps is empty after set_timesteps")
    start_index = 0
    if init_mode == "from_noisy_gt":
        start_index = _find_reconstruct_start_index(full_timesteps, reconstruct_t)
    timesteps_list = full_timesteps[start_index:]
    if not timesteps_list:
        start_index = 0
        timesteps_list = full_timesteps
    if init_mode == "from_noisy_gt" and min_effective_steps is not None:
        min_steps = max(1, int(min_effective_steps))
        if len(timesteps_list) < min_steps:
            fallback_index = max(0, len(full_timesteps) - min_steps)
            if fallback_index < start_index:
                start_index = fallback_index
                timesteps_list = full_timesteps[start_index:]
            if len(timesteps_list) < min_steps:
                start_index = 0
                timesteps_list = full_timesteps
            if len(timesteps_list) < min_steps:
                print(
                    f"[WARN] {label} effective steps ({len(timesteps_list)}) "
                    f"< min_effective_steps={min_steps}; using available timesteps."
                )
            else:
                print(
                    f"[WARN] {label} effective steps too small -> "
                    f"fallback to earlier timesteps (start_index={start_index})"
                )
    return timesteps_list, start_index


def _normalize_timestep(timestep, latents=None, device=None, batch_size=None):
    if isinstance(timestep, torch.Tensor):
        t = timestep.detach()
    else:
        t = torch.tensor(timestep)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    if t.ndim != 1:
        t = t.flatten()
    target_device = device
    if target_device is None:
        if latents is not None:
            target_device = latents.device
        else:
            target_device = t.device
    t = t.to(device=target_device, dtype=torch.long)
    if batch_size is None:
        if latents is not None:
            batch_size = latents.shape[0]
        else:
            batch_size = t.shape[0] if t.numel() > 0 else 1
    if t.numel() == 0:
        return torch.zeros(batch_size, device=target_device, dtype=torch.long)
    if t.shape[0] == 1 and batch_size > 1:
        t = t.expand(batch_size)
    elif t.shape[0] != batch_size:
        if batch_size == 1:
            t = t[:1]
        else:
            t = t[:1].expand(batch_size)
    return t


def _is_timestep_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "timestep" in message
        or "timesteps" in message
        or "0-d tensor" in message
        or "0d tensor" in message
        or "iteration over a 0-d tensor" in message
    )


@torch.no_grad()
def generate_images(
    model,
    vae,
    text_encoder,
    image_encoder,
    tokenizer,
    test_data,
    image_root_path,
    cloth_root_path,
    output_dir,
    device,
    num_inference_steps=50,
    seed=42,
    guidance_scale=3.0,
    scheduler_name="ddim",
    eta=0.0,
    init_mode="from_noisy_gt",
    reconstruct_t=600,
    min_effective_steps=25,
    vae_deterministic=True,
    decode_fp32=True,
    refine_pass=False,
    refine_t=150,
    refine_steps=10,
):
    """
    生成测试图片

    输入：Person图片
    输出：生成的Garment图片
    Ground Truth: cloth_file对应的真实Garment图片
    """
    print(f"\n{'='*60}")
    print("生成图片")
    print(f"{'='*60}")
    print(f"测试样本数量: {len(test_data)}")
    print(f"采样步数: {num_inference_steps}")
    print(f"init_mode: {init_mode} | reconstruct_t: {reconstruct_t} | guidance_scale: {guidance_scale}")

    os.makedirs(output_dir, exist_ok=True)
    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 0.18215)

    # 准备数据转换（与VTONHD.py保持一致）
    vae_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB三通道
    ])

    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                           [0.26862954, 0.26130258, 0.27577711])
    ])

    # 设置 scheduler
    scheduler = _build_scheduler(scheduler_name)
    timesteps_list, start_index = _prepare_timesteps_list(
        scheduler,
        num_inference_steps,
        device,
        init_mode,
        reconstruct_t,
        min_effective_steps,
        label="main",
    )
    active_scheduler_name = scheduler_name
    fallback_used = False

    t_start = int(timesteps_list[0])
    scheduler_display = scheduler.__class__.__name__
    print(
        "[INFO] scheduler={name} ({cls}) num_inference_steps={steps} reconstruct_t={rt} "
        "t_start={t_start} real_timesteps_count={count} guidance_scale={gs} "
        "vae_deterministic={vd} decode_fp32={dfp32}".format(
            name=active_scheduler_name,
            cls=scheduler_display,
            steps=num_inference_steps,
            rt=reconstruct_t,
            t_start=t_start,
            count=len(timesteps_list),
            gs=guidance_scale,
            vd=vae_deterministic,
            dfp32=decode_fp32,
        )
    )
    print(
        f"[INFO] real_timesteps_count={len(timesteps_list)} "
        f"t_start={timesteps_list[0]} t_end={timesteps_list[-1]}"
    )
    if isinstance(scheduler, DDIMScheduler) and eta:
        print(f"[INFO] DDIM eta={eta}")
    if refine_pass:
        print("[WARN] refine_pass enabled; may affect metric fairness.")
        refine_scheduler = _build_scheduler(active_scheduler_name)
        refine_timesteps_list, refine_start_index = _prepare_timesteps_list(
            refine_scheduler,
            refine_steps,
            device,
            "from_noisy_gt",
            refine_t,
            None,
            label="refine",
        )
        refine_t_start = int(refine_timesteps_list[0])
        refine_count = len(refine_timesteps_list)
        print(
            f"[INFO] refine_steps={refine_steps} refine_t={refine_t} "
            f"t_start={refine_t_start} real_timesteps_count={refine_count}"
        )
    else:
        refine_scheduler = None
        refine_timesteps_list = None

    use_guidance = False
    uncond_embeddings = None
    if guidance_scale > 1.0:
        try:
            uncond_inputs = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]
            use_guidance = True
            print("[WARN] CFG enabled; ensure the model was trained with unconditional dropout.")
        except Exception as exc:
            print(f"[WARN] Failed to build unconditional embeddings ({exc}); disabling CFG.")
            use_guidance = False
            guidance_scale = 1.0

    model_dtype = next(model.parameters()).dtype
    amp_enabled = device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16)

    def _amp_context():
        if amp_enabled:
            return torch.autocast(device_type="cuda", dtype=model_dtype)
        return contextlib.nullcontext()

    vae_original_dtype = next(vae.parameters()).dtype
    vae_device = next(vae.parameters()).device
    vae_device_type = "cuda" if vae_device.type == "cuda" else "cpu"
    vae_amp_enabled = vae_device.type == "cuda" and vae_original_dtype in (torch.float16, torch.bfloat16)

    def _decode_latents(latents):
        latents_to_decode = (latents / scaling_factor).to(device=vae_device)
        if decode_fp32:
            if vae_original_dtype != torch.float32:
                vae.to(dtype=torch.float32)
            try:
                with torch.autocast(device_type=vae_device_type, enabled=False):
                    decoded = vae.decode(latents_to_decode.float()).sample
            finally:
                if vae_original_dtype != torch.float32:
                    vae.to(dtype=vae_original_dtype)
            return decoded
        with torch.autocast(device_type=vae_device_type, dtype=vae_original_dtype, enabled=vae_amp_enabled):
            return vae.decode(latents_to_decode.to(dtype=vae_original_dtype)).sample

    def _maybe_fallback_to_ddim(exc):
        nonlocal scheduler
        nonlocal timesteps_list, start_index, t_start, scheduler_display
        nonlocal active_scheduler_name, fallback_used
        nonlocal refine_scheduler, refine_timesteps_list
        if active_scheduler_name != "unipc" or fallback_used or not _is_timestep_error(exc):
            return False
        print("[WARN] UniPC timestep error detected; falling back to DDIM for remaining samples.")
        active_scheduler_name = "ddim"
        scheduler = _build_scheduler(active_scheduler_name)
        timesteps_list, start_index = _prepare_timesteps_list(
            scheduler,
            num_inference_steps,
            device,
            init_mode,
            reconstruct_t,
            min_effective_steps,
            label="main-fallback",
        )
        t_start = int(timesteps_list[0])
        scheduler_display = scheduler.__class__.__name__
        print(
            "[INFO] fallback scheduler={name} ({cls}) steps={steps} t_start={t_start} "
            "real_timesteps_count={count}".format(
                name=active_scheduler_name,
                cls=scheduler_display,
                steps=num_inference_steps,
                t_start=t_start,
                count=len(timesteps_list),
            )
        )
        if refine_pass:
            refine_scheduler = _build_scheduler(active_scheduler_name)
            refine_timesteps_list, _ = _prepare_timesteps_list(
                refine_scheduler,
                refine_steps,
                device,
                "from_noisy_gt",
                refine_t,
                None,
                label="refine-fallback",
            )
            refine_t_start = int(refine_timesteps_list[0])
            refine_count = len(refine_timesteps_list)
            print(
                f"[INFO] fallback refine_steps={refine_steps} refine_t={refine_t} "
                f"t_start={refine_t_start} real_timesteps_count={refine_count}"
            )
        fallback_used = True
        return True

    generated_paths = []
    failed_indices = []
    cuda_poisoned = False
    success_count = 0  # 成功生成的图片计数器
    consecutive_cuda_errors = 0

    for idx, item in enumerate(tqdm(test_data, desc="生成中")):
        try:
            # 检查Person图片和Cloth图片是否都存在（确保一一对应）
            person_path = os.path.join(image_root_path, item["image_file"])
            cloth_path = os.path.join(cloth_root_path, item["cloth_file"])

            if not os.path.exists(person_path):
                print(f"\n警告: Person图片不存在: {person_path}")
                failed_indices.append(idx)
                consecutive_cuda_errors = 0
                continue

            if not os.path.exists(cloth_path):
                print(f"\n警告: Cloth图片不存在: {cloth_path}")
                failed_indices.append(idx)
                consecutive_cuda_errors = 0
                continue

            person_img = Image.open(person_path).convert("RGB")

            # 准备输入
            person_vae = vae_transform(person_img).unsqueeze(0).to(device)
            person_clip = clip_transform(person_img).unsqueeze(0).to(device)

            # 编码Person图片
            person_posterior = vae.encode(person_vae).latent_dist
            if vae_deterministic:
                if hasattr(person_posterior, "mode"):
                    person_latents = person_posterior.mode()
                else:
                    person_latents = person_posterior.mean
            else:
                person_latents = person_posterior.sample()
            person_latents = person_latents * scaling_factor

            person_image_embeds = image_encoder(
                person_clip,
                output_hidden_states=True
            ).hidden_states[-2]

            # 准备文本（添加cloth_type指令，与训练时保持一致）
            text = item.get("item_text", None)
            if text is None:
                text = item.get("text", "")
            if isinstance(text, list):
                text = text[0] if text else ""
            if text is None:
                text = ""
            cloth_type = item.get("cloth_type", "all")

            # 添加与训练时完全相同的prefix（来自IGPair.py）
            prefix = ""
            if cloth_type == "upper":
                prefix = "The image focuses on the upper-body garment. "
            elif cloth_type == "lower":
                prefix = "The image focuses on the lower-body garment. "
            # For "all" or unknown, no prefix

            text_with_prefix = prefix + text

            text_inputs = tokenizer(
                text_with_prefix,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

            if init_mode == "from_noisy_gt":
                cloth_img = Image.open(cloth_path).convert("RGB")
                cloth_vae = vae_transform(cloth_img).unsqueeze(0).to(device)
                cloth_posterior = vae.encode(cloth_vae).latent_dist
                if vae_deterministic:
                    if hasattr(cloth_posterior, "mode"):
                        cloth_latents = cloth_posterior.mode()
                    else:
                        cloth_latents = cloth_posterior.mean
                else:
                    cloth_latents = cloth_posterior.sample()
                cloth_latents = cloth_latents * scaling_factor

            g = torch.Generator(device=device)
            g.manual_seed(seed + idx)
            if init_mode == "from_noisy_gt":
                noise = torch.randn(
                    cloth_latents.shape,
                    generator=g,
                    device=cloth_latents.device,
                    dtype=cloth_latents.dtype,
                )
            else:
                noise = torch.randn(
                    person_latents.shape,
                    generator=g,
                    device=device,
                    dtype=person_latents.dtype,
                )

            def _init_latents_for_scheduler():
                _set_scheduler_timesteps(scheduler, num_inference_steps, device)
                if init_mode == "from_noisy_gt":
                    t_start_local = int(timesteps_list[0])
                    t_tensor = _normalize_timestep(t_start_local, latents=cloth_latents)
                    return scheduler.add_noise(cloth_latents, noise, t_tensor)
                latents_local = noise
                if hasattr(scheduler, "init_noise_sigma"):
                    latents_local = latents_local * scheduler.init_noise_sigma
                return latents_local

            latents_init = _init_latents_for_scheduler()
            step_kwargs = {}
            if isinstance(scheduler, DDIMScheduler):
                step_kwargs["eta"] = eta

            attempt = 0
            while True:
                latents = latents_init.detach().clone()
                try:
                    with _amp_context():
                        for t in timesteps_list:
                            timestep_tensor = _normalize_timestep(t, latents=latents)
                            latents_input = scheduler.scale_model_input(latents, timestep_tensor)

                            noise_pred = model(
                                text_embeddings,
                                latents_input,
                                person_latents,
                                person_image_embeds,
                                timestep_tensor
                            )

                            if use_guidance:
                                noise_pred_uncond = model(
                                    uncond_embeddings,
                                    latents_input,
                                    person_latents,
                                    person_image_embeds,
                                    timestep_tensor
                                )
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                            latents = scheduler.step(
                                noise_pred, timestep_tensor, latents, **step_kwargs
                            ).prev_sample
                    break
                except Exception as step_exc:
                    if attempt == 0 and _maybe_fallback_to_ddim(step_exc):
                        attempt += 1
                        latents_init = _init_latents_for_scheduler()
                        step_kwargs = {}
                        if isinstance(scheduler, DDIMScheduler):
                            step_kwargs["eta"] = eta
                        continue
                    raise

            # 解码到图像（与训练验证代码一致）
            decoded = _decode_latents(latents)
            image = (decoded.float() + 1.0) / 2.0
            image = torch.clamp(image, 0.0, 1.0)

            if refine_pass:
                refine_input = decoded.to(dtype=vae_original_dtype)
                refine_posterior = vae.encode(refine_input).latent_dist
                if vae_deterministic:
                    if hasattr(refine_posterior, "mode"):
                        latents_refine = refine_posterior.mode()
                    else:
                        latents_refine = refine_posterior.mean
                else:
                    latents_refine = refine_posterior.sample()
                latents_refine = latents_refine * scaling_factor
                refine_base_latents = latents_refine
                refine_noise = torch.randn(
                    latents_refine.shape,
                    generator=g,
                    device=latents_refine.device,
                    dtype=latents_refine.dtype,
                )

                def _init_refine_latents():
                    _set_scheduler_timesteps(refine_scheduler, refine_steps, device)
                    t_start_local = int(refine_timesteps_list[0])
                    t_tensor = _normalize_timestep(t_start_local, latents=refine_base_latents)
                    return refine_scheduler.add_noise(refine_base_latents, refine_noise, t_tensor)

                latents_refine_init = _init_refine_latents()
                refine_step_kwargs = {}
                if isinstance(refine_scheduler, DDIMScheduler):
                    refine_step_kwargs["eta"] = eta

                attempt_refine = 0
                while True:
                    latents_refine_run = latents_refine_init.detach().clone()
                    try:
                        with _amp_context():
                            for t in refine_timesteps_list:
                                timestep_tensor = _normalize_timestep(t, latents=latents_refine_run)
                                latents_refine_input = refine_scheduler.scale_model_input(
                                    latents_refine_run, timestep_tensor
                                )

                                noise_pred = model(
                                    text_embeddings,
                                    latents_refine_input,
                                    person_latents,
                                    person_image_embeds,
                                    timestep_tensor
                                )

                                if use_guidance:
                                    noise_pred_uncond = model(
                                        uncond_embeddings,
                                        latents_refine_input,
                                        person_latents,
                                        person_image_embeds,
                                        timestep_tensor
                                    )
                                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                                latents_refine_run = refine_scheduler.step(
                                    noise_pred, timestep_tensor, latents_refine_run, **refine_step_kwargs
                                ).prev_sample
                        latents_refine = latents_refine_run
                        break
                    except Exception as refine_exc:
                        if attempt_refine == 0 and _maybe_fallback_to_ddim(refine_exc):
                            attempt_refine += 1
                            if not refine_timesteps_list or refine_scheduler is None:
                                raise
                            latents_refine_init = _init_refine_latents()
                            refine_step_kwargs = {}
                            if isinstance(refine_scheduler, DDIMScheduler):
                                refine_step_kwargs["eta"] = eta
                            continue
                        raise

                decoded = _decode_latents(latents_refine)
                image = (decoded.float() + 1.0) / 2.0
                image = torch.clamp(image, 0.0, 1.0)

            # 保存生成的图片（使用连续的success_count命名）
            image_np = image[0].cpu().permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            if image_pil.size != (512, 512):
                image_pil = image_pil.resize((512, 512), resample=Image.BICUBIC)

            # 使用连续计数命名，确保与ground truth对应
            save_path = os.path.join(output_dir, f"{success_count:06d}_generated.png")
            image_pil.save(save_path)
            generated_paths.append(save_path)

            success_count += 1  # 只有成功时才递增
            consecutive_cuda_errors = 0

        except Exception as e:
            if _is_cuda_failure(e):
                failed_indices.append(idx)
                consecutive_cuda_errors += 1
                print(f"[ERROR] idx={idx} err={e}")
                safe_cuda_cleanup("after_gen_error")
                if consecutive_cuda_errors >= 3:
                    cuda_poisoned = True
                    print("[FATAL] CUDA poisoned; stop generation early and fallback metrics to CPU.")
                    if idx + 1 < len(test_data):
                        failed_indices.extend(range(idx + 1, len(test_data)))
                    break
                continue
            consecutive_cuda_errors = 0
            print(f"\n错误: 生成第 {idx} 张图片失败: {e}")
            failed_indices.append(idx)
            continue

    print(f"\n{'='*60}")
    print(f"[OK] Successfully generated: {len(generated_paths)} images")
    if failed_indices:
        print(f"[ERROR] Failed: {len(failed_indices)} images")
    print(f"[OK] Images saved to: {output_dir}")
    print(f"{'='*60}")

    return generated_paths, failed_indices, cuda_poisoned


def prepare_ground_truth(
    test_data,
    image_root_path,
    cloth_root_path,
    output_dir,
    failed_indices=None,
    generated_dir=None,
):
    """
    准备Ground Truth图片（复制真实的cloth图片）

    Args:
        test_data: 测试数据
        image_root_path: 图片根目录
        output_dir: 输出目录
        failed_indices: 生成失败的索引列表（跳过这些）
    """
    print(f"\n{'='*60}")
    print("准备Ground Truth图片")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    failed_indices = failed_indices or []

    gt_paths = []
    missing_count = 0
    success_count = 0  # 成功准备的图片计数器

    target_size = None
    if generated_dir and os.path.isdir(generated_dir):
        import glob
        gen_paths = sorted(glob.glob(os.path.join(generated_dir, "*_generated.png")))
        if gen_paths:
            try:
                with Image.open(gen_paths[0]) as gen_img:
                    target_size = gen_img.size
            except Exception:
                target_size = None

    if target_size is None:
        target_size = (512, 512)
    print(f"[INFO] Groundtruth resize target: {target_size[0]}x{target_size[1]}")

    for idx, item in enumerate(tqdm(test_data, desc="复制中")):
        # 跳过生成失败的样本
        if idx in failed_indices:
            continue

        try:
            # 获取Ground Truth（真实的garment图片）
            cloth_path = os.path.join(cloth_root_path, item["cloth_file"])

            if not os.path.exists(cloth_path):
                print(f"\n警告: Ground Truth图片不存在: {cloth_path}")
                missing_count += 1
                continue

            # 使用与生成图片相同的连续计数命名
            dest_path = os.path.join(output_dir, f"{success_count:06d}_groundtruth.png")

            # 打开并保存为PNG（统一格式）
            img = Image.open(cloth_path).convert("RGB")
            img = img.resize(target_size, resample=Image.BICUBIC)
            img.save(dest_path)
            gt_paths.append(dest_path)

            success_count += 1  # 只有成功时才递增

        except Exception as e:
            print(f"\n错误: 处理Ground Truth第 {idx} 张失败: {e}")
            missing_count += 1
            continue

    print(f"\n{'='*60}")
    print(f"[OK] Successfully prepared: {len(gt_paths)} Ground Truth images")
    if missing_count > 0:
        print(f"[ERROR] Missing: {missing_count} images")
    print(f"[OK] Images saved to: {output_dir}")
    print(f"{'='*60}")

    return gt_paths


def calculate_pixel_metrics(real_dir, generated_dir, device):
    """计算像素级指标：PSNR、SSIM、LPIPS"""
    print(f"\n{'='*60}")
    print("计算像素级指标 (PSNR, SSIM, LPIPS)")
    print(f"{'='*60}")

    import glob
    real_paths = sorted(glob.glob(os.path.join(real_dir, "*_groundtruth.png")))
    gen_paths = sorted(glob.glob(os.path.join(generated_dir, "*_generated.png")))

    if len(real_paths) == 0 or len(gen_paths) == 0:
        print("❌ 错误: 未找到图片!")
        return None

    # 确保数量匹配
    if len(real_paths) != len(gen_paths):
        min_count = min(len(real_paths), len(gen_paths))
        real_paths = real_paths[:min_count]
        gen_paths = gen_paths[:min_count]

    # 初始化LPIPS模型 (使用AlexNet)
    print("初始化LPIPS模型...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    psnr_values = []
    ssim_values = []
    lpips_values = []
    resize_count = 0
    skipped_count = 0
    warned_resize = False  # 仅提示一次尺寸对齐

    num_pairs = len(real_paths)
    print(f"计算 {num_pairs} 对图片的指标...")
    for real_path, gen_path in tqdm(zip(real_paths, gen_paths), total=len(real_paths), desc="计算中"):
        try:
            # 读取图片
            real_img = np.array(Image.open(real_path).convert('RGB'))
            gen_img = np.array(Image.open(gen_path).convert('RGB'))

            # 尺寸对齐：生成图通常为 512x512，而 GT 可能保留原始分辨率；像素指标要求尺寸一致
            if real_img.shape[:2] != gen_img.shape[:2]:
                if not warned_resize:
                    print("[WARN] Resizing generated image to match groundtruth size for pixel metrics")
                    warned_resize = True
                resize_count += 1
                gen_img = np.array(
                    Image.fromarray(gen_img).resize(
                        (real_img.shape[1], real_img.shape[0]),
                        resample=Image.BICUBIC,
                    )
                )

            # 计算 PSNR
            psnr_val = psnr(real_img, gen_img, data_range=255)
            psnr_values.append(psnr_val)

            # 计算 SSIM (multichannel for RGB)
            ssim_val = ssim(real_img, gen_img, channel_axis=2, data_range=255)
            ssim_values.append(ssim_val)

            # 计算 LPIPS (需要转换为tensor，范围[-1, 1])
            real_tensor = torch.from_numpy(real_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            gen_tensor = torch.from_numpy(gen_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            real_tensor = real_tensor.to(device)
            gen_tensor = gen_tensor.to(device)

            with torch.no_grad():
                lpips_val = lpips_model(real_tensor, gen_tensor).item()
            lpips_values.append(lpips_val)

        except Exception as e:
            skipped_count += 1
            print(f"\n[WARN] Skipping metric pair for {real_path}: {e}")
            continue

    # 计算平均值
    psnr_mean = np.mean(psnr_values) if psnr_values else None
    psnr_std = np.std(psnr_values) if psnr_values else None
    ssim_mean = np.mean(ssim_values) if ssim_values else None
    ssim_std = np.std(ssim_values) if ssim_values else None
    lpips_mean = np.mean(lpips_values) if lpips_values else None
    lpips_std = np.std(lpips_values) if lpips_values else None

    print(f"\n像素级指标结果:")
    print(f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB" if psnr_mean else "  PSNR: N/A")
    print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}" if ssim_mean else "  SSIM: N/A")
    print(f"  LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}" if lpips_mean else "  LPIPS: N/A")
    if skipped_count > 0:
        print(f"  Skipped pairs: {skipped_count}")
    print(f"[INFO] resized_pairs={resize_count}/{num_pairs}")

    # 清理显存
    del lpips_model
    safe_cuda_cleanup("after_pixel_metrics")

    return {
        'psnr_mean': psnr_mean,
        'psnr_std': psnr_std,
        'ssim_mean': ssim_mean,
        'ssim_std': ssim_std,
        'lpips_mean': lpips_mean,
        'lpips_std': lpips_std,
        'resize_count': resize_count,
        'num_pairs': num_pairs,
        'skipped_count': skipped_count,
    }


def _collect_metric_paths(real_dir, generated_dir):
    import glob
    real_paths = sorted(glob.glob(os.path.join(real_dir, "*_groundtruth.png")))
    gen_paths = sorted(glob.glob(os.path.join(generated_dir, "*_generated.png")))

    print(f"Ground Truth图片数量: {len(real_paths)}")
    print(f"生成图片数量: {len(gen_paths)}")

    if len(real_paths) == 0 or len(gen_paths) == 0:
        return [], []

    if len(real_paths) != len(gen_paths):
        print("⚠ 警告: Ground Truth和生成图片数量不匹配!")
        min_count = min(len(real_paths), len(gen_paths))
        real_paths = real_paths[:min_count]
        gen_paths = gen_paths[:min_count]
        print(f"  使用前 {min_count} 张图片进行评估")

    return real_paths, gen_paths


def _device_to_cleanfid(device):
    if isinstance(device, torch.device):
        return device.type
    device_str = str(device).lower()
    if "cuda" in device_str:
        return "cuda"
    return "cpu"


def _compute_cleanfid_fid(real_dir, fake_dir, mode, device):
    device_str = _device_to_cleanfid(device)
    try:
        return cleanfid_fid.compute_fid(real_dir, fake_dir, mode=mode, device=device_str)
    except TypeError:
        return cleanfid_fid.compute_fid(real_dir, fake_dir, mode=mode)


def _compute_cleanfid_kid(real_dir, fake_dir, mode, device, subset_size):
    device_str = _device_to_cleanfid(device)
    try:
        return cleanfid_fid.compute_kid(
            real_dir,
            fake_dir,
            mode=mode,
            num_subsets=100,
            max_subset_size=subset_size,
            device=device_str,
        )
    except TypeError:
        return cleanfid_fid.compute_kid(real_dir, fake_dir, mode=mode)


def _compute_standard_kid(real_paths, gen_paths, device):
    if not FID_KID_AVAILABLE:
        raise RuntimeError(
            "FID/KID dependencies are missing. Install required packages, e.g.: "
            "pip install scipy pytorch-fid"
        ) from FID_KID_IMPORT_ERROR
    print("计算 KID (standard)...")
    extractor = InceptionV3FeatureExtractor(device=device)
    features_real = extractor.extract_features(real_paths, batch_size=32)
    features_gen = extractor.extract_features(gen_paths, batch_size=32)
    subset_size = min(1000, len(real_paths), len(gen_paths))
    kid_mean, kid_std = calculate_kid_standard(
        features_real,
        features_gen,
        subset_size=subset_size,
        num_subsets=100
    )
    return kid_mean, kid_std


def evaluate_metrics_backend(real_dir, generated_dir, device, fid_backend="auto", kid_backend="auto", cleanfid_mode="clean"):
    if fid_backend == "auto":
        if os.name == "nt":
            fid_backend = "standard"
        else:
            fid_backend = "cleanfid" if CLEANFID_AVAILABLE else "standard"
    if kid_backend == "auto":
        if os.name == "nt":
            kid_backend = "standard"
        else:
            kid_backend = "cleanfid" if CLEANFID_AVAILABLE else "standard"

    if os.name == "nt" and fid_backend == "cleanfid":
        print("[WARN] cleanfid disabled on Windows; using standard backend for FID.")
        fid_backend = "standard"
    if os.name == "nt" and kid_backend == "cleanfid":
        print("[WARN] cleanfid disabled on Windows; using standard backend for KID.")
        kid_backend = "standard"

    real_paths, gen_paths = _collect_metric_paths(real_dir, generated_dir)
    if not real_paths or not gen_paths:
        return None, None, None, "No images found for FID/KID", "No images found for FID/KID"

    fid_value = None
    kid_mean = None
    kid_std = None
    fid_error = None
    kid_error = None

    features_real = None
    features_gen = None
    if fid_backend == "standard" or kid_backend == "standard":
        if not FID_KID_AVAILABLE:
            raise RuntimeError(
                "FID/KID dependencies are missing. Install required packages, e.g.: "
                "pip install scipy pytorch-fid"
            ) from FID_KID_IMPORT_ERROR
        print("计算 FID/KID (standard)...")
        extractor = InceptionV3FeatureExtractor(device=device)
        features_real = extractor.extract_features(real_paths, batch_size=32)
        features_gen = extractor.extract_features(gen_paths, batch_size=32)

    if fid_backend == "standard":
        try:
            fid_value = calculate_fid_standard(features_real, features_gen)
        except Exception as exc:
            fid_error = str(exc)

    if kid_backend == "standard":
        try:
            subset_size = min(1000, len(real_paths), len(gen_paths))
            num_subsets = 1 if subset_size <= 500 else 100
            kid_mean, kid_std = calculate_kid_standard(
                features_real,
                features_gen,
                subset_size=subset_size,
                num_subsets=num_subsets
            )
        except Exception as exc:
            kid_error = str(exc)

    use_cleanfid = (fid_backend == "cleanfid") or (kid_backend == "cleanfid")
    if use_cleanfid:
        if not CLEANFID_AVAILABLE:
            raise RuntimeError(
                "cleanfid is not available. Install it via: pip install clean-fid"
            ) from CLEANFID_IMPORT_ERROR

        print(f"\n{'='*60}")
        print("计算评估指标 (FID & KID)")
        print(f"{'='*60}")
        temp_dir = tempfile.mkdtemp(prefix="cleanfid_eval_")
        real_tmp = os.path.join(temp_dir, "real")
        fake_tmp = os.path.join(temp_dir, "fake")
        os.makedirs(real_tmp, exist_ok=True)
        os.makedirs(fake_tmp, exist_ok=True)
        try:
            for idx, (real_path, gen_path) in enumerate(zip(real_paths, gen_paths)):
                shutil.copy(real_path, os.path.join(real_tmp, f"{idx:06d}.png"))
                shutil.copy(gen_path, os.path.join(fake_tmp, f"{idx:06d}.png"))

            if fid_backend == "cleanfid":
                try:
                    print("\n计算 FID (cleanfid)...")
                    fid_value = _compute_cleanfid_fid(real_tmp, fake_tmp, cleanfid_mode, device)
                except Exception as exc:
                    fid_error = str(exc)
                    fid_value = None

            if kid_backend == "cleanfid":
                try:
                    if hasattr(cleanfid_fid, "compute_kid"):
                        print("计算 KID (cleanfid)...")
                        subset_size = min(1000, len(real_paths), len(gen_paths))
                        kid_result = _compute_cleanfid_kid(real_tmp, fake_tmp, cleanfid_mode, device, subset_size)
                        if isinstance(kid_result, tuple) and len(kid_result) >= 2:
                            kid_mean, kid_std = kid_result[0], kid_result[1]
                        else:
                            kid_mean = float(kid_result) if kid_result is not None else None
                            kid_std = None
                    else:
                        raise RuntimeError("cleanfid.compute_kid not available")
                except Exception as exc:
                    kid_error = str(exc)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return fid_value, kid_mean, kid_std, fid_error, kid_error


def evaluate_metrics(real_dir, generated_dir, device):
    """计算 FID 和 KID 指标"""
    if not FID_KID_AVAILABLE:
        raise RuntimeError(
            "FID/KID dependencies are missing. Install required packages, e.g.: "
            "pip install scipy pytorch-fid"
        ) from FID_KID_IMPORT_ERROR
    print(f"\n{'='*60}")
    print("计算评估指标 (FID & KID)")
    print(f"{'='*60}")

    # 获取图片路径
    import glob
    real_paths = sorted(glob.glob(os.path.join(real_dir, "*_groundtruth.png")))
    gen_paths = sorted(glob.glob(os.path.join(generated_dir, "*_generated.png")))

    print(f"Ground Truth图片数量: {len(real_paths)}")
    print(f"生成图片数量: {len(gen_paths)}")

    if len(real_paths) == 0:
        print("❌ 错误: 未找到Ground Truth图片!")
        return None, None, None

    if len(gen_paths) == 0:
        print("❌ 错误: 未找到生成图片!")
        return None, None, None

    # 确保数量匹配
    if len(real_paths) != len(gen_paths):
        print(f"⚠ 警告: Ground Truth和生成图片数量不匹配!")
        min_count = min(len(real_paths), len(gen_paths))
        real_paths = real_paths[:min_count]
        gen_paths = gen_paths[:min_count]
        print(f"  使用前 {min_count} 张图片进行评估")

    # 初始化特征提取器
    print(f"\n初始化 Inception V3 特征提取器...")
    extractor = InceptionV3FeatureExtractor(device=device)

    # 提取特征
    print(f"提取Ground Truth特征...")
    features_real = extractor.extract_features(real_paths, batch_size=32)

    print(f"提取生成图片特征...")
    features_gen = extractor.extract_features(gen_paths, batch_size=32)

    # 计算 FID
    print(f"\n计算 FID...")
    fid_value = calculate_fid_standard(features_real, features_gen)

    # 计算 KID
    print(f"计算 KID...")
    # 使用合适的子集大小
    subset_size = min(1000, len(real_paths), len(gen_paths))
    kid_mean, kid_std = calculate_kid_standard(
        features_real, features_gen,
        subset_size=subset_size,
        num_subsets=100
    )

    # 显示结果
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"样本数量: {len(real_paths)}")
    print(f"FID: {fid_value:.4f}")
    if kid_mean is not None:
        print(f"KID: {kid_mean:.6f} ± {kid_std:.6f}")
        print(f"KID* (×1000): {kid_mean*1000:.3f} ± {kid_std*1000:.3f}")
    print(f"{'='*60}")
    print("\n说明:")
    print("- FID 和 KID 越低越好")
    print("- FID 衡量生成图片和真实图片的特征分布差异")
    print("- KID 是基于核方法的距离度量，更稳健")
    print(f"{'='*60}\n")

    return fid_value, kid_mean, kid_std


def main():
    parser = argparse.ArgumentParser(description="评估 Extractor 模型生成质量")
    parser.add_argument("--test_set_json", type=str, required=True,
                       help="测试集JSON文件路径")
    parser.add_argument("--image_root", type=str, required=True,
                       help="图片根目录")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Sampling steps")
    parser.add_argument("--scheduler", type=str, choices=["ddim", "unipc", "dpmpp_2m_karras"],
                       default="ddim",
                       help="Scheduler type")
    parser.add_argument("--eta", type=float, default=0.0,
                       help="DDIM eta (ignored by other schedulers)")
    parser.add_argument("--decode_fp32", type=_str_to_bool, nargs="?", const=True, default=True,
                       help="Decode VAE in FP32")
    parser.add_argument("--refine_pass", action="store_true", default=False,
                       help="Enable refine pass")
    parser.add_argument("--refine_t", type=int, default=150,
                       help="Refine start timestep")
    parser.add_argument("--refine_steps", type=int, default=10,
                       help="Refine sampling steps")
    parser.add_argument("--fid_backend", type=str, choices=["auto", "standard", "cleanfid"],
                       default="auto",
                       help="FID backend (auto=cleanfid if available on non-Windows, else standard)")
    parser.add_argument("--kid_backend", type=str, choices=["auto", "standard", "cleanfid"],
                       default="auto",
                       help="KID backend (auto=cleanfid if available on non-Windows, else standard)")
    parser.add_argument("--cleanfid_mode", type=str, choices=["clean", "legacy_pytorch"],
                       default="clean",
                       help="cleanfid mode (only for cleanfid backend)")
    parser.add_argument("--pretrained_model", type=str,
                       default="models/IMAGDressing",
                       help="预训练模型路径")
    parser.add_argument("--image_encoder", type=str,
                       default="models/IMAGDressing/image_encoder",
                       help="Image encoder路径")
    parser.add_argument("--vae_path", type=str,
                       default="models/IMAGDressing/sd-vae-ft-mse",
                       help="VAE路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    parser.add_argument("--skip_generation", action="store_true",
                       help="跳过生成步骤（如果已生成）")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--cloth_root", type=str, default=None,
                       help="Optional cloth root (fallback if image_root is empty)")
    parser.add_argument("--prompts_cache", type=str, default=None,
                       help="Prompt cache path (json/jsonl)")
    parser.add_argument("--cache_key_mode", type=str, choices=["basename_pair", "path_pair", "auto"],
                       default="basename_pair",
                       help="Prompt cache key mode")
    parser.add_argument("--prompt_fallback", type=str, default="",
                       help="Fallback prompt when cache misses")
    parser.add_argument("--init_mode", type=str, choices=["from_noise", "from_noisy_gt"],
                       default="from_noisy_gt",
                       help="Latent init mode")
    parser.add_argument("--reconstruct_t", type=int, default=600,
                       help="Timestep used for noisy GT initialization")
    parser.add_argument("--min_effective_steps", type=int, default=25,
                       help="Minimum effective denoise steps when using from_noisy_gt")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--log_first_n_prompts", type=int, default=0,
                       help="Log first N prompt sources")
    parser.add_argument("--strict_empty_prompt", action="store_true",
                       help="Raise if any prompt is empty after fallback")
    parser.add_argument("--vae_deterministic", type=_str_to_bool, nargs="?", const=True, default=True,
                       help="Use deterministic VAE encode/decode")
    parser.add_argument("--allow_zero_prompt_cache_hit", action="store_true",
                       help="Allow zero prompt cache hit")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples to evaluate (for quick smoke test)")

    args = parser.parse_args()

    seed = getattr(args, "seed", 42)
    prompt_fallback = args.prompt_fallback or ""

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    print(f"[INFO] init_mode={args.init_mode} reconstruct_t={args.reconstruct_t} steps={args.num_inference_steps}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    image_root = args.image_root
    cloth_root = args.cloth_root or image_root
    if not image_root and cloth_root:
        print("[WARN] image_root is empty; using cloth_root for person images")
        image_root = cloth_root
    if not cloth_root:
        cloth_root = image_root

    # 加载测试集
    print(f"\n加载测试集: {args.test_set_json}")
    with open(args.test_set_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    if isinstance(test_data, dict) and "data" in test_data:
        test_data = test_data["data"]
    if args.max_samples is not None:
        test_data = test_data[:args.max_samples]
        print(f"[INFO] max_samples={args.max_samples} -> using {len(test_data)} samples")
    print(f"[OK] Test set size: {len(test_data)}")

    prompts_cache = _load_prompts_cache(args.prompts_cache) if args.prompts_cache else {}
    if args.prompts_cache:
        print(f"[INFO] prompts_cache entries: {len(prompts_cache)}")
    prompt_cache_hit, fallback_count, empty_count = _apply_prompt_cache(
        test_data=test_data,
        prompts_cache=prompts_cache,
        prompt_fallback=prompt_fallback,
        log_first_n_prompts=args.log_first_n_prompts,
        strict_empty_prompt=args.strict_empty_prompt,
        allow_zero_prompt_cache_hit=args.allow_zero_prompt_cache_hit,
        cache_requested=bool(args.prompts_cache),
        cache_key_mode=args.cache_key_mode,
    )

    # 定义输出目录
    generated_dir = os.path.join(args.output_dir, "generated")
    groundtruth_dir = os.path.join(args.output_dir, "groundtruth")

    failed_indices = []
    cuda_poisoned = False

    # 步骤1: 生成图片
    if not args.skip_generation:
        # 加载模型
        model, vae, text_encoder, image_encoder, tokenizer = load_model(
            args.checkpoint,
            args.pretrained_model,
            args.image_encoder,
            args.vae_path,
            device
        )

        # 生成图片
        generated_paths, failed_indices, cuda_poisoned = generate_images(
            model, vae, text_encoder, image_encoder, tokenizer,
            test_data, image_root, cloth_root, generated_dir, device,
            num_inference_steps=args.num_inference_steps,
            seed=seed,
            guidance_scale=args.guidance_scale,
            scheduler_name=args.scheduler,
            eta=args.eta,
            init_mode=args.init_mode,
            reconstruct_t=args.reconstruct_t,
            min_effective_steps=args.min_effective_steps,
            vae_deterministic=args.vae_deterministic,
            decode_fp32=args.decode_fp32,
            refine_pass=args.refine_pass,
            refine_t=args.refine_t,
            refine_steps=args.refine_steps,
        )

        # 清理显存
        del model, vae, text_encoder, image_encoder, tokenizer
        global CUDA_POISONED
        CUDA_POISONED = cuda_poisoned
        safe_cuda_cleanup("after_generation")
    else:
        print(f"\n⏭ 跳过生成步骤，使用已有图片: {generated_dir}")

    CUDA_POISONED = cuda_poisoned

    metrics_device = torch.device("cpu") if cuda_poisoned else device
    if cuda_poisoned:
        print("[WARN] CUDA poisoned; running metrics on CPU.")

    # 步骤2: 准备Ground Truth
    gt_paths = prepare_ground_truth(
        test_data,
        image_root_path=image_root,
        cloth_root_path=cloth_root,
        output_dir=groundtruth_dir,
        failed_indices=failed_indices,
        generated_dir=generated_dir,
    )

    # 步骤3: 计算像素级指标 (PSNR, SSIM, LPIPS)
    pixel_metrics = calculate_pixel_metrics(groundtruth_dir, generated_dir, metrics_device)

    # 步骤4: 计算分布指标 (FID, KID)
    fid = None
    kid_mean = None
    kid_std = None
    fid_error = None
    kid_error = None
    try:
        fid, kid_mean, kid_std, fid_error, kid_error = evaluate_metrics_backend(
            groundtruth_dir,
            generated_dir,
            metrics_device,
            fid_backend=args.fid_backend,
            kid_backend=args.kid_backend,
            cleanfid_mode=args.cleanfid_mode,
        )
    except Exception as e:
        fid_error = str(e)
        kid_error = str(e)
        print(f"[WARN] FID/KID calculation failed: {e}")

    # 保存结果
    checkpoint_name = Path(args.checkpoint).stem
    resize_count = pixel_metrics.get("resize_count", 0) if pixel_metrics else 0
    num_pairs = pixel_metrics.get("num_pairs", 0) if pixel_metrics else 0
    skipped_count = pixel_metrics.get("skipped_count", 0) if pixel_metrics else 0
    if skipped_count > 0:
        print(f"[WARN] Skipped metric pairs: {skipped_count}")

    results = {
        "checkpoint": args.checkpoint,
        "checkpoint_name": checkpoint_name,
        "test_set": args.test_set_json,
        "num_samples": len(test_data),
        "num_successful": len(test_data) - len(failed_indices),
        "num_failed": len(failed_indices),
        "num_inference_steps": args.num_inference_steps,
        "scheduler": args.scheduler,
        "eta": args.eta,
        "seed": seed,
        "guidance_scale": args.guidance_scale,
        "init_mode": args.init_mode,
        "reconstruct_t": args.reconstruct_t,
        "min_effective_steps": args.min_effective_steps,
        "vae_deterministic": bool(args.vae_deterministic),
        "decode_fp32": bool(args.decode_fp32),
        "refine_pass": bool(args.refine_pass),
        "refine_t": args.refine_t,
        "refine_steps": args.refine_steps,
        "max_samples": args.max_samples,
        "FID_backend": args.fid_backend,
        "KID_backend": args.kid_backend,
        "cleanfid_mode": args.cleanfid_mode,
        "FID_error": fid_error,
        "KID_error": kid_error,
        "cache_key_mode": args.cache_key_mode,
        "prompt_cache_hit": prompt_cache_hit,
        "fallback_count": fallback_count,
        "empty_count": empty_count,
        "resize_count": resize_count,
        "num_pairs": num_pairs,
        "skipped_count": skipped_count,
        # Pixel-level metrics (higher PSNR/SSIM is better, lower LPIPS is better)
        "PSNR": float(pixel_metrics['psnr_mean']) if pixel_metrics and pixel_metrics['psnr_mean'] is not None else None,
        "PSNR_std": float(pixel_metrics['psnr_std']) if pixel_metrics and pixel_metrics['psnr_std'] is not None else None,
        "SSIM": float(pixel_metrics['ssim_mean']) if pixel_metrics and pixel_metrics['ssim_mean'] is not None else None,
        "SSIM_std": float(pixel_metrics['ssim_std']) if pixel_metrics and pixel_metrics['ssim_std'] is not None else None,
        "LPIPS": float(pixel_metrics['lpips_mean']) if pixel_metrics and pixel_metrics['lpips_mean'] is not None else None,
        "LPIPS_std": float(pixel_metrics['lpips_std']) if pixel_metrics and pixel_metrics['lpips_std'] is not None else None,
        # Distribution metrics (lower is better)
        "FID": float(fid) if fid is not None else None,
        "KID_mean": float(kid_mean) if kid_mean is not None else None,
        "KID_std": float(kid_std) if kid_std is not None else None,
        "KID_x1000": float(kid_mean * 1000) if kid_mean is not None else None,
        "KID_x1000_std": float(kid_std * 1000) if kid_std is not None else None,
    }
    if fid_error or kid_error:
        combined = []
        if fid_error:
            combined.append(f"FID: {fid_error}")
        if kid_error:
            combined.append(f"KID: {kid_error}")
        results["fid_kid_error"] = " | ".join(combined)

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    summary_results = {
        "psnr_mean": float(pixel_metrics['psnr_mean']) if pixel_metrics and pixel_metrics['psnr_mean'] is not None else None,
        "ssim_mean": float(pixel_metrics['ssim_mean']) if pixel_metrics and pixel_metrics['ssim_mean'] is not None else None,
        "lpips_mean": float(pixel_metrics['lpips_mean']) if pixel_metrics and pixel_metrics['lpips_mean'] is not None else None,
        "fid": float(fid) if fid is not None else None,
        "kid": float(kid_mean) if kid_mean is not None else None,
        "prompt_cache_hit": prompt_cache_hit,
        "fallback_count": fallback_count,
        "empty_count": empty_count,
        "resize_count": resize_count,
        "num_pairs": num_pairs,
        "skipped_count": skipped_count,
    }
    if fid_error:
        summary_results["fid_kid_error"] = fid_error

    summary_path = os.path.join(args.output_dir, "results.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n✅ 评估结果已保存到: {results_path}")
    print(f"[OK] Summary saved to: {summary_path}")
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)


if __name__ == "__main__":
    main()
