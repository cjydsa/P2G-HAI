#!/usr/bin/env python3
from __future__ import annotations

"""Standalone DressCode attention visualization for reverse garment extraction."""

import argparse
import csv
import gc
import json
import math
import os
import random
import statistics
import traceback
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFile
from torch import nn
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

DRESSCODE_CATEGORIES = ["upper_body", "lower_body", "dresses"]
VALID_TEST_ORDERS = ["paired", "unpaired"]
VALID_S_MODES = ["S0", "S1", "S2"]
FALLBACK_PROMPT = "a studio-style product shot of the garment only"
DEFAULT_SUBSET_JSON = "test_subset_500_DC.json"
DEFAULT_FIGURE_TOTAL = 12
DEFAULT_SUMMARY_TOTAL = 90
DEFAULT_PANEL_CELL = 256
PROXY_RESIZE_HW = (64, 64)
MAX_TRUE_ATTENTION_ELEMENTS = 16_000_000

CAttnProcessor2_0 = None
CacheAttnProcessor2_0 = None
RefCAttnProcessor2_0 = None
RefSAttnProcessor2_0 = None
SAttnProcessor2_0 = None
Resampler = None
AutoencoderKL = None
UNet2DConditionModel = None
CLIPTextModel = None
CLIPTokenizer = None
CLIPVisionModelWithProjection = None


@dataclass(frozen=True)
class AblationConfig:
    mode: str
    enable_ieb: bool
    enable_ha: bool
    direct_ieb: bool
    unet_train: str


@dataclass(frozen=True)
class TimestepTarget:
    index: int
    value: int
    label: str


class CheckpointLoadError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_float(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def safe_std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    if value is None:
        return ""
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key)) for key in fieldnames})


def sanitize_name(value: str) -> str:
    sanitized = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_"):
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "item"


def rel_to(root: Path, path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def ensure_runtime_imports() -> None:
    global CAttnProcessor2_0
    global CacheAttnProcessor2_0
    global RefCAttnProcessor2_0
    global RefSAttnProcessor2_0
    global SAttnProcessor2_0
    global Resampler
    global AutoencoderKL
    global UNet2DConditionModel
    global CLIPTextModel
    global CLIPTokenizer
    global CLIPVisionModelWithProjection

    if CAttnProcessor2_0 is not None:
        return

    from adapter.attention_processor import (
        CAttnProcessor2_0 as _CAttnProcessor2_0,
        CacheAttnProcessor2_0 as _CacheAttnProcessor2_0,
        RefCAttnProcessor2_0 as _RefCAttnProcessor2_0,
        RefSAttnProcessor2_0 as _RefSAttnProcessor2_0,
        SAttnProcessor2_0 as _SAttnProcessor2_0,
    )
    from adapter.resampler import Resampler as _Resampler
    from diffusers import AutoencoderKL as _AutoencoderKL, UNet2DConditionModel as _UNet2DConditionModel
    from transformers import (
        CLIPTextModel as _CLIPTextModel,
        CLIPTokenizer as _CLIPTokenizer,
        CLIPVisionModelWithProjection as _CLIPVisionModelWithProjection,
    )

    CAttnProcessor2_0 = _CAttnProcessor2_0
    CacheAttnProcessor2_0 = _CacheAttnProcessor2_0
    RefCAttnProcessor2_0 = _RefCAttnProcessor2_0
    RefSAttnProcessor2_0 = _RefSAttnProcessor2_0
    SAttnProcessor2_0 = _SAttnProcessor2_0
    Resampler = _Resampler
    AutoencoderKL = _AutoencoderKL
    UNet2DConditionModel = _UNet2DConditionModel
    CLIPTextModel = _CLIPTextModel
    CLIPTokenizer = _CLIPTokenizer
    CLIPVisionModelWithProjection = _CLIPVisionModelWithProjection


def _normalize_rel_path(filename: str) -> str:
    if not filename:
        return ""
    normalized = filename.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def resolve_dresscode_person_path(root: str, category: str, image_file: str) -> str:
    if not image_file:
        return os.path.join(root, category, "images")
    normalized = _normalize_rel_path(image_file)
    rel_path = normalized
    if "/images/" in normalized:
        rel_path = normalized.split("/images/", 1)[1]
    elif normalized.startswith("images/"):
        rel_path = normalized.split("images/", 1)[1]
    elif category and normalized.startswith(category + "/"):
        rel_path = normalized.split("/", 1)[1]
        if rel_path.startswith("images/"):
            rel_path = rel_path.split("images/", 1)[1]
    rel_path = rel_path.lstrip("/")
    if not rel_path:
        rel_path = os.path.basename(normalized)
    return os.path.join(root, category, "images", rel_path)


def resolve_dresscode_cloth_path(root: str, category: str, cloth_file: str) -> str:
    if not cloth_file:
        return os.path.join(root, category, "cloth")
    normalized = _normalize_rel_path(cloth_file)
    if category and normalized.startswith(category + "/"):
        normalized = normalized.split("/", 1)[1]
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    rel_name = normalized or os.path.basename(cloth_file)
    cloth_dirs = ["cloth", "clothes", "cloths", "garment", "garments", "images"]
    for subdir in cloth_dirs:
        candidate = os.path.join(root, category, subdir, rel_name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(root, category, cloth_dirs[0], rel_name)


def _pairs_filename(split: str, test_order: str) -> str:
    if split == "train":
        return "train_pairs.txt"
    return f"test_pairs_{test_order}.txt"


def _ensure_image_ext(name: str) -> str:
    base, ext = os.path.splitext(name)
    return name if ext else f"{base}.jpg"


def _normalize_pair_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _split_category_prefix(path: str) -> Tuple[Optional[str], str]:
    normalized = _normalize_pair_path(path)
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return None, ""
    if parts[0] in DRESSCODE_CATEGORIES:
        category = parts[0]
        rest = parts[1:]
        if rest and rest[0] == "images":
            rest = rest[1:]
        return category, "/".join(rest)
    return None, normalized


def _strip_images_prefix(path: str) -> str:
    normalized = _normalize_pair_path(path)
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[0] in ("images", "image"):
        parts = parts[1:]
    return "/".join(parts)


def _load_pairs(
    pairs_path: str,
    expected_category: Optional[str] = None,
    allow_category_prefix: bool = False,
    restrict_categories: Optional[Iterable[str]] = None,
    require_category_prefix: bool = False,
) -> List[Dict[str, str]]:
    allowed_categories = set(restrict_categories or [])
    pairs: List[Dict[str, str]] = []
    with open(pairs_path, "r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                log(f"[WARN] Invalid line {line_num} in {pairs_path}: {line}")
                continue

            person_token, cloth_token = parts[0], parts[1]
            if allow_category_prefix:
                person_cat, person_path = _split_category_prefix(person_token)
                cloth_cat, cloth_path = _split_category_prefix(cloth_token)
            else:
                person_cat, person_path = None, _normalize_pair_path(person_token)
                cloth_cat, cloth_path = None, _normalize_pair_path(cloth_token)

            if person_cat and cloth_cat and person_cat != cloth_cat:
                raise RuntimeError(
                    f"Category mismatch in {pairs_path} line {line_num}: {person_cat} vs {cloth_cat}"
                )

            line_category = person_cat or cloth_cat
            if line_category is None:
                if require_category_prefix:
                    raise RuntimeError(
                        f"Root-level pairs file lacks category prefix at line {line_num} in {pairs_path}."
                    )
                line_category = expected_category

            if line_category is None:
                raise RuntimeError(f"Unable to determine category for line {line_num} in {pairs_path}.")

            if expected_category and line_category != expected_category:
                raise RuntimeError(
                    f"Unexpected category '{line_category}' in {pairs_path} line {line_num}; "
                    f"expected '{expected_category}'."
                )

            if allowed_categories and line_category not in allowed_categories:
                continue

            person_file = Path(_ensure_image_ext(_strip_images_prefix(person_path))).name
            cloth_file = Path(_ensure_image_ext(_strip_images_prefix(cloth_path))).name
            if not person_file or not cloth_file:
                log(f"[WARN] Invalid paths in {pairs_path} line {line_num}: {line}")
                continue

            pairs.append(
                {
                    "image_file": person_file,
                    "cloth_file": cloth_file,
                    "category": line_category,
                }
            )
    return pairs


def load_dresscode_pairs(root: str, split: str, category: str, test_order: str) -> List[Dict[str, str]]:
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split}")
    if test_order not in VALID_TEST_ORDERS:
        raise ValueError(f"Invalid test_order: {test_order}")
    if category not in ["all"] + DRESSCODE_CATEGORIES:
        raise ValueError(f"Invalid category: {category}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"DressCode root not found: {root}")

    categories = DRESSCODE_CATEGORIES if category == "all" else [category]
    pairs_filename = _pairs_filename(split, test_order)
    data: List[Dict[str, str]] = []
    missing_categories: List[str] = []

    for cat in categories:
        pairs_path = os.path.join(root, cat, pairs_filename)
        if os.path.exists(pairs_path):
            data.extend(
                _load_pairs(
                    pairs_path,
                    expected_category=cat,
                    allow_category_prefix=True,
                    restrict_categories=None,
                    require_category_prefix=False,
                )
            )
        else:
            missing_categories.append(cat)

    if missing_categories:
        root_pairs_path = os.path.join(root, pairs_filename)
        if not os.path.exists(root_pairs_path):
            missing_str = ", ".join(missing_categories)
            raise FileNotFoundError(
                "Missing pairs for categories: "
                f"{missing_str}. Expected {os.path.join(root, '<category>', pairs_filename)} "
                f"or fallback {root_pairs_path}."
            )

        fallback_pairs = _load_pairs(
            root_pairs_path,
            expected_category=None,
            allow_category_prefix=True,
            restrict_categories=missing_categories,
            require_category_prefix=True,
        )
        if not fallback_pairs:
            raise RuntimeError("Root-level pairs file did not provide entries for required categories.")

        loaded_fallback = {item["category"] for item in fallback_pairs}
        still_missing = [cat for cat in missing_categories if cat not in loaded_fallback]
        if still_missing:
            raise RuntimeError("Root-level pairs file missing category prefixes for required categories.")
        data.extend(fallback_pairs)

    if not data:
        raise RuntimeError("No pairs loaded from DressCode dataset.")
    return data


def resolve_prompt(item: Dict[str, Any], prompt_override: Optional[str] = None) -> str:
    if prompt_override and str(prompt_override).strip():
        return str(prompt_override).strip()
    for key in ("text", "item_text"):
        value = item.get(key)
        if isinstance(value, list):
            for candidate in value:
                if candidate is not None and str(candidate).strip():
                    return str(candidate).strip()
        elif value is not None and str(value).strip():
            return str(value).strip()
    return FALLBACK_PROMPT


def load_subset_json(path: str, default_category: Optional[str]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise RuntimeError(f"subset_json must contain a JSON list: {path}")

    items: List[Dict[str, Any]] = []
    for idx, raw_item in enumerate(payload):
        if not isinstance(raw_item, dict):
            raise RuntimeError(f"subset_json item {idx} is not an object.")
        image_file = raw_item.get("image_file")
        cloth_file = raw_item.get("cloth_file")
        category = raw_item.get("category") or default_category
        if not image_file or not cloth_file:
            raise RuntimeError(f"subset_json item {idx} is missing image_file or cloth_file.")
        if not category:
            raise RuntimeError(f"subset_json item {idx} is missing category.")
        item = dict(raw_item)
        item["image_file"] = Path(str(image_file).replace("\\", "/")).name
        item["cloth_file"] = Path(str(cloth_file).replace("\\", "/")).name
        item["category"] = str(category)
        items.append(item)
    return items


def allocate_balanced_counts(total: int) -> Dict[str, int]:
    total = max(0, int(total))
    base = total // len(DRESSCODE_CATEGORIES)
    remainder = total % len(DRESSCODE_CATEGORIES)
    allocation = {}
    for idx, category in enumerate(DRESSCODE_CATEGORIES):
        allocation[category] = base + (1 if idx < remainder else 0)
    return allocation


def select_balanced_subset(
    items: Sequence[Dict[str, Any]],
    run_mode: str,
    max_samples: int,
    seed: int,
    use_all_summary_items: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if run_mode == "summary" and use_all_summary_items:
        selected = [dict(item) for item in items]
        counts = {category: sum(1 for item in selected if item.get("category") == category) for category in DRESSCODE_CATEGORIES}
        return selected, {"selection_mode": "all_items", "category_counts": counts, "num_selected": len(selected)}

    default_total = DEFAULT_FIGURE_TOTAL if run_mode == "figure" else DEFAULT_SUMMARY_TOTAL
    requested_total = int(max_samples) if int(max_samples) > 0 else default_total
    allocation = allocate_balanced_counts(requested_total)
    rng = random.Random(seed)
    buckets: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {category: [] for category in DRESSCODE_CATEGORIES}
    for idx, item in enumerate(items):
        category = item.get("category")
        if category in buckets:
            buckets[category].append((idx, item))

    selected_pairs: List[Tuple[int, Dict[str, Any]]] = []
    actual_counts: Dict[str, int] = {}
    for category in DRESSCODE_CATEGORIES:
        pool = buckets[category]
        need = allocation.get(category, 0)
        if len(pool) <= need:
            chosen = list(pool)
        else:
            indices = sorted(rng.sample(range(len(pool)), need))
            chosen = [pool[i] for i in indices]
        selected_pairs.extend(chosen)
        actual_counts[category] = len(chosen)

    selected_pairs.sort(key=lambda pair: pair[0])
    selected = [dict(item) for _, item in selected_pairs]
    return selected, {
        "selection_mode": "balanced",
        "requested_total": requested_total,
        "requested_per_category": allocation,
        "category_counts": actual_counts,
        "num_selected": len(selected),
    }


def resolve_subset(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    subset_path = args.subset_json or DEFAULT_SUBSET_JSON
    default_category = None if args.dresscode_category == "all" else args.dresscode_category
    if subset_path and os.path.exists(subset_path):
        all_items = load_subset_json(subset_path, default_category=default_category)
        source = {"type": "subset_json", "path": Path(subset_path).as_posix(), "total_items": len(all_items)}
    else:
        if not args.dresscode_root:
            raise RuntimeError("--dresscode_root is required when subset_json is unavailable.")
        all_items = load_dresscode_pairs(
            root=args.dresscode_root,
            split="test",
            category=args.dresscode_category,
            test_order=args.dresscode_test_order,
        )
        source = {"type": "dataset", "path": "", "total_items": len(all_items)}

    selected_items, selection_info = select_balanced_subset(
        items=all_items,
        run_mode=args.run_mode,
        max_samples=args.max_samples,
        seed=args.seed,
        use_all_summary_items=bool(args.use_all_summary_items),
    )
    if not selected_items:
        raise RuntimeError("Resolved subset is empty.")
    return selected_items, {"source": source, "selection": selection_info}


def mode_to_ablation_mode(s_mode: str) -> str:
    mapping = {"S0": "A0", "S1": "A1", "S2": "A2"}
    normalized = (s_mode or "").upper()
    if normalized not in mapping:
        raise ValueError(f"Unsupported s_mode: {s_mode}")
    return mapping[normalized]


def build_ablation_config(mode: str) -> AblationConfig:
    normalized = (mode or "A2").upper()
    if normalized == "A0":
        return AblationConfig("A0", False, False, False, "cross_attn")
    if normalized == "A1":
        return AblationConfig("A1", True, False, True, "frozen")
    if normalized == "A2":
        return AblationConfig("A2", True, True, False, "frozen")
    raise ValueError(f"Unknown ablation mode: {mode}")


def _get_autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def resolve_device_and_dtype(requested_device: str, requested_dtype: str) -> Tuple[torch.device, torch.dtype, str, List[str]]:
    warnings: List[str] = []
    device_str = (requested_device or "auto").strip().lower()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        candidate = torch.device(requested_device)
        if candidate.type == "cuda" and not torch.cuda.is_available():
            warnings.append("CUDA requested but unavailable; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = candidate

    aliases = {
        "auto": "auto",
        "fp32": "fp32",
        "float32": "fp32",
        "fp16": "fp16",
        "float16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    key = aliases.get((requested_dtype or "auto").strip().lower())
    if key is None:
        raise ValueError(f"Unsupported dtype: {requested_dtype}")
    if key == "auto":
        key = "fp16" if device.type == "cuda" else "fp32"
    if device.type == "cpu" and key != "fp32":
        warnings.append(f"dtype={key} is not supported on CPU for this script; using fp32.")
        key = "fp32"
    if device.type == "cuda" and key == "bf16" and not torch.cuda.is_bf16_supported():
        warnings.append("CUDA bf16 requested but unsupported on this GPU; using fp16.")
        key = "fp16"
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    return device, dtype_map[key], key, warnings


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=str(device))
    except Exception:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _build_scheduler(name: str, timestep_spacing: str):
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    base_kwargs = dict(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        rescale_betas_zero_snr=True,
        prediction_type="epsilon",
    )
    try:
        base = DDIMScheduler(timestep_spacing=timestep_spacing, **base_kwargs)
    except TypeError:
        base = DDIMScheduler(**base_kwargs)
        if hasattr(base, "config"):
            base.config.timestep_spacing = timestep_spacing

    normalized = name.lower()
    if normalized == "ddim":
        return base
    if normalized == "unipc":
        return UniPCMultistepScheduler.from_config(base.config)
    if normalized == "dpmpp_2m":
        return DPMSolverMultistepScheduler.from_config(
            base.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=False,
        )
    if normalized == "dpmpp_2m_karras":
        return DPMSolverMultistepScheduler.from_config(
            base.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    if normalized == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(base.config)
    raise ValueError(f"Unsupported scheduler: {name}")


def _set_scheduler_timesteps(scheduler, num_inference_steps: int, device: torch.device) -> None:
    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(num_inference_steps)


def as_timestep_tensor(value: Any, device: torch.device, batch_size: int) -> torch.Tensor:
    tensor = value.to(device=device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.flatten()
    if tensor.is_floating_point():
        tensor = torch.round(tensor)
    tensor = tensor.to(dtype=torch.long)
    if tensor.numel() == 1 and batch_size != 1:
        tensor = tensor.expand(batch_size)
    elif tensor.numel() != batch_size:
        tensor = tensor[:1].expand(batch_size)
    return tensor


def _timesteps_to_list(values: Any) -> List[int]:
    if values is None:
        return []
    tensor = values.detach().cpu() if isinstance(values, torch.Tensor) else torch.tensor(values)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    return [int(v) for v in tensor.flatten().tolist()]


def _select_start_index(values: Sequence[int], target_t: Optional[int]) -> int:
    if target_t is None:
        return 0
    target = float(target_t)
    return min(range(len(values)), key=lambda idx: abs(float(values[idx]) - target))


def resolve_scheduler_timesteps(
    scheduler_name: str,
    timestep_spacing: str,
    num_inference_steps: int,
    device: torch.device,
    init_mode: str,
    reconstruct_t: Optional[int],
):
    scheduler = _build_scheduler(scheduler_name, timestep_spacing)
    _set_scheduler_timesteps(scheduler, num_inference_steps, device)
    timesteps = scheduler.timesteps
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
    timesteps = timesteps.to(device)
    timestep_list = _timesteps_to_list(timesteps)
    if not timestep_list:
        raise RuntimeError("scheduler.timesteps is empty after set_timesteps")

    if init_mode == "from_noisy_gt":
        start_idx = _select_start_index(timestep_list, reconstruct_t)
        run_timesteps = timesteps[start_idx:]
        if int(run_timesteps.numel()) == 0:
            start_idx = 0
            run_timesteps = timesteps
    else:
        start_idx = 0
        run_timesteps = timesteps

    if hasattr(scheduler, "set_begin_index"):
        try:
            scheduler.set_begin_index(start_idx)
        except Exception as exc:
            log(f"[WARN] scheduler.set_begin_index failed: {exc}")
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    return scheduler, timesteps, run_timesteps, start_idx


def parse_csv_list(raw: str) -> List[str]:
    values = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if token:
            values.append(token)
    return values


class VisualizationSDModel(nn.Module):
    def __init__(self, unet: nn.Module, ref_unet: nn.Module, proj: nn.Module, ablation_config: AblationConfig):
        super().__init__()
        self.unet = unet
        self.ref_unet = ref_unet
        self.proj = proj
        self.ablation_mode = ablation_config.mode
        self.enable_ieb = ablation_config.enable_ieb
        self.enable_ha = ablation_config.enable_ha
        self.direct_ieb = ablation_config.direct_ieb
        self.unet_train_mode = ablation_config.unet_train

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        latents: torch.Tensor,
        ref_latents: torch.Tensor,
        clip_image_embeddings: Optional[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
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
                _ = self.ref_unet(
                    ref_latents,
                    ref_timesteps,
                    encoder_hidden_states=person_proj_embed,
                    return_dict=False,
                )

            sa_hidden_states = {}
            for name in self.ref_unet.attn_processors.keys():
                processor = self.ref_unet.attn_processors[name]
                if hasattr(processor, "cache") and "hidden_states" in processor.cache:
                    value = processor.cache["hidden_states"]
                    if torch.is_tensor(value) and value.dtype != amp_dtype:
                        value = value.to(dtype=amp_dtype)
                    sa_hidden_states[name] = value
                else:
                    raise RuntimeError(f"Attention cache missing for {name}")
            _clear_ref_unet_cache(self.ref_unet)

        if self.direct_ieb and person_proj_embed is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states, person_proj_embed], dim=1)

        cross_kwargs = {"sa_hidden_states": sa_hidden_states} if self.enable_ha else None
        with _get_autocast_context(latents.device, amp_dtype):
            noise_pred = self.unet(
                latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_kwargs,
            ).sample

        if isinstance(sa_hidden_states, dict):
            sa_hidden_states.clear()
        return noise_pred


def _clear_nested_refs(obj: Any) -> None:
    if obj is None or torch.is_tensor(obj):
        return
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            _clear_nested_refs(obj[key])
            obj[key] = None
        obj.clear()
        return
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            _clear_nested_refs(item)
        if isinstance(obj, (list, set)):
            obj.clear()


def _clear_ref_unet_cache(ref_unet: Optional[nn.Module]) -> None:
    if ref_unet is None:
        return
    for processor in ref_unet.attn_processors.values():
        if hasattr(processor, "clear_cache") and callable(processor.clear_cache):
            try:
                processor.clear_cache()
                continue
            except Exception:
                pass
        if hasattr(processor, "cache"):
            _clear_nested_refs(processor.cache)


def _clear_unet_extra_state(unet: Optional[nn.Module]) -> None:
    if unet is None:
        return
    for processor in unet.attn_processors.values():
        for attr_name in (
            "sa_hidden_states",
            "_sa_hidden_states",
            "_last_sa_hidden_states",
            "last_sa_hidden_states",
            "cached_sa_hidden_states",
        ):
            if hasattr(processor, attr_name):
                try:
                    setattr(processor, attr_name, None)
                except Exception:
                    pass
        if hasattr(processor, "cache"):
            _clear_nested_refs(processor.cache)


def build_unet_attention_processors(unet: UNet2DConditionModel, ablation_config: AblationConfig) -> None:
    ensure_runtime_imports()
    attn_procs: Dict[str, nn.Module] = {}
    unet_state = unet.state_dict() if ablation_config.enable_ha else None
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
        else:
            raise RuntimeError(f"Unexpected attention processor name: {name}")

        if cross_attention_dim is None:
            if ablation_config.enable_ha:
                processor = RefSAttnProcessor2_0(name, hidden_size)
                layer_name = name.split(".processor")[0]
                processor.load_state_dict(
                    {
                        "to_k_ref.weight": unet_state[layer_name + ".to_k.weight"],
                        "to_v_ref.weight": unet_state[layer_name + ".to_v.weight"],
                    }
                )
                attn_procs[name] = processor
            else:
                attn_procs[name] = SAttnProcessor2_0(name, hidden_size)
        else:
            if ablation_config.enable_ha:
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
    if unet_state is not None:
        del unet_state


def create_image_proj(unet: UNet2DConditionModel, image_encoder: CLIPVisionModelWithProjection) -> Resampler:
    ensure_runtime_imports()
    return Resampler(
        dim=unet.config.cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=16,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    )


def load_visualization_checkpoint(checkpoint_path: str, model: VisualizationSDModel) -> Tuple[Dict[str, Any], List[str]]:
    ensure_runtime_imports()
    if not checkpoint_path:
        raise FileNotFoundError("--checkpoint is required.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise CheckpointLoadError(f"Checkpoint must contain a dict payload: {checkpoint_path}")

    warnings: List[str] = []

    def load_state(module: Optional[nn.Module], key: str, label: str) -> None:
        if module is None:
            warnings.append(f"[CKPT] Skip {label}: module not available.")
            return
        state = ckpt.get(key)
        if state is None:
            warnings.append(f"[CKPT] Missing {key}; keeping {label} initialization.")
            return
        try:
            result = module.load_state_dict(state, strict=False)
        except Exception as exc:
            raise CheckpointLoadError(f"[CKPT] Failed to load {label}: {exc}") from exc
        if getattr(result, "missing_keys", None):
            warnings.append(f"[CKPT] {label} missing_keys={len(result.missing_keys)}")
        if getattr(result, "unexpected_keys", None):
            warnings.append(f"[CKPT] {label} unexpected_keys={len(result.unexpected_keys)}")

    load_state(model.proj, "image_proj", "image_proj")
    load_state(model.ref_unet, "ref_unet", "ref_unet")
    if "unet" in ckpt:
        load_state(model.unet, "unet", "unet")
    else:
        warnings.append("[CKPT] Missing unet weights; using pretrained UNet initialization.")

    cross_payload = ckpt.get("cross_attn_processors")
    expected_refc = [
        name for name, processor in model.unet.attn_processors.items() if isinstance(processor, RefCAttnProcessor2_0)
    ]
    if cross_payload is None:
        if expected_refc:
            warnings.append("[CKPT] Missing cross_attn_processors; using initialized RefCAttnProcessor2_0 weights.")
    else:
        loaded = 0
        for name, state in cross_payload.items():
            processor = model.unet.attn_processors.get(name)
            if processor is None or not isinstance(processor, RefCAttnProcessor2_0):
                continue
            try:
                processor.load_state_dict(state, strict=False)
                loaded += 1
            except Exception as exc:
                warnings.append(f"[CKPT] Failed to load cross_attn {name}: {exc}")
        if expected_refc and loaded == 0:
            warnings.append("[CKPT] No RefCAttnProcessor2_0 weights loaded (mode mismatch possible).")
    return ckpt, warnings


def move_modules_to_device(modules: Sequence[nn.Module], device: torch.device, dtype: torch.dtype) -> None:
    for module in modules:
        module.to(device=device, dtype=dtype)
        module.eval()


def build_visualization_bundle(
    args: argparse.Namespace,
    ablation_config: AblationConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> Dict[str, Any]:
    ensure_runtime_imports()
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    image_proj = create_image_proj(unet, image_encoder)
    build_unet_attention_processors(unet, ablation_config)

    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    ref_unet.set_attn_processor({name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_proj.requires_grad_(ablation_config.enable_ieb)
    ref_unet.requires_grad_(ablation_config.enable_ha)

    if ablation_config.enable_ha:
        for processor in unet.attn_processors.values():
            if isinstance(processor, RefSAttnProcessor2_0):
                processor.requires_grad_(False)
                processor.eval()
            elif isinstance(processor, RefCAttnProcessor2_0):
                processor.requires_grad_(True)

    model = VisualizationSDModel(unet=unet, ref_unet=ref_unet, proj=image_proj, ablation_config=ablation_config)
    checkpoint_payload, checkpoint_warnings = load_visualization_checkpoint(args.checkpoint, model)

    move_modules_to_device([model, vae, text_encoder, image_encoder], device=device, dtype=weight_dtype)

    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 0.18215)
    vae_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    clip_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "image_encoder": image_encoder,
        "vae": vae,
        "model": model,
        "scaling_factor": scaling_factor,
        "vae_transform": vae_transform,
        "clip_transform": clip_transform,
        "checkpoint_payload": checkpoint_payload,
        "checkpoint_warnings": checkpoint_warnings,
    }


def discover_unet_layers(unet: nn.Module, s_mode: str) -> List[Dict[str, Any]]:
    ensure_runtime_imports()
    records: List[Dict[str, Any]] = []
    for name, processor in unet.attn_processors.items():
        class_name = processor.__class__.__name__
        if name.startswith("down_blocks"):
            group = "down"
            block_id = int(name[len("down_blocks.")])
        elif name.startswith("mid_block"):
            group = "mid"
            block_id = None
        elif name.startswith("up_blocks"):
            group = "up"
            block_id = int(name[len("up_blocks.")])
        else:
            group = "other"
            block_id = None
        capture_mode = "attention_probs" if class_name in {"RefCAttnProcessor2_0", "CAttnProcessor2_0"} else "proxy_response"
        preferred = (s_mode == "S2" and class_name == "RefCAttnProcessor2_0") or (
            s_mode == "S1" and class_name == "CAttnProcessor2_0"
        )
        records.append(
            {
                "name": name,
                "processor_type": class_name,
                "group": group,
                "block_id": block_id,
                "capture_mode_supported": capture_mode,
                "preferred": preferred,
                "chosen": False,
            }
        )
    return records


def select_layers(layer_records: List[Dict[str, Any]], raw_selected_layers: str, s_mode: str) -> List[str]:
    preferred_type = "RefCAttnProcessor2_0" if s_mode == "S2" else "CAttnProcessor2_0"
    names = [record["name"] for record in layer_records]
    selected: List[str] = []

    patterns = parse_csv_list(raw_selected_layers)
    if patterns:
        for pattern in patterns:
            exact_matches = [name for name in names if name == pattern]
            partial_matches = [name for name in names if pattern in name] if not exact_matches else []
            matches = exact_matches or partial_matches
            if not matches:
                log(f"[WARN] selected_layers pattern matched nothing: {pattern}")
                continue
            for name in matches:
                if name not in selected:
                    selected.append(name)
    else:
        preferred_records = [record for record in layer_records if record["processor_type"] == preferred_type]
        if s_mode == "S2" and not preferred_records:
            raise RuntimeError("No RefCAttnProcessor2_0 layers found for S2 visualization.")
        if s_mode == "S1" and not preferred_records:
            raise RuntimeError("No CAttnProcessor2_0 layers found for S1 visualization.")

        for group in ("down", "mid", "up"):
            group_records = [record for record in preferred_records if record["group"] == group]
            if not group_records:
                continue
            if group == "down":
                group_records.sort(key=lambda record: (-(record["block_id"] or 0), record["name"]))
            elif group == "up":
                group_records.sort(key=lambda record: ((record["block_id"] or 999), record["name"]))
            else:
                group_records.sort(key=lambda record: record["name"])
            chosen = group_records[0]["name"]
            if chosen not in selected:
                selected.append(chosen)

        if len(selected) < 3:
            for record in preferred_records:
                if record["name"] not in selected:
                    selected.append(record["name"])
                if len(selected) >= 3:
                    break

    for record in layer_records:
        record["chosen"] = record["name"] in selected
    if not selected:
        raise RuntimeError("No layers selected for visualization.")
    return selected


def select_timestep_targets(run_timesteps: torch.Tensor, raw_selected_timesteps: str) -> List[TimestepTarget]:
    values = _timesteps_to_list(run_timesteps)
    if not values:
        raise RuntimeError("No run timesteps available for visualization.")

    indices: List[int] = []
    tokens = parse_csv_list(raw_selected_timesteps)
    if not tokens:
        indices = [0, len(values) // 2, len(values) - 1]
    else:
        for token in tokens:
            if token.lower().startswith("idx:"):
                idx = int(token.split(":", 1)[1])
            else:
                raw_value = int(token)
                if raw_value in values:
                    idx = values.index(raw_value)
                elif 0 <= raw_value < len(values):
                    idx = raw_value
                else:
                    idx = min(range(len(values)), key=lambda i: abs(values[i] - raw_value))
                    log(f"[WARN] timestep {raw_value} not exact; using nearest scheduler timestep {values[idx]}.")
            idx = max(0, min(idx, len(values) - 1))
            if idx not in indices:
                indices.append(idx)

    if not indices:
        raise RuntimeError("No visualization timesteps selected.")

    default_labels = ["early", "mid", "late"] if len(indices) == 3 else [f"t{i}" for i in range(len(indices))]
    targets = []
    for pos, idx in enumerate(indices):
        label = default_labels[pos] if pos < len(default_labels) else f"t{pos}"
        targets.append(TimestepTarget(index=idx, value=int(values[idx]), label=label))
    return targets


def infer_spatial_hw(token_count: int, explicit_hw: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    if explicit_hw and explicit_hw[0] * explicit_hw[1] == token_count:
        return explicit_hw
    side = int(math.isqrt(max(1, int(token_count))))
    if side * side == token_count:
        return side, side
    return 1, max(1, int(token_count))


def resize_array(array: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    resized = pil.resize((size[1], size[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def normalize_map(raw_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_map, dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.maximum(raw, 0.0)
    total = float(raw.sum())
    if total <= 1e-12:
        prob = np.full(raw.shape, 1.0 / max(1, raw.size), dtype=np.float32)
    else:
        prob = raw / total
    raw_min = float(raw.min()) if raw.size else 0.0
    raw_max = float(raw.max()) if raw.size else 0.0
    if raw_max - raw_min <= 1e-12:
        norm01 = np.zeros_like(raw, dtype=np.float32)
    else:
        norm01 = (raw - raw_min) / (raw_max - raw_min)
    return prob.astype(np.float32), norm01.astype(np.float32)


def compute_map_metrics(raw_map: np.ndarray) -> Dict[str, float]:
    prob, norm01 = normalize_map(raw_map)
    flat_prob = prob.reshape(-1)
    entropy = 0.0
    if flat_prob.size > 1:
        entropy = float(-(flat_prob * np.log(np.clip(flat_prob, 1e-12, 1.0))).sum() / math.log(flat_prob.size))
    top_sorted = np.sort(flat_prob)[::-1]
    top5_mass = float(top_sorted[: min(5, top_sorted.size)].sum())
    top10_mass = float(top_sorted[: min(10, top_sorted.size)].sum())
    peak_value = float(flat_prob.max()) if flat_prob.size else 0.0
    area_above_05 = float((norm01 >= 0.5).mean()) if norm01.size else 0.0

    height, width = prob.shape
    if prob.size:
        xs = np.linspace(0.0, 1.0, num=width, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
        center_x = float((prob.sum(axis=0) * xs).sum())
        center_y = float((prob.sum(axis=1) * ys).sum())
    else:
        center_x = 0.5
        center_y = 0.5

    return {
        "entropy": round_float(entropy),
        "top5_mass": round_float(top5_mass),
        "top10_mass": round_float(top10_mass),
        "peak_value": round_float(peak_value),
        "area_above_05": round_float(area_above_05),
        "center_x_norm": round_float(center_x),
        "center_y_norm": round_float(center_y),
    }


def make_heatmap_rgb(norm01: np.ndarray) -> np.ndarray:
    heat = np.asarray(norm01, dtype=np.float32)
    red = np.clip(heat * 1.2, 0.0, 1.0)
    green = np.clip(np.sqrt(np.clip(heat, 0.0, 1.0)), 0.0, 1.0)
    blue = np.clip(1.0 - heat, 0.0, 1.0) * 0.25
    return np.stack([red, green, blue], axis=-1)


def overlay_heatmap(base_image: Image.Image, norm01_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = base_image.convert("RGB")
    heat = resize_array(norm01_map, (base.height, base.width))
    heat_rgb = make_heatmap_rgb(heat)
    base_np = np.asarray(base).astype(np.float32) / 255.0
    overlay_np = np.clip((1.0 - alpha) * base_np + alpha * heat_rgb, 0.0, 1.0)
    return Image.fromarray((overlay_np * 255.0).astype(np.uint8))


def grayscale_heatmap(norm01_map: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    resized = resize_array(norm01_map, size)
    return Image.fromarray(np.clip(resized * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")


def tensor_image_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu()
    image = (image + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)
    image_np = (image[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_np)


def build_panel_image(
    tiles: Sequence[Tuple[str, Image.Image]],
    cell_size: int = DEFAULT_PANEL_CELL,
    columns: int = 4,
) -> Image.Image:
    prepared: List[Tuple[str, Image.Image]] = []
    for title, image in tiles:
        canvas = Image.new("RGB", (cell_size, cell_size + 28), color=(255, 255, 255))
        fitted = image.convert("RGB").resize((cell_size, cell_size), resample=Image.Resampling.BILINEAR)
        canvas.paste(fitted, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, cell_size + 6), title[:48], fill=(0, 0, 0))
        prepared.append((title, canvas))

    if not prepared:
        return Image.new("RGB", (cell_size, cell_size), color=(255, 255, 255))

    columns = max(1, int(columns))
    rows = math.ceil(len(prepared) / columns)
    panel = Image.new("RGB", (columns * cell_size, rows * (cell_size + 28)), color=(245, 245, 245))
    for idx, (_, canvas) in enumerate(prepared):
        x = (idx % columns) * cell_size
        y = (idx // columns) * (cell_size + 28)
        panel.paste(canvas, (x, y))
    return panel


def pearson_correlation(map_a: np.ndarray, map_b: np.ndarray) -> Optional[float]:
    a = np.asarray(map_a, dtype=np.float32).reshape(-1)
    b = np.asarray(map_b, dtype=np.float32).reshape(-1)
    if a.size != b.size or a.size == 0:
        return None
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std <= 1e-12 and b_std <= 1e-12:
        return 1.0
    if a_std <= 1e-12 or b_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def center_from_prob(prob: np.ndarray) -> Tuple[float, float]:
    metrics = compute_map_metrics(prob)
    return float(metrics["center_x_norm"]), float(metrics["center_y_norm"])


class CaptureSession:
    def __init__(self, sample_index: int, selected_layers: Sequence[str], timestep_targets: Sequence[TimestepTarget]):
        self.sample_index = sample_index
        self.selected_layers = set(selected_layers)
        self.targets_by_index = {target.index: target for target in timestep_targets}
        self.current_target: Optional[TimestepTarget] = None
        self.capture_enabled = False
        self.records: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.warnings: List[str] = []
        self._warning_keys: set = set()

    def begin_step(self, step_index: int) -> None:
        self.current_target = self.targets_by_index.get(step_index)
        self.capture_enabled = self.current_target is not None

    def end_step(self) -> None:
        self.capture_enabled = False

    def should_capture(self, layer_name: str) -> bool:
        return self.capture_enabled and self.current_target is not None and layer_name in self.selected_layers

    def warn_once(self, key: str, message: str) -> None:
        if key in self._warning_keys:
            return
        self._warning_keys.add(key)
        self.warnings.append(message)
        log(f"[WARN] {message}")

    def record_map(
        self,
        layer_name: str,
        processor_type: str,
        map_type: str,
        branch: str,
        aggregation: str,
        raw_map: np.ndarray,
        spatial_hw: Tuple[int, int],
        note: str = "",
    ) -> None:
        if self.current_target is None:
            return
        key = (layer_name, self.current_target.index)
        if key in self.records:
            return
        self.records[key] = {
            "layer_name": layer_name,
            "processor_type": processor_type,
            "map_type": map_type,
            "branch": branch,
            "aggregation": aggregation,
            "note": note,
            "timestep_index": self.current_target.index,
            "timestep_value": self.current_target.value,
            "timestep_label": self.current_target.label,
            "raw_map": np.asarray(raw_map, dtype=np.float32),
            "spatial_hw": tuple(int(v) for v in spatial_hw),
        }


class AttentionCaptureWrapper(nn.Module):
    def __init__(self, layer_name: str, processor: Any, manager: "AttentionCaptureManager"):
        super().__init__()
        self.layer_name = layer_name
        self.processor = processor
        self.manager = manager

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        cond_hidden_states=None,
        sa_hidden_states=None,
    ) -> torch.Tensor:
        session = self.manager.active_session
        if session is None or not session.should_capture(self.layer_name):
            return self.processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                scale=scale,
                cond_hidden_states=cond_hidden_states,
                sa_hidden_states=sa_hidden_states,
            )

        processor_type = self.processor.__class__.__name__
        try:
            if isinstance(self.processor, RefCAttnProcessor2_0):
                return self._forward_ref_cattn(
                    attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale, sa_hidden_states, session
                )
            if isinstance(self.processor, CAttnProcessor2_0):
                return self._forward_cattn(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale, session)
        except Exception as exc:
            session.warn_once(f"{self.layer_name}:exact_error", f"{self.layer_name} exact capture failed: {exc}")
        output = self.processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            scale=scale,
            cond_hidden_states=cond_hidden_states,
            sa_hidden_states=sa_hidden_states,
        )
        self._capture_proxy(session, self.layer_name, hidden_states, output, processor_type, "fallback_proxy")
        return output

    @staticmethod
    def _prepare_attention_mask(attn, attention_mask, sequence_length, batch_size):
        if attention_mask is None:
            return None
        prepared = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        return prepared.view(batch_size, attn.heads, -1, prepared.shape[-1])

    @staticmethod
    def _compute_attention_probs(query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        total = int(query.shape[0]) * int(query.shape[1]) * int(query.shape[2]) * int(key.shape[2])
        if total > MAX_TRUE_ATTENTION_ELEMENTS:
            return None
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(query.shape[-1])
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                scores = scores + attention_mask.float()
        return torch.softmax(scores, dim=-1)


    @staticmethod
    def _aggregate_probs_to_map(probs: torch.Tensor, explicit_hw: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, Tuple[int, int]]:
        query_map = probs.max(dim=-1).values.mean(dim=1)[0].detach().cpu().float().numpy()
        hw = infer_spatial_hw(query_map.size, explicit_hw=explicit_hw)
        if hw[0] * hw[1] == query_map.size:
            query_map = query_map.reshape(hw)
        else:
            query_map = query_map.reshape(1, -1)
            hw = query_map.shape
        return query_map.astype(np.float32), (int(hw[0]), int(hw[1]))


    @staticmethod
    def _capture_proxy(
        session: CaptureSession,
        layer_name: str,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        processor_type: str,
        note: str,
    ) -> None:
        residual = hidden_states.to(dtype=output.dtype)
        diff = (output - residual).detach()
        if diff.ndim == 4:
            raw_map = diff.abs().mean(dim=1)[0].cpu().float().numpy()
            hw = (raw_map.shape[0], raw_map.shape[1])
        else:
            vector = diff.abs().mean(dim=-1)[0].cpu().float().numpy()
            hw = infer_spatial_hw(vector.size, explicit_hw=None)
            raw_map = vector.reshape(hw) if hw[0] * hw[1] == vector.size else vector.reshape(1, -1)
            hw = (raw_map.shape[0], raw_map.shape[1])
        session.record_map(
            layer_name=layer_name,
            processor_type=processor_type,
            map_type="proxy_response",
            branch="output_delta",
            aggregation="channel_mean_abs",
            raw_map=raw_map,
            spatial_hw=hw,
            note=note,
        )

    def _forward_cattn(self, attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale, session):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        explicit_hw = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            explicit_hw = (height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = self._prepare_attention_mask(attn, attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        probs = self._compute_attention_probs(query, key, attention_mask)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        if probs is None:
            self._capture_proxy(
                session,
                self.layer_name,
                residual,
                hidden_states,
                self.processor.__class__.__name__,
                "true_attention_too_large",
            )
        else:
            raw_map, spatial_hw = self._aggregate_probs_to_map(probs, explicit_hw=explicit_hw)
            session.record_map(
                layer_name=self.layer_name,
                processor_type=self.processor.__class__.__name__,
                map_type="attention_probs",
                branch="cross_branch",
                aggregation="head_mean_token_max",
                raw_map=raw_map,
                spatial_hw=spatial_hw,
            )
        return hidden_states

    def _forward_ref_cattn(self, attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale, sa_hidden_states, session):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        explicit_hw = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            explicit_hw = (height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = self._prepare_attention_mask(attn, attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states_out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        branch = "cross_branch"
        map_type = "proxy_response"
        raw_map = None
        spatial_hw = explicit_hw
        note = ""

        if sa_hidden_states is not None and self.layer_name in sa_hidden_states:
            ref_hidden_states = sa_hidden_states[self.layer_name]
            if ref_hidden_states.ndim == 4:
                bsz, channel_ref, height_ref, width_ref = ref_hidden_states.shape
                ref_hidden_states = ref_hidden_states.view(bsz, channel_ref, height_ref * width_ref).transpose(1, 2)
            ref_hidden_states = ref_hidden_states.to(dtype=self.processor.to_k_ref.weight.dtype)
            ref_key = self.processor.to_k_ref(ref_hidden_states)
            ref_value = self.processor.to_v_ref(ref_hidden_states)
            ref_key = ref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_value = ref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_probs = self._compute_attention_probs(query, ref_key, attention_mask)
            if ref_probs is not None:
                raw_map, spatial_hw = self._aggregate_probs_to_map(ref_probs, explicit_hw=explicit_hw)
                branch = "ref_branch"
                map_type = "attention_probs"
            else:
                note = "true_attention_too_large"
            ref_out = F.scaled_dot_product_attention(
                query, ref_key, ref_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_out = hidden_states_out + ref_out * self.processor.scale

        hidden_states = hidden_states_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        if raw_map is not None:
            session.record_map(
                layer_name=self.layer_name,
                processor_type=self.processor.__class__.__name__,
                map_type=map_type,
                branch=branch,
                aggregation="head_mean_token_max",
                raw_map=raw_map,
                spatial_hw=spatial_hw or infer_spatial_hw(raw_map.size),
                note=note,
            )
        else:
            self._capture_proxy(
                session,
                self.layer_name,
                residual,
                hidden_states,
                self.processor.__class__.__name__,
                note or "fallback_proxy",
            )
        return hidden_states


class AttentionCaptureManager:
    def __init__(self, unet: nn.Module, selected_layers: Sequence[str]):
        self.unet = unet
        self.selected_layers = set(selected_layers)
        self.original_processors = None
        self.active_session: Optional[CaptureSession] = None

    def __enter__(self) -> "AttentionCaptureManager":
        self.original_processors = dict(self.unet.attn_processors)
        patched = {}
        for name, processor in self.original_processors.items():
            patched[name] = AttentionCaptureWrapper(name, processor, self) if name in self.selected_layers else processor
        self.unet.set_attn_processor(patched)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.original_processors is not None:
            self.unet.set_attn_processor(self.original_processors)
        self.active_session = None


def save_map_artifacts(
    output_root: Path,
    sample_dir: Path,
    person_image: Image.Image,
    generated_image: Image.Image,
    gt_image: Optional[Image.Image],
    captured_records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Path]]:
    map_rows: List[Dict[str, Any]] = []
    tiles: List[Tuple[str, Image.Image]] = [("person", person_image), ("generated", generated_image)]
    if gt_image is not None:
        tiles.append(("gt_cloth", gt_image))

    for record in captured_records:
        raw_map = np.asarray(record["raw_map"], dtype=np.float32)
        prob_map, norm01_map = normalize_map(raw_map)
        metrics = compute_map_metrics(raw_map)
        layer_token = sanitize_name(record["layer_name"])
        base_name = f"layer_{layer_token}_t_{record['timestep_value']}"
        raw_path = sample_dir / f"{base_name}_raw.png" if args.save_raw_maps else None
        overlay_person_path = sample_dir / f"{base_name}_overlay_person.png" if args.save_overlay_on_person else None
        overlay_output_path = sample_dir / f"{base_name}_overlay_output.png" if args.save_overlay_on_output else None

        if raw_path is not None:
            grayscale_heatmap(norm01_map, (args.image_size, args.image_size)).save(raw_path)
        if overlay_person_path is not None:
            overlay_person = overlay_heatmap(person_image, norm01_map)
            overlay_person.save(overlay_person_path)
            tiles.append((f"{record['timestep_label']} {layer_token}", overlay_person))
        else:
            tiles.append((f"{record['timestep_label']} {layer_token}", grayscale_heatmap(norm01_map, (args.image_size, args.image_size)).convert("RGB")))
        if overlay_output_path is not None:
            overlay_heatmap(generated_image, norm01_map).save(overlay_output_path)

        stage_prob = resize_array(prob_map, PROXY_RESIZE_HW)
        stage_total = float(stage_prob.sum())
        if stage_total > 1e-12:
            stage_prob = stage_prob / stage_total

        row = {
            "layer_name": record["layer_name"],
            "processor_type": record["processor_type"],
            "map_type": record["map_type"],
            "branch": record["branch"],
            "aggregation": record["aggregation"],
            "note": record.get("note", ""),
            "timestep_index": record["timestep_index"],
            "timestep_value": record["timestep_value"],
            "timestep_label": record["timestep_label"],
            "spatial_height": record["spatial_hw"][0],
            "spatial_width": record["spatial_hw"][1],
            "raw_path": rel_to(output_root, raw_path),
            "overlay_person_path": rel_to(output_root, overlay_person_path),
            "overlay_output_path": rel_to(output_root, overlay_output_path),
            "stage_prob_map": stage_prob,
        }
        row.update(metrics)
        map_rows.append(row)

    panel_path = sample_dir / "panel.png"
    build_panel_image(tiles).save(panel_path)
    return map_rows, tiles, panel_path


def compute_sample_aggregates(map_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    entropies = [float(row["entropy"]) for row in map_rows if row.get("entropy") is not None]
    top10 = [float(row["top10_mass"]) for row in map_rows if row.get("top10_mass") is not None]
    area = [float(row["area_above_05"]) for row in map_rows if row.get("area_above_05") is not None]

    by_stage: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in map_rows:
        if row.get("stage_prob_map") is not None:
            by_stage[row["timestep_label"]].append(np.asarray(row["stage_prob_map"], dtype=np.float32))

    stage_averages: Dict[str, np.ndarray] = {}
    for label, maps in by_stage.items():
        if maps:
            stage_averages[label] = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)

    early = stage_averages.get("early")
    mid = stage_averages.get("mid")
    late = stage_averages.get("late")
    early_mid = pearson_correlation(early, mid) if early is not None and mid is not None else None
    mid_late = pearson_correlation(mid, late) if mid is not None and late is not None else None
    center_shift = None
    if early is not None and late is not None:
        early_prob, _ = normalize_map(early)
        late_prob, _ = normalize_map(late)
        ex, ey = center_from_prob(early_prob)
        lx, ly = center_from_prob(late_prob)
        center_shift = math.sqrt((ex - lx) ** 2 + (ey - ly) ** 2)

    return {
        "num_maps": len(map_rows),
        "mean_entropy": round_float(safe_mean(entropies)),
        "mean_top10_mass": round_float(safe_mean(top10)),
        "mean_area_above_05": round_float(safe_mean(area)),
        "early_mid_correlation": round_float(early_mid),
        "mid_late_correlation": round_float(mid_late),
        "early_late_center_shift": round_float(center_shift),
    }


@torch.inference_mode()
def visualize_single_sample(
    bundle: Dict[str, Any],
    item: Dict[str, Any],
    sample_index: int,
    args: argparse.Namespace,
    device: torch.device,
    weight_dtype: torch.dtype,
    resolved_image_cfg_scale: float,
    timestep_targets: Sequence[TimestepTarget],
    capture_manager: AttentionCaptureManager,
    output_root: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    tokenizer: CLIPTokenizer = bundle["tokenizer"]
    text_encoder: CLIPTextModel = bundle["text_encoder"]
    image_encoder: CLIPVisionModelWithProjection = bundle["image_encoder"]
    vae: AutoencoderKL = bundle["vae"]
    model: VisualizationSDModel = bundle["model"]
    scaling_factor: float = bundle["scaling_factor"]
    vae_transform = bundle["vae_transform"]
    clip_transform = bundle["clip_transform"]

    sample_dir = ensure_dir(output_root / f"sample_{sample_index:03d}")
    category = str(item.get("category") or "")
    prompt_text = resolve_prompt(item, prompt_override=args.prompt_text)
    person_path = resolve_dresscode_person_path(args.dresscode_root, category, item["image_file"])
    cloth_path = resolve_dresscode_cloth_path(args.dresscode_root, category, item["cloth_file"])

    metadata: Dict[str, Any] = {
        "sample_index": sample_index,
        "category": category,
        "image_file": item.get("image_file", ""),
        "cloth_file": item.get("cloth_file", ""),
        "s_mode": args.s_mode,
        "ablation_mode": mode_to_ablation_mode(args.s_mode),
        "selected_timesteps": [
            {"index": target.index, "value": target.value, "label": target.label} for target in timestep_targets
        ],
        "selected_layers": list(capture_manager.selected_layers),
        "status": "ok",
        "error": "",
        "warnings": [],
    }

    capture_session = CaptureSession(sample_index=sample_index, selected_layers=list(capture_manager.selected_layers), timestep_targets=timestep_targets)
    capture_manager.active_session = capture_session

    person_image = None
    cloth_image = None
    generated_image = None
    person_vae = None
    person_clip = None
    cloth_vae = None
    person_latents = None
    cloth_latents = None
    person_image_embeds = None
    zero_image_embeds = None
    text_embeddings = None
    latents = None
    map_rows: List[Dict[str, Any]] = []

    try:
        if not os.path.exists(person_path):
            raise FileNotFoundError(f"Person image not found: {person_path}")
        person_image = Image.open(person_path).convert("RGB")
        input_person_path = sample_dir / "input_person.png"
        person_image.save(input_person_path)
        metadata["input_person_path"] = rel_to(output_root, input_person_path)

        if os.path.exists(cloth_path):
            cloth_image = Image.open(cloth_path).convert("RGB")
            cloth_path_out = sample_dir / "gt_cloth.png"
            cloth_image.save(cloth_path_out)
            metadata["gt_cloth_path"] = rel_to(output_root, cloth_path_out)
        else:
            metadata["gt_cloth_path"] = ""

        person_vae = vae_transform(person_image).unsqueeze(0).to(device=device, dtype=weight_dtype)
        person_clip = clip_transform(person_image).unsqueeze(0).to(device=device, dtype=weight_dtype)
        if args.init_mode == "from_noisy_gt":
            if cloth_image is None:
                raise FileNotFoundError(f"Cloth image not found: {cloth_path}")
            cloth_vae = vae_transform(cloth_image).unsqueeze(0).to(device=device, dtype=weight_dtype)

        text_inputs = tokenizer(
            prompt_text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with _get_autocast_context(device, weight_dtype):
            text_embeddings = text_encoder(text_inputs.input_ids.to(device=device))[0].to(dtype=weight_dtype)
            person_posterior = vae.encode(person_vae).latent_dist
            person_latents = (person_posterior.mode() if args.vae_deterministic else person_posterior.sample()).to(dtype=weight_dtype)
            person_latents = person_latents * scaling_factor
            person_image_embeds = image_encoder(person_clip, output_hidden_states=True).hidden_states[-2].to(dtype=weight_dtype)
            zero_image_embeds = torch.zeros_like(person_image_embeds)
            if args.init_mode == "from_noisy_gt":
                cloth_posterior = vae.encode(cloth_vae).latent_dist
                cloth_latents = (cloth_posterior.mode() if args.vae_deterministic else cloth_posterior.sample()).to(dtype=weight_dtype)
                cloth_latents = cloth_latents * scaling_factor

        scheduler, _, run_timesteps, _ = resolve_scheduler_timesteps(
            scheduler_name=args.scheduler,
            timestep_spacing=args.timestep_spacing,
            num_inference_steps=args.num_inference_steps,
            device=device,
            init_mode=args.init_mode,
            reconstruct_t=args.reconstruct_t,
        )

        generator = make_generator(device, args.seed + sample_index)
        if args.init_mode == "from_noisy_gt":
            noise = torch.randn(
                cloth_latents.shape,
                generator=generator,
                device=cloth_latents.device,
                dtype=cloth_latents.dtype,
            )
            start_t = run_timesteps[0] if int(run_timesteps.numel()) > 0 else scheduler.timesteps[0]
            start_t_tensor = as_timestep_tensor(start_t, device=cloth_latents.device, batch_size=cloth_latents.shape[0])
            latents = scheduler.add_noise(cloth_latents, noise, start_t_tensor)
        else:
            latents = torch.randn(
                person_latents.shape,
                generator=generator,
                device=person_latents.device,
                dtype=person_latents.dtype,
            )
            latents = latents * scheduler.init_noise_sigma

        selected_index_set = {target.index for target in timestep_targets}
        for step_index, timestep in enumerate(run_timesteps):
            capture_session.begin_step(step_index)
            timestep_tensor = as_timestep_tensor(timestep, device=latents.device, batch_size=latents.shape[0])
            latents_input = scheduler.scale_model_input(latents, timestep_tensor) if hasattr(scheduler, "scale_model_input") else latents
            capture_session.capture_enabled = step_index in selected_index_set
            noise_pred = model(text_embeddings, latents_input, person_latents, person_image_embeds, timestep_tensor)
            if resolved_image_cfg_scale != 1.0:
                capture_session.capture_enabled = False
                noise_pred_uncond = model(text_embeddings, latents_input, person_latents, zero_image_embeds, timestep_tensor)
                noise_pred = noise_pred_uncond + resolved_image_cfg_scale * (noise_pred - noise_pred_uncond)
            capture_session.capture_enabled = False
            latents = scheduler.step(noise_pred, timestep_tensor, latents).prev_sample
            capture_session.end_step()

        with _get_autocast_context(device, weight_dtype):
            decoded = vae.decode(latents / scaling_factor).sample
        generated_image = tensor_image_to_pil(decoded)
        generated_path = sample_dir / "generated.png"
        if args.save_generated_images:
            generated_image.save(generated_path)
        metadata["generated_path"] = rel_to(output_root, generated_path if args.save_generated_images else None)

        ordered_records = [capture_session.records[key] for key in sorted(capture_session.records.keys(), key=lambda pair: (pair[1], pair[0]))]
        map_rows, _, panel_path = save_map_artifacts(
            output_root=output_root,
            sample_dir=sample_dir,
            person_image=person_image,
            generated_image=generated_image,
            gt_image=cloth_image,
            captured_records=ordered_records,
            args=args,
        )
        sample_metrics = compute_sample_aggregates(map_rows)
        metadata.update(sample_metrics)
        metadata["warnings"] = capture_session.warnings
        metadata["panel_path"] = rel_to(output_root, panel_path)
        metadata["maps"] = []

        per_map_rows: List[Dict[str, Any]] = []
        for row in map_rows:
            item_row = {
                "sample_index": sample_index,
                "category": category,
                "image_file": item.get("image_file", ""),
                "cloth_file": item.get("cloth_file", ""),
                "s_mode": args.s_mode,
                "ablation_mode": mode_to_ablation_mode(args.s_mode),
            }
            item_row.update({key: value for key, value in row.items() if key != "stage_prob_map"})
            per_map_rows.append(item_row)
            metadata["maps"].append({key: value for key, value in item_row.items() if key != "sample_index"})

        per_sample_row = {
            "sample_index": sample_index,
            "category": category,
            "image_file": item.get("image_file", ""),
            "cloth_file": item.get("cloth_file", ""),
            "s_mode": args.s_mode,
            "ablation_mode": mode_to_ablation_mode(args.s_mode),
            "panel_path": metadata.get("panel_path", ""),
            "generated_path": metadata.get("generated_path", ""),
            "status": "ok",
            "error": "",
        }
        per_sample_row.update(sample_metrics)
        return metadata, per_map_rows, per_sample_row
    except Exception as exc:
        metadata["status"] = "error"
        metadata["error"] = str(exc)
        metadata["traceback"] = traceback.format_exc()
        per_sample_row = {
            "sample_index": sample_index,
            "category": category,
            "image_file": item.get("image_file", ""),
            "cloth_file": item.get("cloth_file", ""),
            "s_mode": args.s_mode,
            "ablation_mode": mode_to_ablation_mode(args.s_mode),
            "panel_path": "",
            "generated_path": "",
            "status": "error",
            "error": str(exc),
            "num_maps": 0,
            "mean_entropy": None,
            "mean_top10_mass": None,
            "mean_area_above_05": None,
            "early_mid_correlation": None,
            "mid_late_correlation": None,
            "early_late_center_shift": None,
        }
        return metadata, [], per_sample_row
    finally:
        capture_manager.active_session = None
        _clear_ref_unet_cache(model.ref_unet)
        _clear_unet_extra_state(model.unet)
        del person_vae, person_clip, cloth_vae, person_latents, cloth_latents
        del person_image_embeds, zero_image_embeds, text_embeddings, latents
        gc.collect()


def summarize_metric_rows(rows: Sequence[Dict[str, Any]], group_key: Optional[str], metric_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = str(row.get(group_key, "all")) if group_key else "all"
        for metric in metric_names:
            value = row.get(metric)
            if value is None or value == "":
                continue
            grouped[group][metric].append(float(value))

    summary: Dict[str, Dict[str, Any]] = {}
    for group, metrics in grouped.items():
        summary[group] = {}
        for metric, values in metrics.items():
            summary[group][metric] = {
                "mean": round_float(safe_mean(values)),
                "std": round_float(safe_std(values)),
                "count": len(values),
            }
    return summary


def build_summary_outputs(per_map_rows: Sequence[Dict[str, Any]], per_sample_rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    map_metrics = ["entropy", "top5_mass", "top10_mass", "peak_value", "area_above_05", "center_x_norm", "center_y_norm"]
    sample_metrics = ["mean_entropy", "mean_top10_mass", "mean_area_above_05"]
    dynamics_metrics = ["early_mid_correlation", "mid_late_correlation", "early_late_center_shift"]

    summary = {
        "map_metrics": {
            "overall": summarize_metric_rows(per_map_rows, None, map_metrics).get("all", {}),
            "per_category": summarize_metric_rows(per_map_rows, "category", map_metrics),
            "per_layer": summarize_metric_rows(per_map_rows, "layer_name", map_metrics),
            "per_timestep": summarize_metric_rows(per_map_rows, "timestep_label", map_metrics),
        },
        "sample_metrics": {
            "overall": summarize_metric_rows(per_sample_rows, None, sample_metrics).get("all", {}),
            "per_category": summarize_metric_rows(per_sample_rows, "category", sample_metrics),
        },
        "cross_stage_dynamics": {
            "overall": summarize_metric_rows(per_sample_rows, None, dynamics_metrics).get("all", {}),
            "per_category": summarize_metric_rows(per_sample_rows, "category", dynamics_metrics),
        },
    }

    csv_rows: List[Dict[str, Any]] = []
    for section, section_payload in summary.items():
        for scope, scope_payload in section_payload.items():
            if scope == "overall":
                scope_iter = {"all": scope_payload}
            else:
                scope_iter = scope_payload
            for group, metrics in scope_iter.items():
                for metric_name, stats in metrics.items():
                    csv_rows.append(
                        {
                            "section": section,
                            "scope": scope,
                            "group": group,
                            "metric": metric_name,
                            "mean": stats.get("mean"),
                            "std": stats.get("std"),
                            "count": stats.get("count"),
                        }
                    )
    return summary, csv_rows


def cleanup_bundle(bundle: Optional[Dict[str, Any]], device: torch.device) -> None:
    if not bundle:
        return
    for key in ("model", "text_encoder", "image_encoder", "vae", "tokenizer"):
        if key in bundle:
            del bundle[key]
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone DressCode attention visualization.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base diffusion model path.")
    parser.add_argument("--image_encoder_path", type=str, required=True, help="CLIP image encoder path.")
    parser.add_argument("--pretrained_vae_model_path", type=str, required=True, help="VAE model path.")
    parser.add_argument("--dresscode_root", type=str, required=True, help="DressCode dataset root.")
    parser.add_argument("--subset_json", type=str, default=DEFAULT_SUBSET_JSON, help="Subset JSON path.")
    parser.add_argument("--dresscode_category", type=str, default="all", choices=["all"] + DRESSCODE_CATEGORIES)
    parser.add_argument("--dresscode_test_order", type=str, default="paired", choices=VALID_TEST_ORDERS)
    parser.add_argument("--s_mode", type=str, default="S2", choices=["S1", "S2"])
    parser.add_argument("--run_mode", type=str, default="figure", choices=["figure", "summary"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--scheduler", type=str, default="dpmpp_2m_karras", choices=["ddim", "unipc", "dpmpp_2m", "dpmpp_2m_karras", "euler_a"])
    parser.add_argument("--timestep_spacing", type=str, default="trailing")
    parser.add_argument("--max_samples", type=int, default=0, help="Override mode default sample count.")
    parser.add_argument("--selected_timesteps", type=str, default="")
    parser.add_argument("--selected_layers", type=str, default="")
    parser.add_argument("--save_raw_maps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_overlay_on_person", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_overlay_on_output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_generated_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_root", type=str, default="outputs/attention_visualization_dresscode")
    parser.add_argument("--prompt_text", type=str, default="")
    parser.add_argument("--init_mode", type=str, default="from_noisy_gt", choices=["from_noisy_gt", "random"])
    parser.add_argument("--reconstruct_t", type=int, default=600)
    parser.add_argument("--image_cfg_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--vae_deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_all_summary_items", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device, weight_dtype, resolved_dtype, dtype_warnings = resolve_device_and_dtype(args.device, args.dtype)
    resolved_image_cfg_scale = float(args.image_cfg_scale)
    if resolved_image_cfg_scale == 1.0 and float(args.guidance_scale) != 1.0:
        log("[WARN] guidance_scale is deprecated; using it as image_cfg_scale.")
        resolved_image_cfg_scale = float(args.guidance_scale)

    output_root = ensure_dir(Path(args.output_root))
    subset_items, subset_info = resolve_subset(args)
    write_json(output_root / "resolved_subset.json", subset_items)

    ablation_mode = mode_to_ablation_mode(args.s_mode)
    ablation_config = build_ablation_config(ablation_mode)
    bundle = None

    try:
        bundle = build_visualization_bundle(args, ablation_config, device, weight_dtype)
        bundle["warnings"] = list(dtype_warnings) + list(bundle.get("checkpoint_warnings", []))

        _, _, base_run_timesteps, _ = resolve_scheduler_timesteps(
            scheduler_name=args.scheduler,
            timestep_spacing=args.timestep_spacing,
            num_inference_steps=args.num_inference_steps,
            device=device,
            init_mode=args.init_mode,
            reconstruct_t=args.reconstruct_t,
        )
        timestep_targets = select_timestep_targets(base_run_timesteps, args.selected_timesteps)

        layer_records = discover_unet_layers(bundle["model"].unet, args.s_mode)
        selected_layers = select_layers(layer_records, args.selected_layers, args.s_mode)
        write_json(
            output_root / "discovered_layers.json",
            {
                "s_mode": args.s_mode,
                "ablation_mode": ablation_mode,
                "selected_layers": selected_layers,
                "layers": layer_records,
            },
        )

        per_map_rows: List[Dict[str, Any]] = []
        per_sample_rows: List[Dict[str, Any]] = []
        sample_entries: List[Dict[str, Any]] = []

        log(
            f"[INFO] mode={args.run_mode} s_mode={args.s_mode} device={device} dtype={resolved_dtype} "
            f"samples={len(subset_items)} layers={len(selected_layers)} timesteps={len(timestep_targets)}"
        )

        with AttentionCaptureManager(bundle["model"].unet, selected_layers) as capture_manager:
            for sample_index, item in enumerate(subset_items):
                metadata, sample_map_rows, sample_row = visualize_single_sample(
                    bundle=bundle,
                    item=item,
                    sample_index=sample_index,
                    args=args,
                    device=device,
                    weight_dtype=weight_dtype,
                    resolved_image_cfg_scale=resolved_image_cfg_scale,
                    timestep_targets=timestep_targets,
                    capture_manager=capture_manager,
                    output_root=output_root,
                )
                sample_entries.append(metadata)
                per_map_rows.extend(sample_map_rows)
                per_sample_rows.append(sample_row)
                if (sample_index + 1) % 5 == 0 or sample_index + 1 == len(subset_items):
                    log(f"[INFO] processed {sample_index + 1}/{len(subset_items)} samples")

        map_fieldnames = [
            "sample_index",
            "category",
            "image_file",
            "cloth_file",
            "s_mode",
            "ablation_mode",
            "layer_name",
            "processor_type",
            "map_type",
            "branch",
            "aggregation",
            "note",
            "timestep_index",
            "timestep_value",
            "timestep_label",
            "spatial_height",
            "spatial_width",
            "entropy",
            "top5_mass",
            "top10_mass",
            "peak_value",
            "area_above_05",
            "center_x_norm",
            "center_y_norm",
            "raw_path",
            "overlay_person_path",
            "overlay_output_path",
        ]
        sample_fieldnames = [
            "sample_index",
            "category",
            "image_file",
            "cloth_file",
            "s_mode",
            "ablation_mode",
            "panel_path",
            "generated_path",
            "status",
            "error",
            "num_maps",
            "mean_entropy",
            "mean_top10_mass",
            "mean_area_above_05",
            "early_mid_correlation",
            "mid_late_correlation",
            "early_late_center_shift",
        ]
        write_csv(output_root / "attention_metrics_per_map.csv", per_map_rows, map_fieldnames)
        write_csv(output_root / "attention_metrics_per_sample.csv", per_sample_rows, sample_fieldnames)

        metrics_summary, metrics_summary_csv = build_summary_outputs(per_map_rows, per_sample_rows)
        write_json(output_root / "attention_metrics_summary.json", metrics_summary)
        write_csv(
            output_root / "attention_metrics_summary.csv",
            metrics_summary_csv,
            ["section", "scope", "group", "metric", "mean", "std", "count"],
        )

        summary_payload = {
            "run_mode": args.run_mode,
            "s_mode": args.s_mode,
            "ablation_mode": ablation_mode,
            "device": str(device),
            "dtype": resolved_dtype,
            "scheduler": args.scheduler,
            "timestep_spacing": args.timestep_spacing,
            "num_inference_steps": args.num_inference_steps,
            "image_size": args.image_size,
            "selected_layers": selected_layers,
            "selected_timesteps": [
                {"index": target.index, "value": target.value, "label": target.label} for target in timestep_targets
            ],
            "subset_info": subset_info,
            "warnings": bundle.get("warnings", []),
            "output_files": {
                "discovered_layers": "discovered_layers.json",
                "per_map_metrics": "attention_metrics_per_map.csv",
                "per_sample_metrics": "attention_metrics_per_sample.csv",
                "metrics_summary_json": "attention_metrics_summary.json",
                "metrics_summary_csv": "attention_metrics_summary.csv",
            },
            "samples": sample_entries,
        }
        write_json(output_root / "summary.json", summary_payload)
        log(f"[DONE] Saved attention visualization outputs to {output_root}")
    finally:
        cleanup_bundle(bundle, device)


if __name__ == "__main__":
    main()
