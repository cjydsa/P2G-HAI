#!/usr/bin/env python3
from __future__ import annotations

"""Standalone DressCode efficiency profiler for IMAGDressing S0/S1/S2."""

import argparse
import csv
import gc
import json
import math
import os
import random
import statistics
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageFile
from torch import nn
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

DRESSCODE_CATEGORIES = ["upper_body", "lower_body", "dresses"]
VALID_TEST_ORDERS = ["paired", "unpaired"]
VALID_S_MODES = ["S0", "S1", "S2"]
FALLBACK_PROMPT = "a studio-style product shot of the garment only"

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
    if ext:
        return name
    return f"{base}.jpg"


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
            missing_str = ", ".join(missing_categories)
            raise RuntimeError(
                "Root-level pairs file did not provide entries for required categories: "
                f"{missing_str}."
            )

        loaded_fallback = {item["category"] for item in fallback_pairs}
        still_missing = [cat for cat in missing_categories if cat not in loaded_fallback]
        if still_missing:
            missing_str = ", ".join(still_missing)
            raise RuntimeError(
                "Root-level pairs file missing category prefixes for required categories: "
                f"{missing_str}."
            )
        data.extend(fallback_pairs)

    if not data:
        raise RuntimeError("No pairs loaded from DressCode dataset.")
    return data


def resolve_prompt(item: Dict[str, Any]) -> str:
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


def select_subset(items: Sequence[Dict[str, Any]], limit: int, seed: int, mode: str) -> List[Dict[str, Any]]:
    if limit <= 0 or len(items) <= limit:
        return [dict(item) for item in items]
    if mode == "random":
        rng = random.Random(seed)
        indices = rng.sample(range(len(items)), limit)
        indices.sort()
        return [dict(items[idx]) for idx in indices]
    return [dict(item) for item in items[:limit]]


def resolve_subset(args: argparse.Namespace, output_root: Path) -> List[Dict[str, Any]]:
    default_category = None if args.dresscode_category == "all" else args.dresscode_category
    if args.subset_json:
        items = load_subset_json(args.subset_json, default_category=default_category)
    else:
        if not args.dresscode_root:
            raise RuntimeError("--dresscode_root is required when --subset_json is not provided.")
        items = load_dresscode_pairs(
            root=args.dresscode_root,
            split="test",
            category=args.dresscode_category,
            test_order=args.dresscode_test_order,
        )

    subset = select_subset(items, args.profile_samples, args.seed, args.subset_sampling)
    if not subset:
        raise RuntimeError("Resolved subset is empty.")
    write_json(output_root / "resolved_subset.json", subset)
    return subset


def mode_to_ablation_mode(s_mode: str) -> str:
    mapping = {"S0": "A0", "S1": "A1", "S2": "A2"}
    normalized = (s_mode or "").upper()
    if normalized not in mapping:
        raise ValueError(f"Unsupported s_mode: {s_mode}")
    return mapping[normalized]


def build_ablation_config(mode: str) -> AblationConfig:
    normalized = (mode or "A2").upper()
    if normalized == "A0":
        return AblationConfig(
            mode="A0",
            enable_ieb=False,
            enable_ha=False,
            direct_ieb=False,
            unet_train="cross_attn",
        )
    if normalized == "A1":
        return AblationConfig(
            mode="A1",
            enable_ieb=True,
            enable_ha=False,
            direct_ieb=True,
            unet_train="frozen",
        )
    if normalized == "A2":
        return AblationConfig(
            mode="A2",
            enable_ieb=True,
            enable_ha=True,
            direct_ieb=False,
            unet_train="frozen",
        )
    raise ValueError(f"Unknown ablation mode: {mode}")


def _get_autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _timer_start(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _timer_end(device: torch.device, start_time: float) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start_time) * 1000.0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return float(allocated), float(reserved)


def bytes_to_mb(value: int) -> float:
    """Convert bytes to the MiB convention used by PyTorch's CUDA memory APIs."""
    return float(value) / (1024 ** 2)


def measure_tensor_cache_bytes(obj: Any) -> Dict[str, int]:
    """Measure logical tensor bytes and unique backing-storage bytes held by a cache."""
    seen_objects: set = set()
    seen_tensors: set = set()
    seen_storages: set = set()
    tensor_bytes = 0
    storage_bytes = 0
    tensor_count = 0

    def visit(value: Any) -> None:
        nonlocal tensor_bytes, storage_bytes, tensor_count
        if torch.is_tensor(value):
            tensor_id = id(value)
            if tensor_id in seen_tensors:
                return
            seen_tensors.add(tensor_id)
            tensor_count += 1
            tensor_bytes += int(value.numel()) * int(value.element_size())
            try:
                storage = value.untyped_storage()
                storage_key = (str(value.device), int(storage.data_ptr()), int(storage.nbytes()))
                if storage_key not in seen_storages:
                    seen_storages.add(storage_key)
                    storage_bytes += int(storage.nbytes())
            except (AttributeError, RuntimeError):
                storage_key = (str(value.device), int(value.data_ptr()), int(value.numel()), int(value.element_size()))
                if storage_key not in seen_storages:
                    seen_storages.add(storage_key)
                    storage_bytes += int(value.numel()) * int(value.element_size())
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        object_id = id(value)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                visit(nested)

    visit(obj)
    return {
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
        "storage_bytes": storage_bytes,
    }


def count_model_params_raw(model: Optional[nn.Module], trainable_only: bool = False) -> int:
    if model is None:
        return 0
    if trainable_only:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in model.parameters())


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


def _enable_unet_cross_attn_training(unet: nn.Module) -> List[nn.Parameter]:
    trainable_params: List[nn.Parameter] = []
    seen: set = set()
    for name, module in unet.named_modules():
        if ".attn2" in name:
            module.requires_grad_(True)
            for parameter in module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if parameter.requires_grad:
                    trainable_params.append(parameter)
    return trainable_params


def count_cross_attn_params(unet: Optional[nn.Module], ablation_config: AblationConfig) -> Dict[str, int]:
    ensure_runtime_imports()
    if unet is None or not ablation_config.enable_ha:
        return {"total": 0, "trainable": 0}
    total = 0
    trainable = 0
    for processor in unet.attn_processors.values():
        if isinstance(processor, RefCAttnProcessor2_0):
            total += count_model_params_raw(processor)
            trainable += count_model_params_raw(processor, trainable_only=True)
    return {"total": total, "trainable": trainable}


def collect_model_stats(
    unet: nn.Module,
    ref_unet: nn.Module,
    image_proj: nn.Module,
    image_encoder: nn.Module,
    text_encoder: nn.Module,
    vae: nn.Module,
    ablation_config: AblationConfig,
    s_mode: str,
) -> Dict[str, Any]:
    cross_attn_stats = count_cross_attn_params(unet, ablation_config)
    unet_cross_train_total = 0
    unet_cross_train_trainable = 0
    if ablation_config.unet_train == "cross_attn":
        unique_params: Dict[int, nn.Parameter] = {}
        for name, module in unet.named_modules():
            if ".attn2" in name:
                for parameter in module.parameters():
                    unique_params.setdefault(id(parameter), parameter)
        unet_cross_train_total = sum(parameter.numel() for parameter in unique_params.values())
        unet_cross_train_trainable = sum(
            parameter.numel() for parameter in unique_params.values() if parameter.requires_grad
        )

    modules = {}
    total_params = 0
    trainable_params = 0
    for name, module in (
        ("unet", unet),
        ("ref_unet", ref_unet),
        ("image_proj", image_proj),
        ("image_encoder", image_encoder),
        ("text_encoder", text_encoder),
        ("vae", vae),
    ):
        module_total = count_model_params_raw(module)
        module_trainable = count_model_params_raw(module, trainable_only=True)
        modules[name] = {"total": module_total, "trainable": module_trainable}
        total_params += module_total
        trainable_params += module_trainable

    modules["cross_attn_processors"] = cross_attn_stats
    modules["unet_cross_attn_train_group"] = {
        "total": unet_cross_train_total,
        "trainable": unet_cross_train_trainable,
    }

    return {
        "mode": s_mode,
        "ablation_mode": ablation_config.mode,
        "modules": modules,
        "totals": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_m": round_float(total_params / 1e6),
            "trainable_params_m": round_float(trainable_params / 1e6),
        },
    }


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


def _timesteps_to_list(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    else:
        tensor = torch.tensor(values)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    return [int(v) for v in tensor.flatten().tolist()]


def as_timestep_tensor(timestep: Any, device: torch.device, batch_size: int) -> torch.Tensor:
    if isinstance(timestep, torch.Tensor):
        tensor = timestep.to(device=device)
    else:
        tensor = torch.tensor(timestep, device=device)
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


def _select_start_index(values: Sequence[int], target_t: Optional[int], min_steps_value: Optional[int]) -> int:
    if target_t is None:
        return 0
    target = float(target_t)
    idx = min(range(len(values)), key=lambda i: abs(float(values[i]) - target))
    if min_steps_value is not None:
        min_steps = max(1, int(min_steps_value))
        if len(values) - idx < min_steps:
            idx = max(0, len(values) - min_steps)
    return idx


def resolve_scheduler_timesteps(
    scheduler_name: str,
    timestep_spacing: str,
    num_inference_steps: int,
    device: torch.device,
    init_mode: str,
    reconstruct_t: Optional[int],
    min_effective_steps: Optional[int],
):
    scheduler = _build_scheduler(scheduler_name, timestep_spacing)
    _set_scheduler_timesteps(scheduler, num_inference_steps, device)
    timesteps = scheduler.timesteps
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
    timesteps = timesteps.to(device)
    timesteps_list = _timesteps_to_list(timesteps)
    if not timesteps_list:
        raise RuntimeError("scheduler.timesteps is empty after set_timesteps")

    if init_mode == "from_noisy_gt":
        start_idx = _select_start_index(timesteps_list, reconstruct_t, min_effective_steps)
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


def resolve_refine_timesteps(
    scheduler_name: str,
    timestep_spacing: str,
    refine_steps: int,
    refine_t: Optional[int],
    device: torch.device,
):
    scheduler = _build_scheduler(scheduler_name, timestep_spacing)
    _set_scheduler_timesteps(scheduler, refine_steps, device)
    timesteps = scheduler.timesteps
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
    timesteps = timesteps.to(device)
    timesteps_list = _timesteps_to_list(timesteps)
    if not timesteps_list:
        return None, None, None

    start_idx = _select_start_index(timesteps_list, refine_t, None)
    run_timesteps = timesteps[start_idx:]
    if int(run_timesteps.numel()) == 0:
        start_idx = 0
        run_timesteps = timesteps

    if hasattr(scheduler, "set_begin_index"):
        try:
            scheduler.set_begin_index(start_idx)
        except Exception as exc:
            log(f"[WARN] refine_scheduler.set_begin_index failed: {exc}")
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    start_t = run_timesteps[0] if int(run_timesteps.numel()) > 0 else timesteps[0]
    return scheduler, run_timesteps, start_t


def resolve_s_modes(raw: str) -> List[str]:
    modes: List[str] = []
    for token in str(raw or "").split(","):
        mode = token.strip().upper()
        if not mode:
            continue
        if mode not in VALID_S_MODES:
            raise ValueError(f"Unsupported mode in --s_modes: {mode}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("--s_modes resolved to an empty list.")
    return modes


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

    dtype_key = (requested_dtype or "auto").strip().lower()
    dtype_aliases = {
        "fp32": "fp32",
        "float32": "fp32",
        "fp16": "fp16",
        "float16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "auto": "auto",
    }
    if dtype_key not in dtype_aliases:
        raise ValueError(f"Unsupported dtype: {requested_dtype}")
    normalized = dtype_aliases[dtype_key]

    if normalized == "auto":
        normalized = "fp16" if device.type == "cuda" else "fp32"

    if device.type == "cpu" and normalized != "fp32":
        warnings.append(f"dtype={normalized} is not supported on CPU for this profiler; using fp32.")
        normalized = "fp32"

    if device.type == "cuda" and normalized == "bf16" and not torch.cuda.is_bf16_supported():
        warnings.append("CUDA bf16 requested but not supported on this GPU; using fp16.")
        normalized = "fp16"

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    return device, dtype_map[normalized], normalized, warnings


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=str(device))
    except Exception:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


class ProfilerSDModel(nn.Module):
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
        timing_meter: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        amp_dtype = latents.dtype
        if encoder_hidden_states.dtype != amp_dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype=amp_dtype)
        if clip_image_embeddings is not None and clip_image_embeddings.dtype != amp_dtype:
            clip_image_embeddings = clip_image_embeddings.to(dtype=amp_dtype)
        if ref_latents.dtype != amp_dtype:
            ref_latents = ref_latents.to(dtype=amp_dtype)

        person_proj_embed = None
        measure_cache_this_call = False
        if self.enable_ieb and clip_image_embeddings is not None:
            with _get_autocast_context(latents.device, amp_dtype):
                person_proj_embed = self.proj(clip_image_embeddings)

        sa_hidden_states = None
        if self.enable_ha:
            if self.ref_unet is None:
                raise RuntimeError("ref_unet is required when enable_ha is True.")
            _clear_ref_unet_cache(self.ref_unet)
            ref_timesteps = torch.zeros_like(timesteps)
            ref_start = _timer_start(latents.device) if timing_meter is not None else None
            with _get_autocast_context(latents.device, amp_dtype):
                _ = self.ref_unet(
                    ref_latents,
                    ref_timesteps,
                    encoder_hidden_states=person_proj_embed,
                    return_dict=False,
                )
            if timing_meter is not None and ref_start is not None:
                timing_meter["ref_unet_ms"] = timing_meter.get("ref_unet_ms", 0.0) + _timer_end(
                    latents.device,
                    ref_start,
                )

            sa_hidden_states = {}
            for name in self.ref_unet.attn_processors.keys():
                processor = self.ref_unet.attn_processors[name]
                if hasattr(processor, "cache") and "hidden_states" in processor.cache:
                    sa_hidden_states[name] = processor.cache["hidden_states"]
                else:
                    raise RuntimeError(f"Attention cache missing for {name}")

            measure_cache_this_call = timing_meter is not None and not bool(
                timing_meter.get("cache_measurement_taken", False)
            )
            if measure_cache_this_call:
                cache_stats = measure_tensor_cache_bytes(sa_hidden_states)
                timing_meter["cache_tensor_count"] = int(cache_stats["tensor_count"])
                timing_meter["cache_tensor_bytes"] = int(cache_stats["tensor_bytes"])
                timing_meter["cache_storage_bytes"] = int(cache_stats["storage_bytes"])
                timing_meter["cache_measurement_taken"] = True
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

        if not torch.is_grad_enabled() and isinstance(sa_hidden_states, dict):
            allocated_before_clear = None
            if measure_cache_this_call and latents.device.type == "cuda" and torch.cuda.is_available():
                allocated_before_clear = int(torch.cuda.memory_allocated(latents.device))
            sa_hidden_states.clear()
            cross_kwargs = None
            if allocated_before_clear is not None:
                torch.cuda.synchronize(latents.device)
                allocated_after_clear = int(torch.cuda.memory_allocated(latents.device))
                timing_meter["cache_cuda_released_bytes"] = max(
                    0,
                    allocated_before_clear - allocated_after_clear,
                )
        return noise_pred


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


def load_profiler_checkpoint(checkpoint_path: str, model: ProfilerSDModel) -> Tuple[Dict[str, Any], List[str]]:
    ensure_runtime_imports()
    if not checkpoint_path:
        raise FileNotFoundError("--checkpoint is required.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise CheckpointLoadError(f"Checkpoint must contain a dict payload: {checkpoint_path}")

    warnings: List[str] = []

    def load_state(module: Optional[nn.Module], key: str, label: str) -> bool:
        if module is None:
            warnings.append(f"[CKPT] Skip {label}: module not available.")
            return False
        state = ckpt.get(key)
        if state is None:
            warnings.append(f"[CKPT] Missing {key}; keeping {label} initialization.")
            return False
        try:
            result = module.load_state_dict(state, strict=False)
        except Exception as exc:
            raise CheckpointLoadError(f"[CKPT] Failed to load {label}: {exc}") from exc
        if getattr(result, "missing_keys", None):
            warnings.append(f"[CKPT] {label} missing_keys={len(result.missing_keys)}")
        if getattr(result, "unexpected_keys", None):
            warnings.append(f"[CKPT] {label} unexpected_keys={len(result.unexpected_keys)}")
        return True

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
        skipped = 0
        errors = 0
        for name, state in cross_payload.items():
            processor = model.unet.attn_processors.get(name)
            if processor is None or not isinstance(processor, RefCAttnProcessor2_0):
                skipped += 1
                continue
            try:
                result = processor.load_state_dict(state, strict=False)
                loaded += 1
            except Exception as exc:
                errors += 1
                warnings.append(f"[CKPT] Failed to load cross_attn {name}: {exc}")
                continue
            if getattr(result, "missing_keys", None):
                warnings.append(f"[CKPT] cross_attn {name} missing={len(result.missing_keys)}")
            if getattr(result, "unexpected_keys", None):
                warnings.append(f"[CKPT] cross_attn {name} unexpected={len(result.unexpected_keys)}")

        missing = sum(1 for name in expected_refc if name not in cross_payload)
        if expected_refc and loaded == 0:
            warnings.append("[CKPT] No RefCAttnProcessor2_0 weights loaded (mode mismatch possible).")
        if missing > 0:
            warnings.append(f"[CKPT] Missing {missing} cross-attn processor weights; using initialization.")
        if skipped > 0:
            warnings.append(f"[CKPT] Skipped {skipped} cross-attn weights due to name/type mismatch.")
        if errors > 0:
            warnings.append(f"[CKPT] Failed to load {errors} cross-attn processor weights.")

    return ckpt, warnings


def move_modules_to_device(modules: Sequence[nn.Module], device: torch.device, dtype: torch.dtype) -> None:
    for module in modules:
        module.to(device=device, dtype=dtype)
        module.eval()


def build_profiler_bundle(
    args: argparse.Namespace,
    s_mode: str,
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
        for _, processor in unet.attn_processors.items():
            if isinstance(processor, RefSAttnProcessor2_0):
                processor.requires_grad_(False)
                processor.eval()
            elif isinstance(processor, RefCAttnProcessor2_0):
                processor.requires_grad_(True)

    unet_trainable_params: List[nn.Parameter] = []
    if ablation_config.unet_train == "cross_attn":
        unet_trainable_params = _enable_unet_cross_attn_training(unet)

    model = ProfilerSDModel(unet=unet, ref_unet=ref_unet, proj=image_proj, ablation_config=ablation_config)
    checkpoint_payload, checkpoint_warnings = load_profiler_checkpoint(args.checkpoint, model)

    move_modules_to_device(
        [model, vae, text_encoder, image_encoder],
        device=device,
        dtype=weight_dtype,
    )

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
        "mode": s_mode,
        "ablation": ablation_config,
        "tokenizer": tokenizer,
        "model": model,
        "text_encoder": text_encoder,
        "image_encoder": image_encoder,
        "vae": vae,
        "scaling_factor": scaling_factor,
        "vae_transform": vae_transform,
        "clip_transform": clip_transform,
        "checkpoint_payload": checkpoint_payload,
        "checkpoint_warnings": checkpoint_warnings,
        "unet_trainable_params": unet_trainable_params,
    }


@torch.inference_mode()
def profile_single_sample(
    bundle: Dict[str, Any],
    item: Dict[str, Any],
    sample_index: int,
    args: argparse.Namespace,
    device: torch.device,
    weight_dtype: torch.dtype,
    resolved_image_cfg_scale: float,
    save_image: bool,
    images_dir: Optional[Path],
) -> Dict[str, Any]:
    tokenizer: CLIPTokenizer = bundle["tokenizer"]
    model: ProfilerSDModel = bundle["model"]
    text_encoder: CLIPTextModel = bundle["text_encoder"]
    image_encoder: CLIPVisionModelWithProjection = bundle["image_encoder"]
    vae: AutoencoderKL = bundle["vae"]
    scaling_factor: float = bundle["scaling_factor"]
    vae_transform = bundle["vae_transform"]
    clip_transform = bundle["clip_transform"]

    prompt_text = resolve_prompt(item)
    category = str(item.get("category") or "")
    person_path = resolve_dresscode_person_path(args.dresscode_root, category, item["image_file"])
    cloth_path = resolve_dresscode_cloth_path(args.dresscode_root, category, item["cloth_file"])

    sample_seed = int(args.seed) + int(sample_index)
    record: Dict[str, Any] = {
        "sample_index": sample_index,
        "mode": bundle["mode"],
        "ablation_mode": bundle["ablation"].mode,
        "category": category,
        "image_file": item.get("image_file", ""),
        "cloth_file": item.get("cloth_file", ""),
        "prompt_text": prompt_text,
        "seed": sample_seed,
        "status": "ok",
        "error": "",
    }

    total_start = _timer_start(device)
    reset_peak_memory(device)

    preprocess_ms = 0.0
    text_encode_ms = 0.0
    image_encode_ms = 0.0
    ref_unet_ms = 0.0
    denoise_ms = 0.0
    vae_decode_ms = 0.0
    postprocess_ms = 0.0
    save_image_ms = 0.0

    person_img = None
    cloth_img = None
    person_vae = None
    person_clip = None
    cloth_vae = None
    person_latents = None
    cloth_latents = None
    person_image_embeds = None
    zero_image_embeds = None
    text_embeddings = None
    latents = None
    image = None
    image_pil = None

    try:
        preprocess_start = _timer_start(device)
        if not os.path.exists(person_path):
            raise FileNotFoundError(f"Person image not found: {person_path}")
        person_img = Image.open(person_path).convert("RGB")
        person_vae = vae_transform(person_img).unsqueeze(0).to(device=device, dtype=weight_dtype)
        person_clip = clip_transform(person_img).unsqueeze(0).to(device=device, dtype=weight_dtype)

        if args.init_mode == "from_noisy_gt":
            if not os.path.exists(cloth_path):
                raise FileNotFoundError(f"Cloth image not found: {cloth_path}")
            cloth_img = Image.open(cloth_path).convert("RGB")
            cloth_vae = vae_transform(cloth_img).unsqueeze(0).to(device=device, dtype=weight_dtype)
        preprocess_ms = _timer_end(device, preprocess_start)

        text_start = _timer_start(device)
        text_inputs = tokenizer(
            prompt_text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with _get_autocast_context(device, weight_dtype):
            text_embeddings = text_encoder(text_inputs.input_ids.to(device=device))[0]
        text_encode_ms = _timer_end(device, text_start)

        image_encode_start = _timer_start(device)
        with _get_autocast_context(device, weight_dtype):
            person_posterior = vae.encode(person_vae).latent_dist
            person_latents = person_posterior.mode() if args.vae_deterministic else person_posterior.sample()
            person_latents = person_latents * scaling_factor

            image_outputs = image_encoder(person_clip, output_hidden_states=True)
            person_image_embeds = image_outputs.hidden_states[-2]
            zero_image_embeds = torch.zeros_like(person_image_embeds)

            if args.init_mode == "from_noisy_gt":
                cloth_posterior = vae.encode(cloth_vae).latent_dist
                cloth_latents = cloth_posterior.mode() if args.vae_deterministic else cloth_posterior.sample()
                cloth_latents = cloth_latents * scaling_factor
        image_encode_ms = _timer_end(device, image_encode_start)

        denoise_stage_start = _timer_start(device)
        scheduler, _, run_timesteps, _ = resolve_scheduler_timesteps(
            scheduler_name=args.scheduler,
            timestep_spacing=args.timestep_spacing,
            num_inference_steps=args.num_inference_steps,
            device=device,
            init_mode=args.init_mode,
            reconstruct_t=args.reconstruct_t,
            min_effective_steps=args.min_effective_steps,
        )

        generator = make_generator(device, sample_seed)
        if args.init_mode == "from_noisy_gt":
            noise = torch.randn(
                cloth_latents.shape,
                generator=generator,
                device=cloth_latents.device,
                dtype=cloth_latents.dtype,
            )
            t_start = run_timesteps[0] if int(run_timesteps.numel()) > 0 else scheduler.timesteps[0]
            t_start_tensor = as_timestep_tensor(t_start, device=cloth_latents.device, batch_size=cloth_latents.shape[0])
            latents = scheduler.add_noise(cloth_latents, noise, t_start_tensor)
        else:
            latents = torch.randn(
                person_latents.shape,
                generator=generator,
                device=person_latents.device,
                dtype=person_latents.dtype,
            )
            latents = latents * scheduler.init_noise_sigma

        timing_meter = {
            "ref_unet_ms": 0.0,
            "cache_measurement_taken": False,
            "cache_tensor_count": 0,
            "cache_tensor_bytes": 0,
            "cache_storage_bytes": 0,
            "cache_cuda_released_bytes": 0,
        }
        for timestep in run_timesteps:
            timestep_tensor = as_timestep_tensor(timestep, device=latents.device, batch_size=latents.shape[0])
            latents_input = scheduler.scale_model_input(latents, timestep_tensor) if hasattr(
                scheduler, "scale_model_input"
            ) else latents
            noise_pred = model(
                text_embeddings,
                latents_input,
                person_latents,
                person_image_embeds,
                timestep_tensor,
                timing_meter=timing_meter,
            )
            if resolved_image_cfg_scale != 1.0:
                noise_pred_uncond = model(
                    text_embeddings,
                    latents_input,
                    person_latents,
                    zero_image_embeds,
                    timestep_tensor,
                    timing_meter=timing_meter,
                )
                noise_pred = noise_pred_uncond + resolved_image_cfg_scale * (noise_pred - noise_pred_uncond)
            latents = scheduler.step(noise_pred, timestep_tensor, latents).prev_sample

        if args.refine_pass:
            refine_scheduler, refine_timesteps, refine_start_t = resolve_refine_timesteps(
                scheduler_name=args.scheduler,
                timestep_spacing=args.timestep_spacing,
                refine_steps=args.refine_steps,
                refine_t=args.refine_t,
                device=device,
            )
            if refine_scheduler is not None and refine_timesteps is not None and int(refine_timesteps.numel()) > 0:
                refine_generator = make_generator(device, sample_seed + 10000)
                refine_noise = torch.randn(
                    latents.shape,
                    generator=refine_generator,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                refine_t_tensor = as_timestep_tensor(refine_start_t, device=latents.device, batch_size=latents.shape[0])
                latents = refine_scheduler.add_noise(latents, refine_noise, refine_t_tensor)
                for timestep in refine_timesteps:
                    timestep_tensor = as_timestep_tensor(timestep, device=latents.device, batch_size=latents.shape[0])
                    latents_input = refine_scheduler.scale_model_input(latents, timestep_tensor) if hasattr(
                        refine_scheduler, "scale_model_input"
                    ) else latents
                    noise_pred = model(
                        text_embeddings,
                        latents_input,
                        person_latents,
                        person_image_embeds,
                        timestep_tensor,
                        timing_meter=timing_meter,
                    )
                    if resolved_image_cfg_scale != 1.0:
                        noise_pred_uncond = model(
                            text_embeddings,
                            latents_input,
                            person_latents,
                            zero_image_embeds,
                            timestep_tensor,
                            timing_meter=timing_meter,
                        )
                        noise_pred = noise_pred_uncond + resolved_image_cfg_scale * (noise_pred - noise_pred_uncond)
                    latents = refine_scheduler.step(noise_pred, timestep_tensor, latents).prev_sample

        denoise_elapsed = _timer_end(device, denoise_stage_start)
        ref_unet_ms = float(timing_meter.get("ref_unet_ms", 0.0))
        if not bundle["ablation"].enable_ha:
            ref_unet_ms = 0.0
        denoise_ms = max(0.0, denoise_elapsed - ref_unet_ms)

        vae_decode_start = _timer_start(device)
        with _get_autocast_context(device, weight_dtype):
            image = vae.decode(latents / scaling_factor).sample
        vae_decode_ms = _timer_end(device, vae_decode_start)

        postprocess_start = _timer_start(device)
        image = (image.float() + 1.0) / 2.0
        image = torch.clamp(image, 0.0, 1.0)
        image_np = (image[0].detach().cpu().permute(1, 2, 0) * 255.0).round().byte().numpy()
        image_pil = Image.fromarray(image_np)
        postprocess_ms = _timer_end(device, postprocess_start)

        if save_image:
            assert images_dir is not None
            save_start = _timer_start(device)
            image_path = images_dir / f"{sample_index:06d}.png"
            image_pil.save(image_path)
            save_image_ms = _timer_end(device, save_start)
            record["saved_image"] = str(image_path)
        else:
            record["saved_image"] = ""

        peak_allocated_mb, peak_reserved_mb = get_peak_memory_mb(device)
        total_core_ms = (
            preprocess_ms
            + text_encode_ms
            + image_encode_ms
            + ref_unet_ms
            + denoise_ms
            + vae_decode_ms
            + postprocess_ms
        )
        total_e2e_ms = _timer_end(device, total_start)

        record.update(
            {
                "preprocess_ms": round_float(preprocess_ms),
                "text_encode_ms": round_float(text_encode_ms),
                "image_encode_ms": round_float(image_encode_ms),
                "ref_unet_ms": round_float(ref_unet_ms),
                "denoise_ms": round_float(denoise_ms),
                "vae_decode_ms": round_float(vae_decode_ms),
                "postprocess_ms": round_float(postprocess_ms),
                "save_image_ms": round_float(save_image_ms),
                "total_core_ms": round_float(total_core_ms),
                "total_e2e_ms": round_float(total_e2e_ms),
                "peak_memory_allocated_mb": round_float(peak_allocated_mb),
                "peak_memory_reserved_mb": round_float(peak_reserved_mb),
                "cache_tensor_count": int(timing_meter.get("cache_tensor_count", 0)),
                "cache_tensor_mb": round_float(bytes_to_mb(int(timing_meter.get("cache_tensor_bytes", 0)))),
                "cache_storage_mb": round_float(bytes_to_mb(int(timing_meter.get("cache_storage_bytes", 0)))),
                "cache_cuda_released_mb": round_float(
                    bytes_to_mb(int(timing_meter.get("cache_cuda_released_bytes", 0)))
                ),
            }
        )
        return record
    finally:
        _clear_ref_unet_cache(model.ref_unet)
        _clear_unet_extra_state(model.unet)
        del person_img, cloth_img, person_vae, person_clip, cloth_vae
        del person_latents, cloth_latents, person_image_embeds, zero_image_embeds
        del text_embeddings, latents, image, image_pil
        gc.collect()


def safe_stat_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def safe_stat_std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def summarize_runtime_records(
    s_mode: str,
    ablation_config: AblationConfig,
    model_stats: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    resolved_device: str,
    resolved_dtype: str,
    resolved_image_cfg_scale: float,
    checkpoint_warnings: Sequence[str],
) -> Dict[str, Any]:
    success_records = [row for row in records if row.get("status") == "ok"]
    core_values = [float(row["total_core_ms"]) for row in success_records]
    e2e_values = [float(row["total_e2e_ms"]) for row in success_records]
    alloc_values = [float(row["peak_memory_allocated_mb"]) for row in success_records]
    reserved_values = [float(row["peak_memory_reserved_mb"]) for row in success_records]
    cache_tensor_values = [float(row.get("cache_tensor_mb", 0.0)) for row in success_records]
    cache_storage_values = [float(row.get("cache_storage_mb", 0.0)) for row in success_records]
    cache_released_values = [float(row.get("cache_cuda_released_mb", 0.0)) for row in success_records]
    cache_tensor_counts = [int(row.get("cache_tensor_count", 0)) for row in success_records]

    summary = {
        "status": "ok" if success_records else "error",
        "mode": s_mode,
        "ablation_mode": ablation_config.mode,
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": resolved_device,
        "dtype": resolved_dtype,
        "scheduler": args.scheduler,
        "timestep_spacing": args.timestep_spacing,
        "num_inference_steps": args.num_inference_steps,
        "image_size": args.image_size,
        "batch_size": 1,
        "seed": args.seed,
        "init_mode": args.init_mode,
        "reconstruct_t": args.reconstruct_t,
        "image_cfg_scale": round_float(resolved_image_cfg_scale),
        "guidance_scale": round_float(args.guidance_scale),
        "min_effective_steps": args.min_effective_steps,
        "vae_deterministic": bool(args.vae_deterministic),
        "refine_pass": bool(args.refine_pass),
        "refine_t": args.refine_t,
        "refine_steps": args.refine_steps,
        "num_requested_samples": len(records),
        "num_profiled_samples": len(success_records),
        "num_failed_samples": len(records) - len(success_records),
        "warnings": list(checkpoint_warnings),
        "model_stats": model_stats,
        "metrics": {
            "mean_total_core_ms": round_float(safe_stat_mean(core_values)),
            "std_total_core_ms": round_float(safe_stat_std(core_values)),
            "mean_total_e2e_ms": round_float(safe_stat_mean(e2e_values)),
            "std_total_e2e_ms": round_float(safe_stat_std(e2e_values)),
            "mean_peak_memory_allocated_mb": round_float(safe_stat_mean(alloc_values)),
            "mean_peak_memory_reserved_mb": round_float(safe_stat_mean(reserved_values)),
            "mean_cache_tensor_mb": round_float(safe_stat_mean(cache_tensor_values)),
            "mean_cache_storage_mb": round_float(safe_stat_mean(cache_storage_values)),
            "mean_cache_cuda_released_mb": round_float(safe_stat_mean(cache_released_values)),
            "max_cache_storage_mb": round_float(max(cache_storage_values)) if cache_storage_values else None,
            "cache_tensor_count": max(cache_tensor_counts) if cache_tensor_counts else 0,
        },
    }
    if not success_records:
        summary["error"] = "No samples were profiled successfully."
    return summary


def build_aggregate_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    model_stats = summary.get("model_stats") or {}
    totals = model_stats.get("totals") or {}
    metrics = summary.get("metrics") or {}
    return {
        "mode": summary.get("mode"),
        "ablation_mode": summary.get("ablation_mode"),
        "trainable_params_m": totals.get("trainable_params_m"),
        "total_params_m": totals.get("total_params_m"),
        "mean_total_core_ms": metrics.get("mean_total_core_ms"),
        "std_total_core_ms": metrics.get("std_total_core_ms"),
        "mean_total_e2e_ms": metrics.get("mean_total_e2e_ms"),
        "std_total_e2e_ms": metrics.get("std_total_e2e_ms"),
        "mean_peak_memory_allocated_mb": metrics.get("mean_peak_memory_allocated_mb"),
        "mean_peak_memory_reserved_mb": metrics.get("mean_peak_memory_reserved_mb"),
        "mean_cache_tensor_mb": metrics.get("mean_cache_tensor_mb"),
        "mean_cache_storage_mb": metrics.get("mean_cache_storage_mb"),
        "mean_cache_cuda_released_mb": metrics.get("mean_cache_cuda_released_mb"),
        "num_profiled_samples": summary.get("num_profiled_samples"),
        "scheduler": summary.get("scheduler"),
        "num_inference_steps": summary.get("num_inference_steps"),
        "status": summary.get("status"),
        "error": summary.get("error", ""),
    }


def build_error_payload(
    s_mode: str,
    ablation_mode: str,
    args: argparse.Namespace,
    error: Exception,
    extra_warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "status": "error",
        "mode": s_mode,
        "ablation_mode": ablation_mode,
        "scheduler": args.scheduler,
        "num_inference_steps": args.num_inference_steps,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "warnings": list(extra_warnings or []),
    }


def write_error_artifacts(mode_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    write_json(mode_dir / "model_stats.json", payload)
    write_json(mode_dir / "runtime_profile.json", payload)
    write_json(mode_dir / "efficiency_summary.json", payload)
    write_csv(
        mode_dir / "runtime_profile.csv",
        [payload],
        fieldnames=["status", "mode", "ablation_mode", "scheduler", "num_inference_steps", "error"],
    )
    return payload


def cleanup_bundle(bundle: Optional[Dict[str, Any]], device: torch.device) -> None:
    if not bundle:
        return
    for key in ("model", "text_encoder", "image_encoder", "vae", "tokenizer"):
        if key in bundle:
            del bundle[key]
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def profile_mode(
    args: argparse.Namespace,
    s_mode: str,
    subset: Sequence[Dict[str, Any]],
    device: torch.device,
    weight_dtype: torch.dtype,
    resolved_device: str,
    resolved_dtype: str,
    resolved_image_cfg_scale: float,
    inherited_warnings: Sequence[str],
) -> Dict[str, Any]:
    ablation_mode = mode_to_ablation_mode(s_mode)
    ablation_config = build_ablation_config(ablation_mode)
    mode_dir = ensure_dir(Path(args.output_root) / s_mode)
    bundle: Optional[Dict[str, Any]] = None

    try:
        log(f"[{s_mode}] Building profiler model ({ablation_mode})")
        bundle = build_profiler_bundle(args, s_mode, ablation_config, device, weight_dtype)
        checkpoint_warnings = list(inherited_warnings) + list(bundle.get("checkpoint_warnings", []))

        model: ProfilerSDModel = bundle["model"]
        model_stats = collect_model_stats(
            unet=model.unet,
            ref_unet=model.ref_unet,
            image_proj=model.proj,
            image_encoder=bundle["image_encoder"],
            text_encoder=bundle["text_encoder"],
            vae=bundle["vae"],
            ablation_config=ablation_config,
            s_mode=s_mode,
        )
        write_json(mode_dir / "model_stats.json", model_stats)

        warmup_count = min(max(0, int(args.warmup)), len(subset))
        images_dir = ensure_dir(mode_dir / "images") if args.save_images else None

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if warmup_count > 0:
            log(f"[{s_mode}] Warmup samples: {warmup_count}")
            for warmup_idx in range(warmup_count):
                try:
                    _ = profile_single_sample(
                        bundle=bundle,
                        item=subset[warmup_idx],
                        sample_index=warmup_idx,
                        args=args,
                        device=device,
                        weight_dtype=weight_dtype,
                        resolved_image_cfg_scale=resolved_image_cfg_scale,
                        save_image=False,
                        images_dir=None,
                    )
                except Exception as exc:
                    log(f"[{s_mode}] warmup sample {warmup_idx} failed: {exc}")

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        log(f"[{s_mode}] Profiling samples: {len(subset)}")
        records: List[Dict[str, Any]] = []
        for sample_index, item in enumerate(subset):
            try:
                record = profile_single_sample(
                    bundle=bundle,
                    item=item,
                    sample_index=sample_index,
                    args=args,
                    device=device,
                    weight_dtype=weight_dtype,
                    resolved_image_cfg_scale=resolved_image_cfg_scale,
                    save_image=bool(args.save_images),
                    images_dir=images_dir,
                )
            except Exception as exc:
                record = {
                    "sample_index": sample_index,
                    "mode": s_mode,
                    "ablation_mode": ablation_mode,
                    "category": item.get("category", ""),
                    "image_file": item.get("image_file", ""),
                    "cloth_file": item.get("cloth_file", ""),
                    "prompt_text": resolve_prompt(item),
                    "seed": int(args.seed) + int(sample_index),
                    "status": "error",
                    "error": str(exc),
                    "preprocess_ms": None,
                    "text_encode_ms": None,
                    "image_encode_ms": None,
                    "ref_unet_ms": None,
                    "denoise_ms": None,
                    "vae_decode_ms": None,
                    "postprocess_ms": None,
                    "save_image_ms": None,
                    "total_core_ms": None,
                    "total_e2e_ms": None,
                    "peak_memory_allocated_mb": None,
                    "peak_memory_reserved_mb": None,
                    "cache_tensor_count": None,
                    "cache_tensor_mb": None,
                    "cache_storage_mb": None,
                    "cache_cuda_released_mb": None,
                    "saved_image": "",
                }
                log(f"[{s_mode}] sample {sample_index} failed: {exc}")
            records.append(record)
            if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(subset):
                log(f"[{s_mode}] {sample_index + 1}/{len(subset)} samples processed")

        runtime_fieldnames = [
            "sample_index",
            "mode",
            "ablation_mode",
            "category",
            "image_file",
            "cloth_file",
            "prompt_text",
            "seed",
            "status",
            "preprocess_ms",
            "text_encode_ms",
            "image_encode_ms",
            "ref_unet_ms",
            "denoise_ms",
            "vae_decode_ms",
            "postprocess_ms",
            "save_image_ms",
            "total_core_ms",
            "total_e2e_ms",
            "peak_memory_allocated_mb",
            "peak_memory_reserved_mb",
            "cache_tensor_count",
            "cache_tensor_mb",
            "cache_storage_mb",
            "cache_cuda_released_mb",
            "saved_image",
            "error",
        ]
        write_csv(mode_dir / "runtime_profile.csv", records, runtime_fieldnames)
        write_json(
            mode_dir / "runtime_profile.json",
            {
                "mode": s_mode,
                "ablation_mode": ablation_mode,
                "records": records,
            },
        )

        summary = summarize_runtime_records(
            s_mode=s_mode,
            ablation_config=ablation_config,
            model_stats=model_stats,
            records=records,
            args=args,
            resolved_device=resolved_device,
            resolved_dtype=resolved_dtype,
            resolved_image_cfg_scale=resolved_image_cfg_scale,
            checkpoint_warnings=checkpoint_warnings,
        )
        write_json(mode_dir / "efficiency_summary.json", summary)
        return summary
    except Exception as exc:
        payload = build_error_payload(
            s_mode=s_mode,
            ablation_mode=ablation_mode,
            args=args,
            error=exc,
            extra_warnings=inherited_warnings,
        )
        write_error_artifacts(mode_dir, payload)
        log(f"[{s_mode}] failed: {exc}")
        return payload
    finally:
        cleanup_bundle(bundle, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone DressCode efficiency profiler for IMAGDressing.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file to load for all modes.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Base diffusion model path.",
    )
    parser.add_argument("--image_encoder_path", type=str, required=True, help="CLIP image encoder path.")
    parser.add_argument("--pretrained_vae_model_path", type=str, required=True, help="VAE model path.")
    parser.add_argument("--dresscode_root", type=str, default="", help="DressCode dataset root.")
    parser.add_argument("--subset_json", type=str, default="", help="Optional resolved subset JSON.")
    parser.add_argument(
        "--dresscode_category",
        type=str,
        default="all",
        choices=["all"] + DRESSCODE_CATEGORIES,
        help="DressCode category when subset_json is not provided.",
    )
    parser.add_argument(
        "--dresscode_test_order",
        type=str,
        default="paired",
        choices=VALID_TEST_ORDERS,
        help="DressCode test pair order when subset_json is not provided.",
    )
    parser.add_argument("--s_modes", type=str, default="S0,S1,S2", help="Comma-separated modes to profile.")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpmpp_2m_karras",
        choices=["ddim", "unipc", "dpmpp_2m", "dpmpp_2m_karras", "euler_a"],
        help="Sampling scheduler.",
    )
    parser.add_argument("--timestep_spacing", type=str, default="trailing", help="Scheduler timestep spacing.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Execution device, for example cpu or cuda:0.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"],
        help="Compute dtype.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup samples before recording.")
    parser.add_argument("--profile_samples", type=int, default=50, help="Number of samples to profile.")
    parser.add_argument("--save_images", action="store_true", help="Save generated images under output_root/<mode>/images.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/efficiency_profile_dresscode",
        help="Output directory for profiler artifacts.",
    )
    parser.add_argument(
        "--init_mode",
        type=str,
        default="from_noisy_gt",
        choices=["from_noisy_gt", "random"],
        help="Latent initialization mode.",
    )
    parser.add_argument("--reconstruct_t", type=int, default=600, help="Start timestep for from_noisy_gt mode.")
    parser.add_argument("--image_cfg_scale", type=float, default=1.0, help="Image CFG scale.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Backward-compatible alias used when image_cfg_scale remains 1.0.",
    )
    parser.add_argument(
        "--min_effective_steps",
        type=int,
        default=10,
        help="Minimum effective steps after reconstruct_t selection.",
    )
    parser.add_argument("--vae_deterministic", action="store_true", help="Use posterior mode instead of sampling.")
    parser.add_argument("--refine_pass", action="store_true", help="Enable a second refine denoising pass.")
    parser.add_argument("--refine_t", type=int, default=150, help="Refine start timestep.")
    parser.add_argument("--refine_steps", type=int, default=10, help="Refine scheduler steps.")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size for VAE transforms.")
    parser.add_argument(
        "--subset_sampling",
        type=str,
        default="first",
        choices=["first", "random"],
        help="Subset selection mode when more than profile_samples items are available.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.dresscode_root:
        raise RuntimeError("--dresscode_root is required for DressCode image resolution.")

    output_root = ensure_dir(Path(args.output_root))
    s_modes = resolve_s_modes(args.s_modes)
    device, weight_dtype, resolved_dtype, device_warnings = resolve_device_and_dtype(args.device, args.dtype)
    resolved_device = str(device)
    resolved_image_cfg_scale = float(args.image_cfg_scale)
    if resolved_image_cfg_scale == 1.0 and float(args.guidance_scale) != 1.0:
        log("[WARN] guidance_scale is deprecated; using it as image_cfg_scale.")
        resolved_image_cfg_scale = float(args.guidance_scale)

    if args.init_mode != "from_noisy_gt" and args.reconstruct_t is not None:
        log("[INFO] reconstruct_t is ignored when init_mode != from_noisy_gt")
    if args.min_effective_steps is not None and int(args.min_effective_steps) <= 0:
        args.min_effective_steps = None

    subset = resolve_subset(args, output_root)
    log(
        f"[INFO] device={resolved_device} dtype={resolved_dtype} "
        f"scheduler={args.scheduler} steps={args.num_inference_steps} subset={len(subset)}"
    )

    summaries: List[Dict[str, Any]] = []
    for s_mode in s_modes:
        summary = profile_mode(
            args=args,
            s_mode=s_mode,
            subset=subset,
            device=device,
            weight_dtype=weight_dtype,
            resolved_device=resolved_device,
            resolved_dtype=resolved_dtype,
            resolved_image_cfg_scale=resolved_image_cfg_scale,
            inherited_warnings=device_warnings,
        )
        summaries.append(summary)

    aggregate_rows = [build_aggregate_row(summary) for summary in summaries]
    write_json(output_root / "efficiency_table.json", aggregate_rows)
    write_csv(
        output_root / "efficiency_table.csv",
        aggregate_rows,
        fieldnames=[
            "mode",
            "ablation_mode",
            "trainable_params_m",
            "total_params_m",
            "mean_total_core_ms",
            "std_total_core_ms",
            "mean_total_e2e_ms",
            "std_total_e2e_ms",
            "mean_peak_memory_allocated_mb",
            "mean_peak_memory_reserved_mb",
            "mean_cache_tensor_mb",
            "mean_cache_storage_mb",
            "mean_cache_cuda_released_mb",
            "num_profiled_samples",
            "scheduler",
            "num_inference_steps",
            "status",
            "error",
        ],
    )
    log(f"[DONE] Wrote profiler outputs to {output_root}")


if __name__ == "__main__":
    main()
