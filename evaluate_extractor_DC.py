#!/usr/bin/env python3
# Changes: lazy-load heavy deps for prompt-only modes, fix output_dir default,
# add streaming prompt-cache checks and auto handling on cache_only misses.
"""
评估 DressCode Extractor 模型的生成质量
使用 DressCode 测试集 pairs，生成 garment 图片，并与 Ground Truth 计算 FID 和 KID 指标
"""

import os
import sys
import json
import argparse
import random
import gc
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


CUDA_POISONED = False


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


def _build_scheduler(name: str, timestep_spacing: str):
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        UniPCMultistepScheduler,
        EulerAncestralDiscreteScheduler,
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
    if name == "ddim":
        return base
    if name == "unipc":
        return UniPCMultistepScheduler.from_config(base.config)
    if name == "dpmpp_2m":
        return DPMSolverMultistepScheduler.from_config(
            base.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=False,
        )
    if name == "dpmpp_2m_karras":
        return DPMSolverMultistepScheduler.from_config(
            base.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(base.config)
    raise ValueError(f"Unsupported scheduler: {name}")


def _set_scheduler_timesteps(scheduler, num_inference_steps, device):
    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(num_inference_steps)


def _timesteps_to_list(values):
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    else:
        tensor = torch.tensor(values)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    return tensor.flatten().tolist()


def as_timestep_tensor(t, device, batch_size):
    if isinstance(t, torch.Tensor):
        t_tensor = t.to(device=device)
    else:
        t_tensor = torch.tensor(t, device=device)
    if t_tensor.ndim == 0:
        t_tensor = t_tensor.unsqueeze(0)
    t_tensor = t_tensor.flatten()
    if t_tensor.is_floating_point():
        t_tensor = torch.round(t_tensor)
    t_tensor = t_tensor.to(dtype=torch.long)
    if batch_size is not None:
        batch_size = int(batch_size)
        if t_tensor.numel() == 1 and batch_size != 1:
            t_tensor = t_tensor.expand(batch_size)
        elif t_tensor.numel() != batch_size:
            t_tensor = t_tensor[:1].expand(batch_size)
    return t_tensor


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


def _basename(value):
    if not value:
        return ""
    return Path(str(value).replace("\\", "/")).name


def make_pair_key(category, image_file, cloth_file):
    category = "" if category is None else str(category).strip()
    image_file = _basename(image_file)
    cloth_file = _basename(cloth_file)
    return f"{category}|||{image_file}|||{cloth_file}"


def _make_pair_key_raw(category, image_file, cloth_file):
    category = "" if category is None else str(category).strip()
    image_file = "" if image_file is None else str(image_file)
    cloth_file = "" if cloth_file is None else str(cloth_file)
    return f"{category}|||{image_file}|||{cloth_file}"


_PROMPT_KEY_CASEFOLD = os.name == "nt"


def _casefold_prompt_key(key):
    if key is None:
        return key
    text = str(key)
    return text.casefold() if _PROMPT_KEY_CASEFOLD else text


def normalize_prompt_cache_key(key):
    if key is None:
        return key
    text = str(key).strip()
    if not text:
        return text
    parsed = _split_pair_key_any(text)
    if not parsed:
        return _casefold_prompt_key(text)
    category, image_file, cloth_file = parsed
    if category:
        normalized = make_pair_key(category, image_file, cloth_file)
    else:
        normalized = f"{image_file}|||{cloth_file}"
    return _casefold_prompt_key(normalized)


def _split_pair_key_any(key):
    if key is None:
        return None
    text = str(key).strip()
    if not text:
        return None
    if "|||" in text:
        parts = [p.strip() for p in text.split("|||")]
        if len(parts) >= 3:
            category = parts[0] or None
            image_part = parts[1]
            cloth_part = "|||".join(parts[2:]) if len(parts) > 3 else parts[2]
        elif len(parts) == 2:
            category = None
            image_part, cloth_part = parts
        else:
            return None
    elif "|" in text:
        parts = [p.strip() for p in text.split("|")]
        if len(parts) >= 3:
            category = parts[0] or None
            image_part = parts[1]
            cloth_part = "|".join(parts[2:]) if len(parts) > 3 else parts[2]
        elif len(parts) == 2:
            category = None
            image_part, cloth_part = parts
        else:
            return None
    else:
        return None

    if not image_part or not cloth_part:
        return None
    image_base = Path(image_part.replace("\\", "/")).name
    cloth_base = Path(cloth_part.replace("\\", "/")).name
    return category, image_base, cloth_base


def _strip_extension(filename):
    if not filename:
        return filename
    root, ext = os.path.splitext(filename)
    return root if root else filename


def canonicalize_cache_key(raw_key):
    if raw_key is None:
        return []
    raw_key = str(raw_key).strip()
    if not raw_key:
        return []

    seen = set()
    keys = []

    def add(key):
        if not key:
            return
        key_str = str(key).strip()
        if not key_str:
            return
        key_str = _casefold_prompt_key(key_str)
        if key_str in seen:
            return
        seen.add(key_str)
        keys.append(key_str)

    add(raw_key)

    parsed = _split_pair_key_any(raw_key)
    if not parsed:
        return keys

    category, image_part, cloth_part = parsed
    image_variants = []
    cloth_variants = []
    for value, dest in ((image_part, image_variants), (cloth_part, cloth_variants)):
        base = Path(str(value).replace("\\", "/")).name
        dest.append(base)
        no_ext = _strip_extension(base)
        if no_ext and no_ext != base:
            dest.append(no_ext)

    categories = []
    if category:
        categories.append(str(category).strip())
    categories.append("")

    for sep in ("|||", "|"):
        for cat in categories:
            for img in image_variants:
                for clo in cloth_variants:
                    if not img or not clo:
                        continue
                    if cat:
                        add(f"{cat}{sep}{img}{sep}{clo}")
                        add(f"{cat}{sep}{clo}{sep}{img}")
                    else:
                        add(f"{img}{sep}{clo}")
                        add(f"{clo}{sep}{img}")

    return keys


def _is_main_prompt_key(key):
    return _split_pair_key_any(key) is not None


def _get_basename_variants(value):
    if not value:
        return []
    base = Path(str(value).replace("\\", "/")).name
    if not base:
        return []
    variants = [base]
    no_ext = _strip_extension(base)
    if no_ext and no_ext != base:
        variants.append(no_ext)
    return variants


def _build_prompt_key_groups(category, image_file, cloth_file):
    category_text = "" if category is None else str(category).strip()
    image_variants = _get_basename_variants(image_file)
    cloth_variants = _get_basename_variants(cloth_file)
    keys = []
    seen = set()
    source_map = {}

    def add_key(candidate, source):
        if not candidate:
            return
        key_text = _casefold_prompt_key(str(candidate).strip())
        if not key_text or key_text.replace("|", "") == "":
            return
        if key_text in seen:
            return
        seen.add(key_text)
        keys.append(key_text)
        source_map[key_text] = source

    def add_group(cat_value, source):
        cat_text = "" if cat_value is None else str(cat_value).strip()
        for sep in ("|||", "|"):
            for img in image_variants:
                for clo in cloth_variants:
                    if not img or not clo:
                        continue
                    if cat_text:
                        add_key(f"{cat_text}{sep}{img}{sep}{clo}", source)
                    else:
                        add_key(f"{img}{sep}{clo}", source)
            for img in image_variants:
                for clo in cloth_variants:
                    if not img or not clo:
                        continue
                    if cat_text:
                        add_key(f"{cat_text}{sep}{clo}{sep}{img}", source)
                    else:
                        add_key(f"{clo}{sep}{img}", source)

    if category_text:
        add_group(category_text, "cache_hit")

    add_group("", "cache_hit_no_category")

    category_folded = _casefold_prompt_key(category_text)
    for cat in ("upper_body", "lower_body", "dresses"):
        if _casefold_prompt_key(cat) == category_folded:
            continue
        add_group(cat, "cache_hit_cross_category")

    return keys, source_map


def _build_prompt_keys(category, image_file, cloth_file):
    keys, _ = _build_prompt_key_groups(category, image_file, cloth_file)
    return keys


def canonicalize_prompt_key(key: str) -> str:
    if key is None:
        return key
    text = str(key).strip()
    if not text:
        return text
    normalized = normalize_prompt_cache_key(text)
    return normalized if isinstance(normalized, str) and normalized else _casefold_prompt_key(text)


def _build_prompt_candidate_index(test_data, default_category):
    candidate_to_indices = {}
    main_keys = []
    item_candidates = []
    requested_samples = []

    for idx, item in enumerate(test_data):
        category = item.get("category") or default_category or "unknown"
        image_file = str(item.get("image_file", ""))
        cloth_file = str(item.get("cloth_file", ""))
        raw_key = make_pair_key(category, image_file, cloth_file)
        main_key = normalize_prompt_cache_key(raw_key)
        candidate_keys, _ = _build_prompt_key_groups(category, image_file, cloth_file)
        if not candidate_keys and main_key:
            candidate_keys = [main_key]
        if main_key and main_key not in candidate_keys:
            candidate_keys.insert(0, main_key)
        main_keys.append(main_key)
        item_candidates.append(candidate_keys)
        for key in candidate_keys:
            candidate_to_indices.setdefault(key, []).append(idx)
        if len(requested_samples) < 5 and main_key:
            requested_samples.append(main_key)
    return candidate_to_indices, main_keys, item_candidates, requested_samples


def _extract_entry_candidate_keys(entry):
    raw_keys = []
    if not isinstance(entry, dict):
        return raw_keys
    key = _extract_cache_key(entry)
    if key:
        raw_keys.append(key)
    category = entry.get("category")
    image_file = entry.get("image_file")
    cloth_file = entry.get("cloth_file")
    if category or image_file or cloth_file:
        raw_keys.append(_make_pair_key_raw(category, image_file, cloth_file))
    return raw_keys


def _check_prompts_only_streaming(
    test_data,
    prompts_cache_path,
    default_category,
    prompt_mode="cache_only",
    preview_count=10,
):
    if not prompts_cache_path or not os.path.exists(prompts_cache_path):
        raise RuntimeError(f"prompts_cache not found: {prompts_cache_path}")

    candidate_to_indices, main_keys, item_candidates, requested_samples = _build_prompt_candidate_index(
        test_data, default_category
    )
    required_keys = [key for key in main_keys if key]
    required_set = set(required_keys)
    hits = [False] * len(main_keys)
    cache_sample_keys = []
    cache_sample_seen = set()
    remaining = len(main_keys)

    def register_sample(raw_key):
        if len(cache_sample_keys) >= 5:
            return
        normalized = normalize_prompt_cache_key(raw_key)
        if not normalized:
            return
        if normalized in cache_sample_seen:
            return
        cache_sample_seen.add(normalized)
        cache_sample_keys.append(normalized)

    ext = os.path.splitext(prompts_cache_path)[1].lower()
    if ext == ".jsonl":
        with open(prompts_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for raw_key in _extract_entry_candidate_keys(obj):
                    register_sample(raw_key)
                    for alias in canonicalize_cache_key(raw_key):
                        indices = candidate_to_indices.get(alias)
                        if not indices:
                            continue
                        for idx in indices:
                            if not hits[idx]:
                                hits[idx] = True
                                remaining -= 1
                        if remaining == 0:
                            break
                    if remaining == 0:
                        break
                if remaining == 0:
                    break
    elif ext == ".json":
        with open(prompts_cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = []
        if isinstance(data, dict):
            entries = [{"key": k, "prompt": v} for k, v in data.items()]
        elif isinstance(data, list):
            entries = data
        for obj in entries:
            for raw_key in _extract_entry_candidate_keys(obj):
                register_sample(raw_key)
                for alias in canonicalize_cache_key(raw_key):
                    indices = candidate_to_indices.get(alias)
                    if not indices:
                        continue
                    for idx in indices:
                        if not hits[idx]:
                            hits[idx] = True
                            remaining -= 1
                    if remaining == 0:
                        break
                if remaining == 0:
                    break
            if remaining == 0:
                break
    else:
        raise RuntimeError(f"Unsupported prompts_cache format: {ext}")

    missing = []
    missing_candidates = {}
    cache_hit = 0
    for idx, main_key in enumerate(main_keys):
        if main_key and hits[idx]:
            cache_hit += 1
        else:
            if main_key:
                missing.append(main_key)
                if main_key not in missing_candidates:
                    missing_candidates[main_key] = item_candidates[idx][:10]

    if requested_samples:
        print(f"[CHECK PROMPT] Requested key samples: {', '.join(requested_samples)}")
    if cache_sample_keys:
        print(f"[CHECK PROMPT] Cache key samples: {', '.join(cache_sample_keys)}")
    missing_count = len(missing)
    print(f"[CHECK PROMPT STATS] total={len(main_keys)} cache_hit={cache_hit} missing={missing_count}")

    if missing_count:
        preview_missing = missing[:20]
        print(f"[CHECK PROMPT] Missing key samples: {', '.join(preview_missing)}")
        for key in preview_missing:
            candidates = missing_candidates.get(key, [])
            print(f"[CHECK PROMPT] Missing key={key} candidates={candidates}")
        print(
            "[CHECK PROMPT] Missing prompts detected. "
            "Run --prepare_eval_assets to build missing prompts for subset_json."
        )

    return {
        "total": len(main_keys),
        "cache_hit": cache_hit,
        "missing": missing_count,
        "missing_keys": missing,
    }


def _prefer_longer_prompt(cache, key, prompt):
    if not key or prompt is None:
        return
    prompt_text = str(prompt)
    if prompt_text == "":
        return
    current = cache.get(key)
    if current is None or len(prompt_text) > len(str(current)):
        cache[key] = prompt


def _maybe_add_prompt(cache, key, prompt):
    _prefer_longer_prompt(cache, key, prompt)
    canonical = canonicalize_prompt_key(key)
    if canonical and canonical != key:
        _prefer_longer_prompt(cache, canonical, prompt)


def _set_prompt(cache, key, prompt):
    _prefer_longer_prompt(cache, key, prompt)
    canonical = canonicalize_prompt_key(key)
    if canonical and canonical != key:
        _prefer_longer_prompt(cache, canonical, prompt)


def _coerce_prompt_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, dict):
        for field in ("prompt", "text", "caption", "prompts"):
            if field in value:
                return _coerce_prompt_text(value[field])
    return None


def _extract_cache_key(entry):
    if not isinstance(entry, dict):
        return None
    for field in ("key", "pair_key", "id"):
        value = entry.get(field)
        if value is not None:
            return str(value)
    return None


def _extract_prompt_from_entry(entry):
    if not isinstance(entry, dict):
        return None
    for field in ("prompt", "text", "caption", "prompts"):
        if field in entry:
            return _coerce_prompt_text(entry[field])
    return None


def _add_prompt_cache_entry(
    cache,
    prompt,
    key=None,
    category=None,
    image_file=None,
    cloth_file=None,
    legacy_stats=None,
):
    raw_key = None
    if key:
        raw_key = str(key)
    elif category or image_file or cloth_file:
        raw_key = _make_pair_key_raw(category, image_file, cloth_file)
    if not raw_key:
        return
    raw_key = str(raw_key).strip()
    if not raw_key or raw_key.replace("|", "") == "":
        return
    alias_keys = canonicalize_cache_key(raw_key)
    if (category or image_file or cloth_file) and _split_pair_key_any(raw_key) is None:
        fallback_key = _make_pair_key_raw(category, image_file, cloth_file)
        if fallback_key and fallback_key != raw_key:
            alias_keys.extend(canonicalize_cache_key(fallback_key))
    for alias in alias_keys:
        _prefer_longer_prompt(cache, alias, prompt)
    if legacy_stats is not None:
        normalized_key = normalize_prompt_cache_key(raw_key)
        raw_key_norm = _casefold_prompt_key(raw_key)
        if normalized_key != raw_key_norm:
            legacy_stats["mapped"] += 1


def _collect_normalized_prompt_key(keys, key=None, category=None, image_file=None, cloth_file=None):
    raw_key = None
    if key:
        raw_key = str(key)
    elif category or image_file or cloth_file:
        raw_key = _make_pair_key_raw(category, image_file, cloth_file)
    if not raw_key:
        return
    raw_key = str(raw_key).strip()
    if not raw_key or raw_key.replace("|", "") == "":
        return
    alias_keys = canonicalize_cache_key(raw_key)
    if (category or image_file or cloth_file) and _split_pair_key_any(raw_key) is None:
        fallback_key = _make_pair_key_raw(category, image_file, cloth_file)
        if fallback_key and fallback_key != raw_key:
            alias_keys.extend(canonicalize_cache_key(fallback_key))
    for alias in alias_keys:
        keys.add(alias)


def load_prompts_cache(cache_path):
    cache = {}
    if not cache_path:
        return cache
    if not os.path.exists(cache_path):
        print(f"[WARN] prompts_cache not found: {cache_path}")
        return cache
    legacy_stats = {"mapped": 0}
    key_stats = {"triple": 0, "single": 0, "other": 0, "parse_failed": 0}
    key_samples = []
    normalized_samples = []
    normalized_seen = set()

    def register_key(raw_key):
        if raw_key is None:
            return
        raw_key_str = str(raw_key).strip()
        if not raw_key_str or raw_key_str.replace("|", "") == "":
            return
        if len(key_samples) < 5:
            key_samples.append(raw_key_str)
        normalized_key = normalize_prompt_cache_key(raw_key_str)
        if (
            normalized_key
            and normalized_key not in normalized_seen
            and len(normalized_samples) < 5
        ):
            normalized_samples.append(normalized_key)
            normalized_seen.add(normalized_key)
        if "|||" in raw_key_str:
            key_stats["triple"] += 1
        elif "|" in raw_key_str:
            key_stats["single"] += 1
        else:
            key_stats["other"] += 1
        if _split_pair_key_any(raw_key_str) is None:
            key_stats["parse_failed"] += 1

    ext = os.path.splitext(cache_path)[1].lower()
    if ext == ".jsonl":
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = _extract_prompt_from_entry(obj)
                if prompt is None:
                    continue
                key = _extract_cache_key(obj)
                category = obj.get("category")
                image_file = obj.get("image_file")
                cloth_file = obj.get("cloth_file")
                if key:
                    register_key(key)
                    _add_prompt_cache_entry(
                        cache,
                        prompt,
                        key=key,
                        legacy_stats=legacy_stats,
                    )
                if category or image_file or cloth_file:
                    field_key = _make_pair_key_raw(category, image_file, cloth_file)
                    register_key(field_key)
                    _add_prompt_cache_entry(
                        cache,
                        prompt,
                        category=category,
                        image_file=image_file,
                        cloth_file=cloth_file,
                        legacy_stats=None if key else legacy_stats,
                    )
    elif ext == ".json":
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key, value in data.items():
                prompt = _coerce_prompt_text(value)
                if prompt is None:
                    continue
                register_key(key)
                _add_prompt_cache_entry(cache, prompt, key=key, legacy_stats=legacy_stats)
        elif isinstance(data, list):
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                prompt = _extract_prompt_from_entry(obj)
                if prompt is None:
                    continue
                key = _extract_cache_key(obj)
                category = obj.get("category")
                image_file = obj.get("image_file")
                cloth_file = obj.get("cloth_file")
                if key:
                    register_key(key)
                    _add_prompt_cache_entry(
                        cache,
                        prompt,
                        key=key,
                        legacy_stats=legacy_stats,
                    )
                if category or image_file or cloth_file:
                    field_key = _make_pair_key_raw(category, image_file, cloth_file)
                    register_key(field_key)
                    _add_prompt_cache_entry(
                        cache,
                        prompt,
                        category=category,
                        image_file=image_file,
                        cloth_file=cloth_file,
                        legacy_stats=None if key else legacy_stats,
                    )
    else:
        print(f"[WARN] Unsupported prompt cache format: {ext}")
    if legacy_stats["mapped"] > 0:
        print(
            f"[PromptCache] Detected legacy keys with paths; mapped {legacy_stats['mapped']} entries to normalized keys."
        )
    if key_samples:
        print(f"[PromptCache] Sample keys: {', '.join(key_samples)}")
    if normalized_samples:
        print(f"[PromptCache] Sample normalized keys: {', '.join(normalized_samples)}")
    print(
        "[PromptCache] Key formats: "
        f"triple={key_stats['triple']} single={key_stats['single']} "
        f"other={key_stats['other']} parse_failed={key_stats['parse_failed']}"
    )
    return cache


def load_checkpoint_prompt_config(checkpoint_path):
    if not checkpoint_path:
        return {}
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] checkpoint not found for prompt_config: {checkpoint_path}")
        return {}
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] Failed to load checkpoint for prompt_config: {exc}")
        return {}
    if not isinstance(ckpt, dict):
        return {}
    prompt_config = ckpt.get("prompt_config")
    if not isinstance(prompt_config, dict):
        return {}
    return prompt_config


def _resolve_prompts_cache_path(args, checkpoint_prompt_config):
    if args.prompts_cache:
        return args.prompts_cache, "cli", os.path.exists(args.prompts_cache)

    env_path = os.environ.get("DRESSCODE_PROMPTS_CACHE", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path, "env", True

    default_path = r"E:\BaiduNetdiskDownload\DressCode\prompts_dresscode_cache.jsonl"
    if os.path.exists(default_path):
        return default_path, "default", True

    if args.dresscode_root:
        root_path = os.path.join(args.dresscode_root, "prompts_dresscode_cache.jsonl")
        if os.path.exists(root_path):
            return root_path, "dresscode_root", True

    ckpt_cache = checkpoint_prompt_config.get("prompts_cache_path") if checkpoint_prompt_config else ""
    if ckpt_cache and os.path.exists(ckpt_cache):
        return ckpt_cache, "checkpoint", True

    return "", "not_found", False


def _get_main_prompt_key(item, default_category):
    category = item.get("category") or default_category or ""
    image_file = str(item.get("image_file", ""))
    cloth_file = str(item.get("cloth_file", ""))
    raw_key = make_pair_key(category, image_file, cloth_file)
    return normalize_prompt_cache_key(raw_key)


def _load_prompt_cache_keys(cache_path):
    keys = set()
    if not cache_path or not os.path.exists(cache_path):
        return keys
    ext = os.path.splitext(cache_path)[1].lower()
    if ext == ".jsonl":
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                key = _extract_cache_key(obj)
                category = obj.get("category")
                image_file = obj.get("image_file")
                cloth_file = obj.get("cloth_file")
                _collect_normalized_prompt_key(
                    keys,
                    key=key,
                    category=category,
                    image_file=image_file,
                    cloth_file=cloth_file,
                )
    elif ext == ".json":
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in data.keys():
                _collect_normalized_prompt_key(keys, key=key)
        elif isinstance(data, list):
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                key = _extract_cache_key(obj)
                category = obj.get("category")
                image_file = obj.get("image_file")
                cloth_file = obj.get("cloth_file")
                _collect_normalized_prompt_key(
                    keys,
                    key=key,
                    category=category,
                    image_file=image_file,
                    cloth_file=cloth_file,
                )
    return keys


def _encode_image_to_data_url(path):
    img = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _build_prompt_messages(cloth_path, person_path, image_mode):
    system_text = (
        "你是服装描述助手，只描述服装本身材质/颜色/版型/领口/袖长/图案/装饰/闭合方式，不描述人物与背景"
    )
    if image_mode == "person_plus_cloth":
        user_text = (
            "The first image is the garment. The second image shows the person wearing it. "
            "Describe only the garment in 1-2 English sentences. No bullet points or line breaks."
        )
    else:
        user_text = (
            "Describe the garment in the image in 1-2 English sentences. "
            "Do not mention the person or background. No bullet points or line breaks."
        )
    content = [{"type": "text", "text": user_text}]
    content.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_url(cloth_path)}})
    if image_mode == "person_plus_cloth" and person_path:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_url(person_path)}})
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]


def _create_openai_client(provider, api_base, api_key):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for prompt generation.") from exc
    if provider == "openai":
        if api_base:
            return OpenAI(api_key=api_key, base_url=api_base)
        return OpenAI(api_key=api_key)
    if not api_base:
        raise RuntimeError("--prompt_api_base is required for dashscope_openai_compat.")
    return OpenAI(api_key=api_key, base_url=api_base)


def _call_prompt_api(client, model, messages, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    text = " ".join(text.replace("\n", " ").replace("\t", " ").split()).strip()
    text = text.strip().strip('"').strip("'")
    if not text:
        raise RuntimeError("Empty prompt returned by API.")
    return text


def build_prompts_cache(
    dresscode_root,
    items,
    prompts_cache_path,
    default_category,
    prompt_provider,
    prompt_model,
    prompt_api_base,
    prompt_api_key_env,
    prompt_api_key_file,
    prompt_image_mode,
    prompt_max_tokens,
    prompt_temperature,
    prompt_concurrency,
    prompt_overwrite,
):
    if not prompts_cache_path:
        raise ValueError("--prompts_cache is required for prompt generation.")

    ext = os.path.splitext(prompts_cache_path)[1].lower()
    if ext not in (".jsonl", ".json"):
        raise ValueError("prompts_cache must be .jsonl or .json")

    existing_keys = _load_prompt_cache_keys(prompts_cache_path)
    seen_keys = set()
    items_to_generate = []
    skipped_count = 0

    for item in items:
        key = _get_main_prompt_key(item, default_category)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        if key in existing_keys and not prompt_overwrite:
            skipped_count += 1
            continue
        items_to_generate.append(item)

    total = len(seen_keys)
    print(
        f"[PromptBuilder] total={total} cached={len(existing_keys)} to_generate={len(items_to_generate)} overwrite={bool(prompt_overwrite)}"
    )

    if not items_to_generate:
        print("[PromptBuilder] prompts_cache already complete; skip generation.")
        return {"total": total, "generated": 0, "skipped": skipped_count, "failed": 0}

    api_key = None
    if prompt_api_key_file:
        if not os.path.exists(prompt_api_key_file):
            raise RuntimeError(f"prompt_api_key_file not found: {prompt_api_key_file}")
        with open(prompt_api_key_file, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
    if not api_key:
        api_key = os.environ.get(prompt_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {prompt_api_key_env} in the environment or provide --prompt_api_key_file."
        )

    client = _create_openai_client(prompt_provider, prompt_api_base, api_key)

    os.makedirs(os.path.dirname(prompts_cache_path) or ".", exist_ok=True)

    lock = threading.Lock()
    generated = 0
    failed = 0
    generated_records = []

    def worker(item):
        key = _get_main_prompt_key(item, default_category)
        category = item.get("category") or default_category or "unknown"
        cloth_path = resolve_dresscode_cloth_path(dresscode_root, category, item.get("cloth_file", ""))
        person_path = resolve_dresscode_person_path(dresscode_root, category, item.get("image_file", ""))

        if not os.path.exists(cloth_path):
            return {"key": key, "prompt": None, "error": f"cloth not found: {cloth_path}"}
        if prompt_image_mode == "person_plus_cloth" and not os.path.exists(person_path):
            return {"key": key, "prompt": None, "error": f"person not found: {person_path}"}

        try:
            messages = _build_prompt_messages(
                cloth_path,
                person_path if prompt_image_mode == "person_plus_cloth" else None,
                prompt_image_mode,
            )
            prompt = _call_prompt_api(
                client,
                prompt_model,
                messages,
                prompt_temperature,
                prompt_max_tokens,
            )
            record = {
                "key": key,
                "prompt": prompt,
                "text": prompt,
                "category": category,
                "image_file": _basename(item.get("image_file", "")),
                "cloth_file": _basename(item.get("cloth_file", "")),
            }
            if ext == ".jsonl":
                line = json.dumps(record, ensure_ascii=False)
                with lock:
                    with open(prompts_cache_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
            return {"key": key, "prompt": prompt, "record": record, "error": None}
        except Exception as exc:
            return {"key": key, "prompt": None, "error": str(exc)}

    max_workers = max(1, int(prompt_concurrency))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(worker, item): item for item in items_to_generate}
        for future in as_completed(future_to_item):
            result = future.result()
            if result.get("error"):
                failed += 1
                print(f"[PromptBuilder] ERROR key={result.get('key')}: {result.get('error')}")
                continue
            generated += 1
            if ext == ".json":
                generated_records.append(result["record"])

    if ext == ".json" and generated_records:
        existing_map = {}
        if os.path.exists(prompts_cache_path):
            with open(prompts_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    existing_map.update(data)
        for record in generated_records:
            existing_map[record["key"]] = record["prompt"]
        with open(prompts_cache_path, "w", encoding="utf-8") as f:
            json.dump(existing_map, f, ensure_ascii=False, indent=2)

    print(
        f"[PromptBuilder] done generated={generated} skipped={skipped_count} failed={failed} cache={prompts_cache_path}"
    )
    return {"total": total, "generated": generated, "skipped": skipped_count, "failed": failed}


def _normalize_text(value):
    if isinstance(value, list):
        value = value[0] if value else ""
    if value is None:
        return ""
    return str(value)


def prepare_prompts_for_items(
    test_data,
    prompts_cache,
    prompt_mode,
    prompt_fallback,
    log_first_n_prompts,
    strict_empty_prompt,
    strict_prompt_cache,
    default_category,
):
    valid_modes = {"cache_only", "cache_or_fallback", "fallback_only", "empty_only"}
    if prompt_mode not in valid_modes:
        raise ValueError(f"Invalid prompt_mode: {prompt_mode}. Must be one of {sorted(valid_modes)}")
    total = 0
    cache_hit = 0
    fallback_count = 0
    empty_count = 0
    missing_keys = []
    missing_set = set()
    missing_info = None

    for idx, item in enumerate(test_data):
        total += 1
        category = item.get("category") or default_category or "unknown"
        image_file = str(item.get("image_file", ""))
        cloth_file = str(item.get("cloth_file", ""))
        raw_key = make_pair_key(category, image_file, cloth_file)
        main_key = normalize_prompt_cache_key(raw_key)
        candidate_keys, source_map = _build_prompt_key_groups(category, image_file, cloth_file)
        if idx < 3:
            print(f"[PromptKey] idx={idx} normalized={main_key}")
        if not candidate_keys and main_key:
            candidate_keys = [main_key]
            source_map = {main_key: "cache_hit"}
        if main_key and main_key not in candidate_keys:
            candidate_keys.insert(0, main_key)
            source_map.setdefault(main_key, "cache_hit")

        prompt = ""
        source = "empty"
        key_used = None

        if prompt_mode == "fallback_only":
            prompt = prompt_fallback or ""
            source = "fallback"
            fallback_count += 1
        elif prompt_mode != "empty_only":
            for key in candidate_keys:
                if key in prompts_cache:
                    prompt = prompts_cache[key]
                    source = source_map.get(key, "cache_hit")
                    key_used = key
                    cache_hit += 1
                    break

        cache_miss = key_used is None and prompt_mode in {"cache_only", "cache_or_fallback"}
        if cache_miss:
            if main_key:
                if main_key not in missing_set:
                    missing_keys.append(main_key)
                    missing_set.add(main_key)
                if missing_info is None:
                    missing_info = {
                        "expected_key": main_key,
                        "category": category,
                        "image_file": image_file,
                        "cloth_file": cloth_file,
                        "candidate_keys": candidate_keys[:10],
                    }
            if prompt_mode == "cache_only":
                prompt = ""
                source = "missing"
            elif prompt_mode == "cache_or_fallback" and prompt_fallback:
                prompt = prompt_fallback
                source = "fallback"
                fallback_count += 1
            elif strict_empty_prompt:
                if main_key:
                    print(f"[PROMPT MISS] key={main_key}")
                raise RuntimeError(
                    f"Prompt cache miss (mode={prompt_mode}, key={main_key})."
                )
            else:
                prompt = ""
                source = "empty"

        prompt = _normalize_text(prompt)
        if prompt == "" and source not in {"empty", "missing"}:
            source = "empty"

        if source == "empty":
            empty_count += 1
            if strict_empty_prompt:
                raise RuntimeError(
                    f"Empty prompt encountered (mode={prompt_mode}, key={main_key})."
                )

        item["text"] = prompt

        if log_first_n_prompts and idx < log_first_n_prompts:
            key_msg = key_used or main_key
            print(f"[PROMPT] idx={idx} source={source} key={key_msg} text={prompt}")

    missing_count = len(missing_keys)
    empty_ratio = (empty_count / total) if total else 0.0
    print(
        f"[PROMPT STATS] total={total} cache_hit={cache_hit} fallback={fallback_count} empty={empty_count}"
    )
    print(f"[PromptCache] hit={cache_hit} fallback={fallback_count} empty={empty_count}")
    if missing_count:
        print(f"[PROMPT MISSING] count={missing_count}")

    if prompt_mode == "cache_only" and missing_count > 0:
        preview_list = missing_keys[:20]
        preview = ", ".join(preview_list)
        for key in preview_list:
            print(f"[PROMPT MISS] key={key}")
        detail_msg = ""
        suffix_matches = []
        if missing_info:
            detail_msg = (
                "\nFirst miss details:"
                f"\n  expected_key={missing_info.get('expected_key')}"
                f"\n  category={missing_info.get('category')}"
                f"\n  image_file={missing_info.get('image_file')}"
                f"\n  cloth_file={missing_info.get('cloth_file')}"
                f"\n  candidate_keys={missing_info.get('candidate_keys')}"
            )
            image_base = _basename(missing_info.get("image_file", ""))
            cloth_base = _basename(missing_info.get("cloth_file", ""))
            if image_base and cloth_base:
                tail = _casefold_prompt_key(f"|||{image_base}|||{cloth_base}")
                for key in prompts_cache.keys():
                    key_text = str(key)
                    if _casefold_prompt_key(key_text).endswith(tail):
                        suffix_matches.append(key_text)
                        if len(suffix_matches) >= 10:
                            break
        if suffix_matches:
            print(
                "[PROMPT DIAG] keys ending with "
                f"'|||{_basename(missing_info.get('image_file', ''))}|||{_basename(missing_info.get('cloth_file', ''))}': "
                + ", ".join(suffix_matches)
            )
        hint_msg = (
            "\nNext steps: ensure prompts_cache contains these samples. "
            "You can regenerate missing prompts with:\n"
            "  python evaluate_extractor_DC.py --prepare_eval_assets "
            "--dresscode_root <dresscode_root> --subset_json <subset_json> "
            "--prompts_cache <prompts_cache> --prompt_provider <provider> "
            "[--prompt_model <model>]\n"
            "Or regenerate the cache using your prompt generation script."
        )
        raise RuntimeError(
            "Prompt cache miss in cache_only. "
            f"missing={missing_count} first={preview}. "
            "Please check that prompt key formats are consistent "
            "(category/image/cloth, separators, basenames, extensions, order)."
            f"{detail_msg}{hint_msg}"
        )

    return {
        "total": total,
        "cache_hit": cache_hit,
        "missing": missing_count,
        "fallback": fallback_count,
        "empty": empty_count,
        "empty_ratio": empty_ratio,
        "missing_keys": missing_keys,
    }


def check_prompts_only(
    test_data,
    prompts_cache,
    default_category,
    prompt_mode="cache_only",
    preview_count=10,
):
    if isinstance(prompts_cache, str):
        return _check_prompts_only_streaming(
            test_data=test_data,
            prompts_cache_path=prompts_cache,
            default_category=default_category,
            prompt_mode=prompt_mode,
            preview_count=preview_count,
        )
    total = 0
    cache_hit = 0
    missing = []
    missing_candidates = {}
    requested_samples = []
    cache_sample_keys = []
    cache_sample_seen = set()

    for key in prompts_cache:
        normalized_key = normalize_prompt_cache_key(key)
        if not normalized_key:
            continue
        if normalized_key in cache_sample_seen:
            continue
        cache_sample_keys.append(normalized_key)
        cache_sample_seen.add(normalized_key)
        if len(cache_sample_keys) >= 5:
            break

    for idx, item in enumerate(test_data):
        total += 1
        category = item.get("category") or default_category or "unknown"
        image_file = str(item.get("image_file", ""))
        cloth_file = str(item.get("cloth_file", ""))
        raw_key = make_pair_key(category, image_file, cloth_file)
        main_key = normalize_prompt_cache_key(raw_key)
        candidate_keys = _build_prompt_keys(category, image_file, cloth_file)
        if not candidate_keys and main_key:
            candidate_keys = [main_key]
        if main_key and main_key not in candidate_keys:
            candidate_keys.insert(0, main_key)
        if len(requested_samples) < 5 and main_key:
            requested_samples.append(main_key)

        prompt = None
        key_used = None
        for key in candidate_keys:
            if key in prompts_cache:
                prompt = prompts_cache[key]
                key_used = key
                cache_hit += 1
                break
        if prompt is None:
            missing.append(main_key)
            if main_key not in missing_candidates:
                missing_candidates[main_key] = candidate_keys[:10]

        if idx < preview_count:
            preview = _normalize_text(prompt)
            preview = " ".join(preview.split()).strip()
            if len(preview) > 120:
                preview = preview[:117] + "..."
            status = "hit" if prompt is not None else "miss"
            key_msg = key_used or main_key
            print(
                f"[CHECK PROMPT] idx={idx} key={raw_key} normalized={main_key} "
                f"status={status} used={key_msg} preview={preview}"
            )

    missing_count = len(missing)
    if requested_samples:
        print(f"[CHECK PROMPT] Requested key samples: {', '.join(requested_samples)}")
    if cache_sample_keys:
        print(f"[CHECK PROMPT] Cache key samples: {', '.join(cache_sample_keys)}")
    print(f"[CHECK PROMPT STATS] total={total} cache_hit={cache_hit} missing={missing_count}")
    if missing_count:
        preview_missing = missing[:20]
        print(f"[CHECK PROMPT] Missing key samples: {', '.join(preview_missing)}")
        for key in preview_missing:
            candidates = missing_candidates.get(key, [])
            print(f"[CHECK PROMPT] Missing key={key} candidates={candidates}")
        print(
            "[CHECK PROMPT] Missing prompts detected. Possible causes: "
            "key separators ('|' vs '|||'), missing category, swapped image/cloth, "
            "or filename extension differences."
        )
    return {"total": total, "cache_hit": cache_hit, "missing": missing_count, "missing_keys": missing}


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
        # Sanity check: cache must exist for every attn processor (keep consistent with training)
        if len(sa_hidden_states) != len(self.ref_unet.attn_processors):
            missing = []
            for _name in self.ref_unet.attn_processors.keys():
                _proc = self.ref_unet.attn_processors[_name]
                if not (hasattr(_proc, "cache") and "hidden_states" in _proc.cache):
                    missing.append(_name)
            raise RuntimeError(
                f"Attention cache missing: got {len(sa_hidden_states)}/{len(self.ref_unet.attn_processors)}. "
                f"Missing examples: {missing[:8]}"
            )


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
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
    from adapter.resampler import Resampler
    from adapter.attention_processor import CacheAttnProcessor2_0, RefCAttnProcessor2_0, RefSAttnProcessor2_0
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


@torch.inference_mode()
def generate_images(
    model,
    vae,
    text_encoder,
    image_encoder,
    tokenizer,
    test_data,
    dresscode_root,
    output_dir,
    device,
    num_inference_steps=50,
    seed=42,
    init_mode="from_noisy_gt",
    reconstruct_t=600,
    scheduler_name="dpmpp_2m_karras",
    timestep_spacing="trailing",
    eta=0.0,
    decode_fp32=True,
    guidance_scale=1.0,
    image_cfg_scale=1.0,
    min_effective_steps=10,
    vae_deterministic=False,
    refine_pass=False,
    refine_t=150,
    refine_steps=10,
    debug_timesteps=False,
    cuda_empty_cache_interval=0,
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

    os.makedirs(output_dir, exist_ok=True)
    from torchvision import transforms
    from diffusers import DDIMScheduler

    if image_cfg_scale is None:
        image_cfg_scale = 1.0
    if guidance_scale is None:
        guidance_scale = 1.0
    if image_cfg_scale == 1.0 and guidance_scale != 1.0:
        print("[WARN] guidance_scale is deprecated; using it as image_cfg_scale.")
        image_cfg_scale = float(guidance_scale)

    print(
        f"init_mode: {init_mode} | reconstruct_t: {reconstruct_t} "
        f"| image_cfg_scale: {image_cfg_scale} | guidance_scale: {guidance_scale}"
    )

    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 0.18215)

    # 准备数据转换
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

    def _select_start_index(values, target_t, min_steps_value):
        if target_t is None:
            return 0
        target = float(target_t)
        arr = np.asarray(values, dtype=np.float64)
        idx = int(np.abs(arr - target).argmin())
        if min_steps_value is not None:
            min_steps = max(1, int(min_steps_value))
            if len(values) - idx < min_steps:
                idx = max(0, len(values) - min_steps)
        return idx

    def make_scheduler_for_sample(num_steps):
        sch = _build_scheduler(scheduler_name, timestep_spacing)
        _set_scheduler_timesteps(sch, num_steps, device)
        timesteps = sch.timesteps
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
        timesteps = timesteps.to(device)
        return sch, timesteps

    def _prepare_main_timesteps(timesteps):
        timesteps_list = _timesteps_to_list(timesteps)
        if not timesteps_list:
            raise ValueError("scheduler.timesteps is empty after set_timesteps")
        if init_mode == "from_noisy_gt":
            t_start_idx = _select_start_index(timesteps_list, reconstruct_t, min_effective_steps)
            run_timesteps = timesteps[t_start_idx:]
            if run_timesteps.numel() == 0:
                t_start_idx = 0
                run_timesteps = timesteps
        else:
            t_start_idx = 0
            run_timesteps = timesteps
        return timesteps_list, t_start_idx, run_timesteps

    # Base scheduler for logging/debug; actual sampling creates a fresh scheduler per sample.
    base_scheduler, base_timesteps = make_scheduler_for_sample(num_inference_steps)
    base_timesteps_list, base_t_start_idx, base_run_timesteps = _prepare_main_timesteps(base_timesteps)
    if hasattr(base_scheduler, "set_begin_index"):
        try:
            base_scheduler.set_begin_index(base_t_start_idx)
        except Exception as exc:
            print(f"[WARN] scheduler.set_begin_index failed: {exc}")

    base_t_start = base_run_timesteps[0] if base_run_timesteps.numel() else base_timesteps[0]
    base_t_start_value = float(base_t_start.item()) if torch.is_tensor(base_t_start) else float(base_t_start)
    base_effective_steps = int(base_run_timesteps.numel())
    # README: If effective_steps is too low and guidance/image_cfg_scale is too small,
    # fine details (small text, fabric texture) can blur; increase steps or guidance to recover detail.
    print(
        f"[INFO] scheduler={scheduler_name} spacing={timestep_spacing} "
        f"t_start={base_t_start_value} idx_start={base_t_start_idx} effective_steps={base_effective_steps}"
    )
    # Debug self-check: ensure timesteps come from scheduler (no manual linspace).
    if debug_timesteps:
        head = base_timesteps_list[:5]
        tail = base_timesteps_list[-5:]
        print(f"[DEBUG] timesteps head={head} tail={tail}")

    refine_enabled = bool(refine_pass)
    if refine_enabled:
        base_refine_scheduler, base_refine_timesteps = make_scheduler_for_sample(refine_steps)
        base_refine_timesteps_list = _timesteps_to_list(base_refine_timesteps)
        if not base_refine_timesteps_list:
            print("[WARN] refine scheduler.timesteps is empty; disabling refine pass.")
            refine_enabled = False
        else:
            base_refine_start_idx = _select_start_index(base_refine_timesteps_list, refine_t, None)
            base_refine_run_timesteps = base_refine_timesteps[base_refine_start_idx:]
            if base_refine_run_timesteps.numel() == 0:
                base_refine_start_idx = 0
                base_refine_run_timesteps = base_refine_timesteps
            if hasattr(base_refine_scheduler, "set_begin_index"):
                try:
                    base_refine_scheduler.set_begin_index(base_refine_start_idx)
                except Exception as exc:
                    print(f"[WARN] refine_scheduler.set_begin_index failed: {exc}")
            base_refine_start_t = base_refine_run_timesteps[0]
            base_refine_start_value = (
                float(base_refine_start_t.item())
                if torch.is_tensor(base_refine_start_t)
                else float(base_refine_start_t)
            )
            print(
                f"[INFO] refine_steps={refine_steps} refine_t={refine_t} "
                f"t_start={base_refine_start_value} effective_steps={int(base_refine_run_timesteps.numel())}"
            )

    if image_cfg_scale < 0:
        raise ValueError("image_cfg_scale must be >= 0")

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

    generated_paths = []
    failed_indices = []
    cuda_poisoned = False
    printed_t_start_debug = False
    printed_timestep_debug = False
    printed_scheduler_debug = False
    success_count = 0  # 成功生成的图片计数器

    for idx, item in enumerate(tqdm(test_data, desc="生成中")):
        try:
            # 检查Person图片和Cloth图片是否都存在（确保一一对应）
            category = item.get("category") or "unknown"
            person_path = resolve_dresscode_person_path(dresscode_root, category, item["image_file"])
            cloth_path = resolve_dresscode_cloth_path(dresscode_root, category, item["cloth_file"])

            if not os.path.exists(person_path):
                print(f"\n警告: Person图片不存在: {person_path}")
                failed_indices.append(idx)
                continue

            if not os.path.exists(cloth_path):
                print(f"\n警告: Cloth图片不存在: {cloth_path}")
                failed_indices.append(idx)
                continue

            scheduler, timesteps = make_scheduler_for_sample(num_inference_steps)
            timesteps_list, t_start_idx, run_timesteps = _prepare_main_timesteps(timesteps)
            if hasattr(scheduler, "set_begin_index"):
                try:
                    scheduler.set_begin_index(t_start_idx)
                except Exception as exc:
                    print(f"[WARN] scheduler.set_begin_index failed: {exc}")
            if hasattr(scheduler, "_step_index"):
                if scheduler._step_index is not None:
                    print(f"[WARN] scheduler._step_index={scheduler._step_index} at sample start; resetting.")
                scheduler._step_index = None
            if (idx == 0 or debug_timesteps) and not printed_scheduler_debug:
                info = f"[DEBUG] scheduler={scheduler.__class__.__name__} timesteps={len(timesteps_list)}"
                if hasattr(scheduler, "sigmas") and torch.is_tensor(scheduler.sigmas):
                    info += f" sigmas={len(scheduler.sigmas)}"
                if hasattr(scheduler, "_step_index"):
                    info += f" step_index={scheduler._step_index}"
                print(info)
                printed_scheduler_debug = True

            step_kwargs = {}
            if isinstance(scheduler, DDIMScheduler) and eta:
                step_kwargs["eta"] = eta

            refine_scheduler = None
            refine_timesteps = None
            refine_start_t = None
            refine_step_kwargs = {}
            if refine_enabled:
                refine_scheduler, refine_timesteps = make_scheduler_for_sample(refine_steps)
                refine_timesteps_list = _timesteps_to_list(refine_timesteps)
                if not refine_timesteps_list:
                    print("[WARN] refine scheduler.timesteps is empty; disabling refine pass.")
                    refine_scheduler = None
                    refine_timesteps = None
                else:
                    refine_start_idx = _select_start_index(refine_timesteps_list, refine_t, None)
                    refine_timesteps_full = refine_timesteps
                    refine_timesteps = refine_timesteps_full[refine_start_idx:]
                    if refine_timesteps.numel() == 0:
                        refine_start_idx = 0
                        refine_timesteps = refine_timesteps_full
                    if hasattr(refine_scheduler, "set_begin_index"):
                        try:
                            refine_scheduler.set_begin_index(refine_start_idx)
                        except Exception as exc:
                            print(f"[WARN] refine_scheduler.set_begin_index failed: {exc}")
                    if hasattr(refine_scheduler, "_step_index"):
                        if refine_scheduler._step_index is not None:
                            print(
                                f"[WARN] refine_scheduler._step_index={refine_scheduler._step_index} "
                                "at sample start; resetting."
                            )
                        refine_scheduler._step_index = None
                    refine_start_t = refine_timesteps[0]
                    if isinstance(refine_scheduler, DDIMScheduler) and eta:
                        refine_step_kwargs["eta"] = eta

            t_start = run_timesteps[0] if run_timesteps.numel() else timesteps[0]

            person_img = Image.open(person_path).convert("RGB")

            # 准备输入
            person_vae = vae_transform(person_img).unsqueeze(0).to(device)
            person_clip = clip_transform(person_img).unsqueeze(0).to(device)

            # 编码Person图片
            posterior = vae.encode(person_vae).latent_dist
            person_latents = posterior.mode() if vae_deterministic else posterior.sample()
            person_latents = person_latents * scaling_factor

            person_image_embeds = image_encoder(
                person_clip,
                output_hidden_states=True
            ).hidden_states[-2]
            zero_image_embeds = torch.zeros_like(person_image_embeds)

            # 准备文本（DressCode 直接使用 cache/fallback 的文本）
            text = item.get("text", "")
            if isinstance(text, list) and len(text) > 0:
                text = text[0]
            if text is None:
                text = ""
            prompt_text = text

            text_inputs = tokenizer(
                prompt_text,
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
                cloth_latents = cloth_posterior.mode() if vae_deterministic else cloth_posterior.sample()
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
                t_start_tensor = as_timestep_tensor(
                    t_start,
                    device=cloth_latents.device,
                    batch_size=cloth_latents.shape[0],
                )
                if idx == 0 and not printed_t_start_debug:
                    print(
                        "[DEBUG] t_start_tensor shape="
                        f"{tuple(t_start_tensor.shape)} dtype={t_start_tensor.dtype} device={t_start_tensor.device}"
                    )
                    printed_t_start_debug = True
                latents = scheduler.add_noise(cloth_latents, noise, t_start_tensor)
            else:
                latents = torch.randn(
                    person_latents.shape,
                    generator=g,
                    device=device,
                    dtype=person_latents.dtype,
                )
                latents = latents * scheduler.init_noise_sigma

            for t in run_timesteps:
                timestep_tensor = as_timestep_tensor(
                    t,
                    device=latents.device,
                    batch_size=latents.shape[0],
                )
                if idx == 0 and not printed_timestep_debug:
                    print(
                        "[DEBUG] timestep_tensor shape="
                        f"{tuple(timestep_tensor.shape)} dtype={timestep_tensor.dtype} device={timestep_tensor.device}"
                    )
                    printed_timestep_debug = True
                if hasattr(scheduler, "scale_model_input"):
                    latents_input = scheduler.scale_model_input(latents, timestep_tensor)
                else:
                    latents_input = latents
                noise_pred = model(
                    text_embeddings,
                    latents_input,
                    person_latents,
                    person_image_embeds,
                    timestep_tensor
                )

                if image_cfg_scale != 1.0:
                    noise_pred_uncond = model(
                        text_embeddings,
                        latents_input,
                        person_latents,
                        zero_image_embeds,
                        timestep_tensor
                    )
                    noise_pred = noise_pred_uncond + image_cfg_scale * (noise_pred - noise_pred_uncond)

                if step_kwargs:
                    latents = scheduler.step(noise_pred, timestep_tensor, latents, **step_kwargs).prev_sample
                else:
                    latents = scheduler.step(noise_pred, timestep_tensor, latents).prev_sample

            if refine_scheduler is not None and refine_timesteps:
                g_refine = torch.Generator(device=latents.device)
                g_refine.manual_seed(seed + idx + 10000)
                refine_noise = torch.randn(
                    latents.shape,
                    generator=g_refine,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                refine_t_tensor = as_timestep_tensor(
                    refine_start_t,
                    device=latents.device,
                    batch_size=latents.shape[0],
                )
                latents = refine_scheduler.add_noise(latents, refine_noise, refine_t_tensor)

                for t in refine_timesteps:
                    timestep_tensor = as_timestep_tensor(
                        t,
                        device=latents.device,
                        batch_size=latents.shape[0],
                    )
                    if hasattr(refine_scheduler, "scale_model_input"):
                        latents_input = refine_scheduler.scale_model_input(latents, timestep_tensor)
                    else:
                        latents_input = latents

                    noise_pred = model(
                        text_embeddings,
                        latents_input,
                        person_latents,
                        person_image_embeds,
                        timestep_tensor
                    )

                    if image_cfg_scale != 1.0:
                        noise_pred_uncond = model(
                            text_embeddings,
                            latents_input,
                            person_latents,
                            zero_image_embeds,
                            timestep_tensor
                        )
                        noise_pred = noise_pred_uncond + image_cfg_scale * (noise_pred - noise_pred_uncond)

                    if refine_step_kwargs:
                        latents = refine_scheduler.step(
                            noise_pred,
                            timestep_tensor,
                            latents,
                            **refine_step_kwargs,
                        ).prev_sample
                    else:
                        latents = refine_scheduler.step(
                            noise_pred,
                            timestep_tensor,
                            latents,
                        ).prev_sample

            image = _decode_latents(latents)

            # 归一化到[0,1]
            image = (image.float() + 1.0) / 2.0
            image = torch.clamp(image, 0.0, 1.0)

            # 保存生成的图片（使用连续的success_count命名）
            image_np = image[0].cpu().permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)

            # 使用连续计数命名，确保与ground truth对应
            save_path = os.path.join(output_dir, f"{success_count:06d}_generated.png")
            image_pil.save(save_path)
            generated_paths.append(save_path)

            # Free large tensors between samples to reduce VRAM pressure.
            del person_img, person_vae, person_clip, posterior, person_latents
            del person_image_embeds, zero_image_embeds, text_inputs, text_embeddings
            if init_mode == "from_noisy_gt":
                del cloth_img, cloth_vae, cloth_posterior, cloth_latents
            del latents, image, image_np, image_pil
            if cuda_empty_cache_interval and (idx + 1) % cuda_empty_cache_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            success_count += 1  # 只有成功时才递增

        except Exception as e:
            if _is_cuda_failure(e):
                global CUDA_POISONED
                CUDA_POISONED = True
                cuda_poisoned = True
                failed_indices.append(idx)
                print(f"[ERROR] idx={idx} err={e}")
                print("[FATAL] CUDA poisoned; stop generation early and fallback metrics to CPU.")
                if idx + 1 < len(test_data):
                    failed_indices.extend(range(idx + 1, len(test_data)))
                safe_cuda_cleanup("after_gen_error")
                break
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


def _resize_center_crop(img, size):
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        resize_mode = resampling.BICUBIC
    else:
        resize_mode = Image.BICUBIC

    width, height = img.size
    if width <= 0 or height <= 0:
        return img.resize((size, size), resample=resize_mode)

    if width < height:
        new_width = size
        new_height = int(round(height * (size / width)))
    else:
        new_height = size
        new_width = int(round(width * (size / height)))
    img = img.resize((new_width, new_height), resample=resize_mode)

    left = int((new_width - size) / 2)
    top = int((new_height - size) / 2)
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))


def prepare_ground_truth(test_data, dresscode_root, output_dir, failed_indices=None):
    """
    准备Ground Truth图片（复制真实的cloth图片）

    Args:
        test_data: 测试数据
        dresscode_root: DressCode 根目录
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

    for idx, item in enumerate(tqdm(test_data, desc="复制中")):
        # 跳过生成失败的样本
        if idx in failed_indices:
            continue

        try:
            # 获取Ground Truth（真实的garment图片）
            category = item.get("category") or "unknown"
            cloth_path = resolve_dresscode_cloth_path(dresscode_root, category, item["cloth_file"])

            if not os.path.exists(cloth_path):
                print(f"\n警告: Ground Truth图片不存在: {cloth_path}")
                missing_count += 1
                continue

            # 使用与生成图片相同的连续计数命名
            dest_path = os.path.join(output_dir, f"{success_count:06d}_groundtruth.png")

            # 打开并保存为PNG（统一格式）
            img = Image.open(cloth_path).convert("RGB")
            img = _resize_center_crop(img, 512)
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


def _collect_common_paths(real_dir, generated_dir):
    import glob
    real_paths = sorted(glob.glob(os.path.join(real_dir, "*_groundtruth.png")))
    gen_paths = sorted(glob.glob(os.path.join(generated_dir, "*_generated.png")))

    if len(real_paths) == 0 or len(gen_paths) == 0:
        print("[ERROR] No images found for metrics.")
        return [], [], [], set(), set()

    real_map = {}
    for path in real_paths:
        name = os.path.basename(path)
        if name.endswith("_groundtruth.png"):
            key = name[:-len("_groundtruth.png")]
            real_map[key] = path
    gen_map = {}
    for path in gen_paths:
        name = os.path.basename(path)
        if name.endswith("_generated.png"):
            key = name[:-len("_generated.png")]
            gen_map[key] = path

    common_keys = sorted(set(real_map.keys()) & set(gen_map.keys()))
    missing_real = set(gen_map.keys()) - set(real_map.keys())
    missing_gen = set(real_map.keys()) - set(gen_map.keys())
    if missing_real:
        print(f"[WARN] Missing groundtruth for {len(missing_real)} generated images")
    if missing_gen:
        print(f"[WARN] Missing generated images for {len(missing_gen)} groundtruths")
    if not common_keys:
        print("[ERROR] No matched image pairs for metrics.")
        return [], [], [], missing_real, missing_gen

    real_common = [real_map[key] for key in common_keys]
    gen_common = [gen_map[key] for key in common_keys]
    return real_common, gen_common, common_keys, missing_real, missing_gen


def _compute_pixel_metrics(real_paths, gen_paths, keys, device, label=None):
    label_text = f" - {label}" if label else ""
    print(f"\n{'='*60}")
    print(f"Compute pixel metrics (PSNR, SSIM, LPIPS, DISTS){label_text}")
    print(f"{'='*60}")

    if len(real_paths) == 0 or len(gen_paths) == 0:
        print("[ERROR] No images found for pixel metrics.")
        return None

    import lpips
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    min_count = min(len(real_paths), len(gen_paths))
    if len(real_paths) != len(gen_paths):
        print("[WARN] Mismatched counts; truncating to min_count.")
        real_paths = real_paths[:min_count]
        gen_paths = gen_paths[:min_count]
        if keys:
            keys = keys[:min_count]

    print("Init LPIPS (alex)...")
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    dists_model = None
    dists_error = None
    try:
        try:
            from piq import DISTS as DISTSModel
        except Exception:
            from dists_pytorch import DISTS as DISTSModel
        dists_model = DISTSModel().to(device)
        dists_model.eval()
    except Exception as exc:
        dists_model = None
        dists_error = str(exc)
        print("[WARN] DISTS not available, skip. Install: pip install piq  (or)  pip install dists-pytorch")

    psnr_values = []
    ssim_values = []
    lpips_values = []
    dists_values = []
    resize_count = 0
    warned_resize = False
    skipped_count = 0

    print(f"Compute metrics for {min_count} pairs...")
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        resize_mode = resampling.BICUBIC
    else:
        resize_mode = Image.BICUBIC

    if keys is None:
        keys = [os.path.basename(p) for p in real_paths]

    for idx, (real_path, gen_path) in enumerate(tqdm(list(zip(real_paths, gen_paths))[:min_count], total=min_count, desc="metrics")):
        key = keys[idx] if idx < len(keys) else str(idx)
        try:
            real_img = np.array(Image.open(real_path).convert("RGB"))
            gen_img = np.array(Image.open(gen_path).convert("RGB"))

            if real_img.shape[:2] != gen_img.shape[:2]:
                if not warned_resize:
                    print("[WARN] Resizing generated images to match groundtruth for pixel metrics")
                    warned_resize = True
                resize_count += 1
                gen_img = np.array(
                    Image.fromarray(gen_img).resize(
                        (real_img.shape[1], real_img.shape[0]),
                        resample=resize_mode,
                    )
                )

            psnr_val = psnr(real_img, gen_img, data_range=255)
            ssim_val = ssim(real_img, gen_img, channel_axis=-1, data_range=255)

            real_tensor = torch.from_numpy(real_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            gen_tensor = torch.from_numpy(gen_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            real_tensor = real_tensor.to(device)
            gen_tensor = gen_tensor.to(device)

            with torch.no_grad():
                lpips_val = lpips_model(real_tensor, gen_tensor).item()

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            lpips_values.append(lpips_val)
        except Exception as exc:
            skipped_count += 1
            print(f"\n[WARN] Skipping metric pair for {real_path}: {exc}")
            continue

        if dists_model is not None:
            try:
                real_tensor_01 = torch.from_numpy(real_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                gen_tensor_01 = torch.from_numpy(gen_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                real_tensor_01 = real_tensor_01.to(device)
                gen_tensor_01 = gen_tensor_01.to(device)
                with torch.no_grad():
                    dists_val = dists_model(real_tensor_01, gen_tensor_01).item()
                dists_values.append(dists_val)
            except Exception as exc:
                if dists_error is None:
                    dists_error = str(exc)
                print(f"[WARN] DISTS failed for {key}: {exc}")
                dists_model = None

    psnr_mean = np.mean(psnr_values) if psnr_values else None
    psnr_std = np.std(psnr_values) if psnr_values else None
    ssim_mean = np.mean(ssim_values) if ssim_values else None
    ssim_std = np.std(ssim_values) if ssim_values else None
    lpips_mean = np.mean(lpips_values) if lpips_values else None
    lpips_std = np.std(lpips_values) if lpips_values else None
    dists_mean = np.mean(dists_values) if dists_values else None
    dists_std = np.std(dists_values) if dists_values else None

    print("\nPixel metrics summary")
    print(f"  PSNR: {psnr_mean:.2f} +/- {psnr_std:.2f} dB" if psnr_mean is not None else "  PSNR: N/A")
    print(f"  SSIM: {ssim_mean:.4f} +/- {ssim_std:.4f}" if ssim_mean is not None else "  SSIM: N/A")
    print(f"  LPIPS: {lpips_mean:.4f} +/- {lpips_std:.4f}" if lpips_mean is not None else "  LPIPS: N/A")
    print(f"  DISTS: {dists_mean:.4f} +/- {dists_std:.4f}" if dists_mean is not None else "  DISTS: N/A")
    if skipped_count > 0:
        print(f"  Skipped pairs: {skipped_count}")
    print(f"[INFO] resized_pairs={resize_count}/{min_count}")

    del lpips_model
    if dists_model is not None:
        del dists_model
    safe_cuda_cleanup("after_pixel_metrics")

    return {
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "dists_mean": dists_mean,
        "dists_std": dists_std,
        "dists_error": dists_error,
        "resize_count": resize_count,
        "num_pairs": min_count,
        "skipped_count": skipped_count,
    }


def calculate_pixel_metrics(real_dir, generated_dir, device):
    real_paths, gen_paths, common_keys, _, _ = _collect_common_paths(real_dir, generated_dir)
    if not real_paths or not gen_paths:
        return None
    return _compute_pixel_metrics(real_paths, gen_paths, common_keys, device)


def calculate_pixel_metrics_from_paths(real_paths, gen_paths, device, label=None):
    return _compute_pixel_metrics(real_paths, gen_paths, None, device, label=label)

def evaluate_metrics_from_paths(real_paths, gen_paths, device, label=None):
    """Compute FID/KID metrics from path lists."""
    label_text = f" - {label}" if label else ""
    print(f"\n{'='*60}")
    print(f"Compute distribution metrics (FID & KID){label_text}")
    print(f"{'='*60}")

    print(f"Ground Truth images: {len(real_paths)}")
    print(f"Generated images: {len(gen_paths)}")

    min_count = min(len(real_paths), len(gen_paths))
    if min_count < 2:
        print("[WARN] Not enough images to compute FID/KID (need >=2 per set).")
        return None, None, None, "not_enough_images"

    if len(real_paths) != len(gen_paths):
        print("[WARN] Mismatched counts; truncating to min_count.")
        real_paths = real_paths[:min_count]
        gen_paths = gen_paths[:min_count]

    try:
        from calculate_fid_kid_standard import (
            InceptionV3FeatureExtractor,
            calculate_fid_standard,
            calculate_kid_standard,
        )
        print("Init InceptionV3 feature extractor...")
        extractor = InceptionV3FeatureExtractor(device=device)

        print("Extracting features (real)...")
        features_real = extractor.extract_features(real_paths, batch_size=32)

        print("Extracting features (generated)...")
        features_gen = extractor.extract_features(gen_paths, batch_size=32)

        print("Computing FID...")
        fid_value = calculate_fid_standard(features_real, features_gen)

        print("Computing KID...")
        subset_size = min(1000, min_count)
        kid_mean, kid_std = calculate_kid_standard(
            features_real,
            features_gen,
            subset_size=subset_size,
            num_subsets=100,
        )

        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Samples: {min_count}")
        print(f"FID: {fid_value:.4f}")
        if kid_mean is not None:
            print(f"KID: {kid_mean:.6f} +/- {kid_std:.6f}")
            print(f"KID* (x1000): {kid_mean * 1000:.3f} +/- {kid_std * 1000:.3f}")
        print(f"{'='*60}")

        return fid_value, kid_mean, kid_std, None
    except Exception as exc:
        print(f"[WARN] FID/KID calculation failed: {exc}")
        return None, None, None, str(exc)

def evaluate_metrics(real_dir, generated_dir, device):
    """Compute FID/KID metrics for common pairs."""
    real_paths, gen_paths, _, _, _ = _collect_common_paths(real_dir, generated_dir)
    if not real_paths or not gen_paths:
        return None, None, None, "no_images"
    return evaluate_metrics_from_paths(real_paths, gen_paths, device)

def _normalize_subset_items(items, default_category):
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        category = item.get("category") or default_category or "unknown"
        image_file = str(item.get("image_file", ""))
        cloth_file = str(item.get("cloth_file", ""))
        text = item.get("text", "")
        if isinstance(text, list):
            text = text[0] if text else ""
        if text is None:
            text = ""
        normalized.append(
            {
                "category": category,
                "image_file": image_file,
                "cloth_file": cloth_file,
                "text": text,
            }
        )
    return normalized

def main():
    parser = argparse.ArgumentParser(description="Evaluate DressCode Extractor model")
    parser.add_argument("--dresscode_root", type=str, required=True, help="DressCode root directory")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--pretrained_model", type=str, default="models/IMAGDressing", help="Pretrained model path")
    parser.add_argument("--image_encoder", type=str, default="models/IMAGDressing/image_encoder", help="Image encoder path")
    parser.add_argument("--vae_path", type=str, default="models/IMAGDressing/sd-vae-ft-mse", help="VAE path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation step")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpmpp_2m_karras",
        choices=["ddim", "dpmpp_2m", "dpmpp_2m_karras", "unipc", "euler_a"],
        help="Sampling scheduler",
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="trailing",
        choices=["leading", "trailing"],
        help="Scheduler timestep spacing",
    )
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (ignored by other schedulers)")
    parser.add_argument(
        "--decode_fp32",
        dest="decode_fp32",
        action="store_true",
        help="Decode VAE in fp32 for sharper outputs",
    )
    parser.add_argument(
        "--no-decode_fp32",
        dest="decode_fp32",
        action="store_false",
        help="Decode VAE in model dtype",
    )
    parser.set_defaults(decode_fp32=True)
    parser.set_defaults(strict_empty_prompt=True)

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset_json", type=str, default="", help="Subset JSON path")
    parser.add_argument("--subset_size", type=int, default=500, help="Subset size when building subset JSON")
    parser.add_argument("--subset_seed", type=int, default=42, help="Subset sampling seed")
    parser.add_argument("--build_subset_only", action="store_true", help="Build subset JSON and exit")
    parser.add_argument("--subset_overwrite", action="store_true", help="Overwrite subset JSON if it exists")
    parser.add_argument("--build_prompts_only", action="store_true", help="Build prompts cache and exit")
    parser.add_argument(
        "--prepare_eval_assets",
        action="store_true",
        help="Build subset_json and prompts_cache (if missing) and exit",
    )

    parser.add_argument("--prompts_cache", type=str, default="", help="Prompt cache path (json/jsonl)")
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default=None,
        choices=["cache_only", "cache_or_fallback", "fallback_only", "empty_only"],
        help="Prompt mode (default: checkpoint prompt_mode or cache_only)",
    )
    parser.add_argument(
        "--prompt_fallback",
        type=str,
        nargs="*",
        default=None,
        help="Fallback prompt when cache misses",
    )
    parser.add_argument("--log_first_n_prompts", type=int, default=10, help="Log first N prompts")
    parser.add_argument(
        "--check_prompts_only",
        action="store_true",
        help="Load subset_json + prompts_cache, validate prompt hits, and exit",
    )
    parser.add_argument(
        "--strict_empty_prompt",
        dest="strict_empty_prompt",
        action="store_true",
        help="Error on empty prompt or cache miss",
    )
    parser.add_argument(
        "--no_strict_empty_prompt",
        dest="strict_empty_prompt",
        action="store_false",
        help="Allow empty prompts and cache misses",
    )
    parser.add_argument(
        "--strict_prompt_cache",
        action="store_true",
        help="Error when cache_only misses any prompt (print missing keys and exit)",
    )
    parser.add_argument(
        "--auto_fallback_on_prompt_miss",
        dest="auto_fallback_on_prompt_miss",
        action="store_true",
        help="Auto-generate missing prompts or fall back when cache_only misses (default: enabled)",
    )
    parser.add_argument(
        "--no_auto_fallback_on_prompt_miss",
        dest="auto_fallback_on_prompt_miss",
        action="store_false",
        help="Disable auto handling when cache_only misses",
    )
    parser.set_defaults(auto_fallback_on_prompt_miss=True)
    parser.add_argument(
        "--prompt_provider",
        type=str,
        default="openai",
        choices=["dashscope_openai_compat", "openai"],
        help="Prompt API provider (OpenAI-compatible).",
    )
    parser.add_argument("--prompt_model", type=str, default="gpt-4o-mini", help="Prompt model name")
    parser.add_argument(
        "--prompt_api_base",
        type=str,
        default="https://api.openai.com/v1",
        help="Prompt API base URL (OpenAI-compatible).",
    )
    parser.add_argument(
        "--prompt_api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for prompt API key",
    )
    parser.add_argument(
        "--prompt_api_key_file",
        type=str,
        default="",
        help="Optional file containing prompt API key (takes precedence over env var).",
    )
    parser.add_argument(
        "--prompt_image_mode",
        type=str,
        default="cloth_only",
        choices=["cloth_only", "person_plus_cloth"],
        help="Prompt image mode",
    )
    parser.add_argument("--prompt_max_tokens", type=int, default=128, help="Max tokens for prompt generation")
    parser.add_argument("--prompt_temperature", type=float, default=0.2, help="Prompt generation temperature")
    parser.add_argument("--prompt_concurrency", type=int, default=8, help="Prompt generation concurrency")
    parser.add_argument("--prompt_overwrite", action="store_true", help="Overwrite existing prompts in cache")

    parser.add_argument(
        "--init_mode",
        type=str,
        default="from_noisy_gt",
        choices=["from_noise", "from_noisy_gt"],
        help="Sampling init mode",
    )
    parser.add_argument("--reconstruct_t", type=int, default=600, help="Noisy GT timestep")
    parser.add_argument(
        "--min_effective_steps",
        type=int,
        default=10,
        help="Minimum effective denoising steps when using from_noisy_gt",
    )
    parser.add_argument(
        "--image_cfg_scale",
        type=float,
        default=1.0,
        help="Image CFG scale (uses zeroed person_image_embeds for uncond branch)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Deprecated text CFG scale (kept for compatibility)",
    )
    parser.add_argument("--vae_deterministic", action="store_true", help="Deterministic VAE encode/decode")
    parser.add_argument("--refine_pass", action="store_true", help="Enable refine pass for sharper outputs")
    parser.add_argument("--refine_t", type=int, default=150, help="Refine pass start timestep")
    parser.add_argument("--refine_steps", type=int, default=10, help="Refine pass steps")
    parser.add_argument("--debug_timesteps", action="store_true", help="Log scheduler timesteps for debugging")
    parser.add_argument(
        "--cuda_empty_cache_interval",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache() every N samples (0 disables)",
    )

    parser.add_argument("--dresscode_category", type=str, default="all", help="DressCode category")
    parser.add_argument("--dresscode_test_order", type=str, default="paired", help="DressCode test order")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples when subset_json not provided")
    parser.add_argument("--metrics_sanity_check", action="store_true", help="Run metric sanity check (GT vs GT)")

    args = parser.parse_args()

    if "--prompt_provider" not in sys.argv:
        env_provider = os.environ.get("DRESSCODE_PROMPT_PROVIDER", "").strip()
        if env_provider:
            args.prompt_provider = env_provider
        else:
            args.prompt_provider = None
    if "--prompt_model" not in sys.argv:
        env_model = os.environ.get("DRESSCODE_PROMPT_MODEL", "").strip()
        if env_model:
            args.prompt_model = env_model
    if "--prompt_api_base" not in sys.argv:
        env_api_base = os.environ.get("DRESSCODE_PROMPT_API_BASE", "").strip()
        if env_api_base:
            args.prompt_api_base = env_api_base
    if "--prompt_api_key_env" not in sys.argv:
        env_api_key_env = os.environ.get("DRESSCODE_PROMPT_API_KEY_ENV", "").strip()
        if env_api_key_env:
            args.prompt_api_key_env = env_api_key_env
    if "--prompt_api_key_file" not in sys.argv:
        env_api_key_file = os.environ.get("DRESSCODE_PROMPT_API_KEY_FILE", "").strip()
        if env_api_key_file:
            args.prompt_api_key_file = env_api_key_file
    if "--prompt_image_mode" not in sys.argv:
        env_image_mode = os.environ.get("DRESSCODE_PROMPT_IMAGE_MODE", "").strip()
        if env_image_mode:
            args.prompt_image_mode = env_image_mode
    if "--prompt_max_tokens" not in sys.argv:
        env_max_tokens = os.environ.get("DRESSCODE_PROMPT_MAX_TOKENS", "").strip()
        if env_max_tokens.isdigit():
            args.prompt_max_tokens = int(env_max_tokens)
    if "--prompt_temperature" not in sys.argv:
        env_temp = os.environ.get("DRESSCODE_PROMPT_TEMPERATURE", "").strip()
        if env_temp:
            try:
                args.prompt_temperature = float(env_temp)
            except ValueError:
                pass
    if "--prompt_concurrency" not in sys.argv:
        env_concurrency = os.environ.get("DRESSCODE_PROMPT_CONCURRENCY", "").strip()
        if env_concurrency.isdigit():
            args.prompt_concurrency = int(env_concurrency)

    if args.build_subset_only and (args.build_prompts_only or args.prepare_eval_assets):
        parser.error("--build_subset_only cannot be combined with other asset-only flags")
    if args.build_prompts_only and args.prepare_eval_assets:
        parser.error("--build_prompts_only cannot be combined with --prepare_eval_assets")

    if args.build_subset_only and not args.subset_json:
        parser.error("--build_subset_only requires --subset_json")
    if args.build_prompts_only:
        if not args.subset_json:
            parser.error("--build_prompts_only requires --subset_json")
        if not args.prompts_cache:
            parser.error("--build_prompts_only requires --prompts_cache")
    if args.prepare_eval_assets:
        if not args.subset_json:
            parser.error("--prepare_eval_assets requires --subset_json")
        if not args.prompts_cache:
            parser.error("--prepare_eval_assets requires --prompts_cache")
    if args.check_prompts_only:
        if not args.subset_json:
            parser.error("--check_prompts_only requires --subset_json")
        if not args.prompts_cache:
            parser.error("--check_prompts_only requires --prompts_cache")

    asset_only = (
        args.build_subset_only
        or args.build_prompts_only
        or args.prepare_eval_assets
        or args.check_prompts_only
    )
    if not asset_only:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless building assets only")
        if not args.output_dir:
            parser.error("--output_dir is required unless building assets only")

    prompt_cache_source = "cli"
    prompt_mode_source = "cli"
    if not asset_only:
        cache_path, prompt_cache_source, cache_found = _resolve_prompts_cache_path(args, {})
        if cache_found:
            args.prompts_cache = cache_path
            print(f"[PromptCache] using: {args.prompts_cache}")
        else:
            expected = cache_path or r"E:\BaiduNetdiskDownload\DressCode\prompts_dresscode_cache.jsonl"
            parser.error(
                f"prompts_cache not found: {expected}\n"
                "Pass --prompts_cache or ensure the cache file exists."
            )
        if args.prompt_mode is None:
            args.prompt_mode = "cache_only"
            prompt_mode_source = "default"
        valid_modes = {"cache_only", "cache_or_fallback", "fallback_only", "empty_only"}
        if args.prompt_mode not in valid_modes:
            print(f"[WARN] Unknown prompt_mode '{args.prompt_mode}', defaulting to cache_only.")
            args.prompt_mode = "cache_only"
            prompt_mode_source = "default"
    if args.prompt_mode is None:
        args.prompt_mode = "cache_only"
        prompt_mode_source = "default"

    seed = getattr(args, "seed", 42)
    subset_seed = getattr(args, "subset_seed", seed)
    if args.prompt_fallback is not None:
        prompt_fallback = " ".join(args.prompt_fallback).strip()
    else:
        prompt_fallback = "" if args.prompt_mode == "cache_only" else "a photo of a garment"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def ensure_prompt_provider():
        if args.prompt_provider:
            return True
        print(
            "[ERROR] prompt_provider not configured. Set DRESSCODE_PROMPT_PROVIDER or pass --prompt_provider."
        )
        return False

    def load_dataset_items():
        from transformers import CLIPTokenizer
        tokenizer_for_dc = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
        from Dresscode import DressCodeDataset

        ds = DressCodeDataset(
            root=args.dresscode_root,
            split=args.split,
            tokenizer=tokenizer_for_dc,
            category=args.dresscode_category,
            test_order=args.dresscode_test_order,
            prompts_cache=None,
            prompt_mode="fallback_only",
            prompt_fallback="",
            size=512,
            verify_files=False,
        )
        return _normalize_subset_items(ds.data, args.dresscode_category)

    subset_path = args.subset_json or ""
    test_data = []

    def load_subset_json(path):
        with open(path, "r", encoding="utf-8") as f:
            subset_data = json.load(f)
        if isinstance(subset_data, dict) and "data" in subset_data:
            subset_data = subset_data["data"]
        return _normalize_subset_items(subset_data, args.dresscode_category)

    def build_subset_json(path):
        all_items = load_dataset_items()
        target_size = len(all_items)
        if args.max_samples and args.max_samples > 0:
            target_size = min(target_size, args.max_samples)
        elif path and args.subset_size and args.subset_size > 0:
            target_size = min(target_size, args.subset_size)
        if target_size < len(all_items):
            rng = random.Random(subset_seed)
            subset_items = rng.sample(all_items, target_size)
        else:
            subset_items = list(all_items)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset_items, f, indent=2)
        print(f"[OK] Saved subset_json: {path} ({len(subset_items)} items)")
        return subset_items

    def find_missing_prompt_items(items):
        existing_keys = set()
        if not items:
            return [], existing_keys
        if not args.prompts_cache or not os.path.exists(args.prompts_cache):
            return list(items), existing_keys

        candidate_to_indices, main_keys, _, _ = _build_prompt_candidate_index(
            items, args.dresscode_category
        )
        hits = [False] * len(main_keys)
        remaining = len(main_keys)

        def register_hit(idx):
            if hits[idx]:
                return
            hits[idx] = True
            if main_keys[idx]:
                existing_keys.add(main_keys[idx])

        ext = os.path.splitext(args.prompts_cache)[1].lower()
        if ext == ".jsonl":
            with open(args.prompts_cache, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for raw_key in _extract_entry_candidate_keys(obj):
                        for alias in canonicalize_cache_key(raw_key):
                            indices = candidate_to_indices.get(alias)
                            if not indices:
                                continue
                            for idx in indices:
                                if not hits[idx]:
                                    register_hit(idx)
                                    remaining -= 1
                            if remaining == 0:
                                break
                        if remaining == 0:
                            break
                    if remaining == 0:
                        break
        elif ext == ".json":
            with open(args.prompts_cache, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = []
            if isinstance(data, dict):
                entries = [{"key": k, "prompt": v} for k, v in data.items()]
            elif isinstance(data, list):
                entries = data
            for obj in entries:
                for raw_key in _extract_entry_candidate_keys(obj):
                    for alias in canonicalize_cache_key(raw_key):
                        indices = candidate_to_indices.get(alias)
                        if not indices:
                            continue
                        for idx in indices:
                            if not hits[idx]:
                                register_hit(idx)
                                remaining -= 1
                        if remaining == 0:
                            break
                    if remaining == 0:
                        break
                if remaining == 0:
                    break

        missing = [item for idx, item in enumerate(items) if not hits[idx]]
        return missing, existing_keys

    if args.build_subset_only:
        if os.path.exists(subset_path) and not args.subset_overwrite:
            test_data = load_subset_json(subset_path)
            print(f"[INFO] Loaded subset_json: {subset_path} ({len(test_data)} items)")
        else:
            test_data = build_subset_json(subset_path)
        print("[OK] build_subset_only requested; exiting.")
        return

    if args.prepare_eval_assets:
        if os.path.exists(subset_path) and not args.subset_overwrite:
            test_data = load_subset_json(subset_path)
            print(f"[INFO] Loaded subset_json: {subset_path} ({len(test_data)} items)")
        else:
            test_data = build_subset_json(subset_path)
        if not test_data:
            print("[ERROR] No test data found for prompt generation.")
            return
        missing_items, existing_keys = find_missing_prompt_items(test_data)
        if not os.path.exists(args.prompts_cache) or missing_items or args.prompt_overwrite:
            if not ensure_prompt_provider():
                return
            items_for_generation = (
                test_data if args.prompt_overwrite or not missing_items else missing_items
            )
            print(
                f"[INFO] Building prompts_cache (missing={len(missing_items)} existing={len(existing_keys)})"
            )
            try:
                build_prompts_cache(
                    dresscode_root=args.dresscode_root,
                    items=items_for_generation,
                    prompts_cache_path=args.prompts_cache,
                    default_category=args.dresscode_category,
                    prompt_provider=args.prompt_provider,
                    prompt_model=args.prompt_model,
                    prompt_api_base=args.prompt_api_base,
                    prompt_api_key_env=args.prompt_api_key_env,
                    prompt_api_key_file=args.prompt_api_key_file,
                    prompt_image_mode=args.prompt_image_mode,
                    prompt_max_tokens=args.prompt_max_tokens,
                    prompt_temperature=args.prompt_temperature,
                    prompt_concurrency=args.prompt_concurrency,
                    prompt_overwrite=args.prompt_overwrite,
                )
            except Exception as exc:
                print(f"[ERROR] Prompt generation failed: {exc}")
                return
        else:
            print("[OK] prompts_cache already complete; skipping prompt generation.")
        print("[OK] prepare_eval_assets requested; exiting.")
        return

    if args.build_prompts_only:
        if not os.path.exists(subset_path):
            print(f"[ERROR] subset_json not found: {subset_path}")
            return
        test_data = load_subset_json(subset_path)
        if not test_data:
            print("[ERROR] No test data found for prompt generation.")
            return
        if not ensure_prompt_provider():
            return
        missing_items, existing_keys = find_missing_prompt_items(test_data)
        items_for_generation = (
            test_data if args.prompt_overwrite or not missing_items else missing_items
        )
        try:
            build_prompts_cache(
                dresscode_root=args.dresscode_root,
                items=items_for_generation,
                prompts_cache_path=args.prompts_cache,
                default_category=args.dresscode_category,
                prompt_provider=args.prompt_provider,
                prompt_model=args.prompt_model,
                prompt_api_base=args.prompt_api_base,
                prompt_api_key_env=args.prompt_api_key_env,
                prompt_api_key_file=args.prompt_api_key_file,
                prompt_image_mode=args.prompt_image_mode,
                prompt_max_tokens=args.prompt_max_tokens,
                prompt_temperature=args.prompt_temperature,
                prompt_concurrency=args.prompt_concurrency,
                prompt_overwrite=args.prompt_overwrite,
            )
        except Exception as exc:
            print(f"[ERROR] Prompt generation failed: {exc}")
            return
        print("[OK] build_prompts_only requested; exiting.")
        return

    if subset_path:
        if os.path.exists(subset_path) and not args.subset_overwrite:
            test_data = load_subset_json(subset_path)
            print(f"[INFO] Loaded subset_json: {subset_path} ({len(test_data)} items)")
        else:
            test_data = build_subset_json(subset_path)
    else:
        all_items = load_dataset_items()
        if args.max_samples and args.max_samples > 0 and args.max_samples < len(all_items):
            rng = random.Random(seed)
            test_data = rng.sample(all_items, args.max_samples)
        else:
            test_data = list(all_items)

    if not test_data:
        print("[ERROR] No test data found.")
        return

    if args.check_prompts_only:
        try:
            stats = check_prompts_only(
                test_data=test_data,
                prompts_cache=args.prompts_cache,
                default_category=args.dresscode_category,
                prompt_mode=args.prompt_mode,
                preview_count=min(10, max(0, int(args.log_first_n_prompts))),
            )
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(2)
        if args.prompt_mode == "cache_only" and stats.get("missing", 0) > 0:
            sys.exit(2)
        print("[OK] check_prompts_only requested; exiting.")
        return

    if args.prompt_mode == "cache_only":
        missing_items, existing_keys = find_missing_prompt_items(test_data)
        missing_count = len(missing_items)
        if missing_count > 0:
            print(
                f"[PROMPT CACHE] cache_only missing={missing_count} existing={len(existing_keys)} "
                "before evaluation."
            )
            if args.auto_fallback_on_prompt_miss:
                if ensure_prompt_provider():
                    print("[PROMPT CACHE] Auto-generating missing prompts for subset_json...")
                    try:
                        build_prompts_cache(
                            dresscode_root=args.dresscode_root,
                            items=missing_items,
                            prompts_cache_path=args.prompts_cache,
                            default_category=args.dresscode_category,
                            prompt_provider=args.prompt_provider,
                            prompt_model=args.prompt_model,
                            prompt_api_base=args.prompt_api_base,
                            prompt_api_key_env=args.prompt_api_key_env,
                            prompt_api_key_file=args.prompt_api_key_file,
                            prompt_image_mode=args.prompt_image_mode,
                            prompt_max_tokens=args.prompt_max_tokens,
                            prompt_temperature=args.prompt_temperature,
                            prompt_concurrency=args.prompt_concurrency,
                            prompt_overwrite=False,
                        )
                    except Exception as exc:
                        print(f"[WARN] Auto-generate prompts failed: {exc}")
                        if not prompt_fallback:
                            prompt_fallback = "a photo of the garment only"
                        args.prompt_mode = "cache_or_fallback"
                        print(
                            "[PROMPT CACHE] Falling back to cache_or_fallback due to generation failure."
                        )
                    else:
                        missing_items, _ = find_missing_prompt_items(test_data)
                        if missing_items:
                            print(
                                f"[ERROR] prompts_cache still missing {len(missing_items)} items after generation."
                            )
                            raise RuntimeError(
                                "Prompt cache incomplete after auto-generation. "
                                "Please regenerate prompts_cache for this subset."
                            )
                        print(
                            f"[PROMPT CACHE] Auto-generation complete. Filled {missing_count} prompts."
                        )
                else:
                    if not prompt_fallback:
                        prompt_fallback = "a photo of the garment only"
                    args.prompt_mode = "cache_or_fallback"
                    print(
                        "[PROMPT CACHE] prompt_provider not configured; "
                        "downgrading to cache_or_fallback with fallback prompt. "
                        "Metrics may be less comparable."
                    )
            else:
                raise RuntimeError(
                    "Prompt cache miss in cache_only. "
                    "Use --prepare_eval_assets / --build_prompts_only to fill prompts_cache."
                )

    if not args.output_dir:
        args.output_dir = "."
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(
        f"[INFO] scheduler={args.scheduler} spacing={args.timestep_spacing} eta={args.eta} "
        f"decode_fp32={args.decode_fp32}"
    )
    print(
        f"[INFO] init_mode={args.init_mode} reconstruct_t={args.reconstruct_t} "
        f"steps={args.num_inference_steps} min_effective_steps={args.min_effective_steps} "
        f"image_cfg_scale={args.image_cfg_scale} refine_pass={args.refine_pass}"
    )
    if not asset_only:
        cache_info = args.prompts_cache if args.prompts_cache else "(not found)"
        print(f"[INFO] prompts_cache ({prompt_cache_source}): {cache_info}")
        print(f"[INFO] prompt_mode ({prompt_mode_source}): {args.prompt_mode}")
        print(f"[INFO] strict_prompt_cache: {bool(args.strict_prompt_cache)}")

    prompt_cache_map = load_prompts_cache(args.prompts_cache)
    if args.prompts_cache:
        print(f"[INFO] prompts_cache entries: {len(prompt_cache_map)}")

    prompt_stats = prepare_prompts_for_items(
        test_data=test_data,
        prompts_cache=prompt_cache_map,
        prompt_mode=args.prompt_mode,
        prompt_fallback=prompt_fallback,
        log_first_n_prompts=args.log_first_n_prompts,
        strict_empty_prompt=args.strict_empty_prompt,
        strict_prompt_cache=args.strict_prompt_cache,
        default_category=args.dresscode_category,
    )
    print(f"[OK] Test set size: {len(test_data)}")

    generated_dir = os.path.join(args.output_dir, "generated")
    groundtruth_dir = os.path.join(args.output_dir, "groundtruth")

    failed_indices = []
    cuda_poisoned = False

    if not args.skip_generation:
        model, vae, text_encoder, image_encoder, tokenizer = load_model(
            args.checkpoint,
            args.pretrained_model,
            args.image_encoder,
            args.vae_path,
            device,
        )

        generated_paths, failed_indices, cuda_poisoned = generate_images(
            model,
            vae,
            text_encoder,
            image_encoder,
            tokenizer,
            test_data,
            dresscode_root=args.dresscode_root,
            output_dir=generated_dir,
            device=device,
            num_inference_steps=args.num_inference_steps,
            seed=seed,
            init_mode=args.init_mode,
            reconstruct_t=args.reconstruct_t,
            scheduler_name=args.scheduler,
            timestep_spacing=args.timestep_spacing,
            eta=args.eta,
            decode_fp32=bool(args.decode_fp32),
            guidance_scale=args.guidance_scale,
            image_cfg_scale=args.image_cfg_scale,
            min_effective_steps=args.min_effective_steps,
            vae_deterministic=bool(args.vae_deterministic),
            refine_pass=bool(args.refine_pass),
            refine_t=args.refine_t,
            refine_steps=args.refine_steps,
            debug_timesteps=bool(args.debug_timesteps),
            cuda_empty_cache_interval=args.cuda_empty_cache_interval,
        )

        del model, vae, text_encoder, image_encoder, tokenizer
        global CUDA_POISONED
        CUDA_POISONED = cuda_poisoned
        safe_cuda_cleanup("after_generation")
    else:
        print(f"\n[INFO] Skip generation; using existing images in {generated_dir}")

    CUDA_POISONED = cuda_poisoned

    metrics_device = torch.device("cpu") if cuda_poisoned else device
    if cuda_poisoned:
        print("[WARN] CUDA poisoned; running metrics on CPU.")

    prepare_ground_truth(
        test_data,
        dresscode_root=args.dresscode_root,
        output_dir=groundtruth_dir,
        failed_indices=failed_indices,
    )

    pixel_metrics = calculate_pixel_metrics(groundtruth_dir, generated_dir, metrics_device)

    fid = None
    kid_mean = None
    kid_std = None
    fid_error = None
    fid, kid_mean, kid_std, fid_error = evaluate_metrics(groundtruth_dir, generated_dir, metrics_device)

    import glob

    sanity_check = None
    if args.metrics_sanity_check:
        sanity_real_paths = sorted(glob.glob(os.path.join(groundtruth_dir, "*_groundtruth.png")))
        if not sanity_real_paths:
            sanity_check = {"error": "no_groundtruth_images"}
        else:
            sanity_pixel = calculate_pixel_metrics_from_paths(
                sanity_real_paths,
                sanity_real_paths,
                metrics_device,
                label="sanity_check",
            )
            sanity_fid, sanity_kid, sanity_kid_std, sanity_fid_err = evaluate_metrics_from_paths(
                sanity_real_paths,
                sanity_real_paths,
                metrics_device,
                label="sanity_check",
            )
            sanity_check = {
                "count": len(sanity_real_paths),
                "psnr_mean": float(sanity_pixel["psnr_mean"]) if sanity_pixel and sanity_pixel["psnr_mean"] is not None else None,
                "ssim_mean": float(sanity_pixel["ssim_mean"]) if sanity_pixel and sanity_pixel["ssim_mean"] is not None else None,
                "lpips_mean": float(sanity_pixel["lpips_mean"]) if sanity_pixel and sanity_pixel["lpips_mean"] is not None else None,
                "dists_mean": float(sanity_pixel["dists_mean"]) if sanity_pixel and sanity_pixel["dists_mean"] is not None else None,
                "fid": float(sanity_fid) if sanity_fid is not None else None,
                "kid": float(sanity_kid) if sanity_kid is not None else None,
                "kid_std": float(sanity_kid_std) if sanity_kid_std is not None else None,
            }
            if sanity_pixel and sanity_pixel.get("dists_error"):
                sanity_check["dists_error"] = sanity_pixel.get("dists_error")
            if sanity_fid_err:
                sanity_check["fid_kid_error"] = sanity_fid_err

    checkpoint_name = Path(args.checkpoint).stem
    resize_count = pixel_metrics.get("resize_count", 0) if pixel_metrics else 0
    num_pairs = pixel_metrics.get("num_pairs", 0) if pixel_metrics else 0
    skipped_count = pixel_metrics.get("skipped_count", 0) if pixel_metrics else 0
    count_generated = len(glob.glob(os.path.join(generated_dir, "*_generated.png")))
    count_groundtruth = len(glob.glob(os.path.join(groundtruth_dir, "*_groundtruth.png")))

    results = {
        "dataset": "DressCode",
        "dresscode_root": args.dresscode_root,
        "dresscode_category": args.dresscode_category,
        "dresscode_test_order": args.dresscode_test_order,
        "subset_json": subset_path,
        "subset_size": len(test_data),
        "subset_requested": (args.max_samples if args.max_samples > 0 else args.subset_size) if subset_path else None,
        "subset_seed": subset_seed if subset_path else None,
        "max_samples": args.max_samples if not subset_path else None,
        "seed": seed,
        "num_inference_steps": args.num_inference_steps,
        "scheduler": args.scheduler,
        "timestep_spacing": args.timestep_spacing,
        "eta": args.eta,
        "decode_fp32": bool(args.decode_fp32),
        "init_mode": args.init_mode,
        "reconstruct_t": args.reconstruct_t,
        "min_effective_steps": args.min_effective_steps,
        "guidance_scale": args.guidance_scale,
        "image_cfg_scale": args.image_cfg_scale,
        "vae_deterministic": bool(args.vae_deterministic),
        "refine_pass": bool(args.refine_pass),
        "refine_t": args.refine_t,
        "refine_steps": args.refine_steps,
        "metrics_sanity_check": bool(args.metrics_sanity_check),
        "prompt_mode": args.prompt_mode,
        "prompt_fallback_text": prompt_fallback,
        "strict_prompt_cache": bool(args.strict_prompt_cache),
        "prompts_cache": args.prompts_cache,
        "prompt_provider": args.prompt_provider,
        "prompt_model": args.prompt_model,
        "prompt_api_base": args.prompt_api_base,
        "prompt_api_key_env": args.prompt_api_key_env,
        "prompt_image_mode": args.prompt_image_mode,
        "prompt_max_tokens": args.prompt_max_tokens,
        "prompt_temperature": args.prompt_temperature,
        "prompt_concurrency": args.prompt_concurrency,
        "prompt_overwrite": bool(args.prompt_overwrite),
        "prompt_cache_hit": prompt_stats.get("cache_hit", 0),
        "prompt_fallback": prompt_stats.get("fallback", 0),
        "prompt_empty": prompt_stats.get("empty", 0),
        "prompt_empty_ratio": prompt_stats.get("empty_ratio", 0.0),
        "checkpoint": args.checkpoint,
        "checkpoint_name": checkpoint_name,
        "num_samples": len(test_data),
        "num_successful": len(test_data) - len(failed_indices),
        "num_failed": len(failed_indices),
        "resize_count": resize_count,
        "count_generated": count_generated,
        "count_groundtruth": count_groundtruth,
        "count_used": num_pairs,
        "num_pairs": num_pairs,
        "skipped_count": skipped_count,
        "cuda_poisoned": bool(cuda_poisoned),
        "PSNR": float(pixel_metrics["psnr_mean"]) if pixel_metrics and pixel_metrics["psnr_mean"] is not None else None,
        "PSNR_std": float(pixel_metrics["psnr_std"]) if pixel_metrics and pixel_metrics["psnr_std"] is not None else None,
        "SSIM": float(pixel_metrics["ssim_mean"]) if pixel_metrics and pixel_metrics["ssim_mean"] is not None else None,
        "SSIM_std": float(pixel_metrics["ssim_std"]) if pixel_metrics and pixel_metrics["ssim_std"] is not None else None,
        "LPIPS": float(pixel_metrics["lpips_mean"]) if pixel_metrics and pixel_metrics["lpips_mean"] is not None else None,
        "LPIPS_std": float(pixel_metrics["lpips_std"]) if pixel_metrics and pixel_metrics["lpips_std"] is not None else None,
        "DISTS": float(pixel_metrics["dists_mean"]) if pixel_metrics and pixel_metrics["dists_mean"] is not None else None,
        "DISTS_std": float(pixel_metrics["dists_std"]) if pixel_metrics and pixel_metrics["dists_std"] is not None else None,
        "FID": float(fid) if fid is not None else None,
        "KID": float(kid_mean) if kid_mean is not None else None,
        "KID_mean": float(kid_mean) if kid_mean is not None else None,
        "KID_std": float(kid_std) if kid_std is not None else None,
        "KID_x1000": float(kid_mean * 1000) if kid_mean is not None else None,
        "KID_x1000_std": float(kid_std * 1000) if kid_std is not None else None,
        "psnr_mean": float(pixel_metrics["psnr_mean"]) if pixel_metrics and pixel_metrics["psnr_mean"] is not None else None,
        "ssim_mean": float(pixel_metrics["ssim_mean"]) if pixel_metrics and pixel_metrics["ssim_mean"] is not None else None,
        "lpips_mean": float(pixel_metrics["lpips_mean"]) if pixel_metrics and pixel_metrics["lpips_mean"] is not None else None,
        "dists_mean": float(pixel_metrics["dists_mean"]) if pixel_metrics and pixel_metrics["dists_mean"] is not None else None,
        "fid": float(fid) if fid is not None else None,
        "kid": float(kid_mean) if kid_mean is not None else None,
        "kid_std": float(kid_std) if kid_std is not None else None,
    }

    if pixel_metrics and pixel_metrics.get("dists_error"):
        results["dists_error"] = pixel_metrics.get("dists_error")
    if fid_error:
        results["fid_kid_error"] = fid_error
    if sanity_check is not None:
        results["sanity_check"] = sanity_check

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    legacy_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_results = {
        "psnr_mean": float(pixel_metrics["psnr_mean"]) if pixel_metrics and pixel_metrics["psnr_mean"] is not None else None,
        "ssim_mean": float(pixel_metrics["ssim_mean"]) if pixel_metrics and pixel_metrics["ssim_mean"] is not None else None,
        "lpips_mean": float(pixel_metrics["lpips_mean"]) if pixel_metrics and pixel_metrics["lpips_mean"] is not None else None,
        "dists_mean": float(pixel_metrics["dists_mean"]) if pixel_metrics and pixel_metrics["dists_mean"] is not None else None,
        "fid": float(fid) if fid is not None else None,
        "kid": float(kid_mean) if kid_mean is not None else None,
        "kid_std": float(kid_std) if kid_std is not None else None,
        "prompt_cache_hit": prompt_stats.get("cache_hit", 0),
        "prompt_fallback": prompt_stats.get("fallback", 0),
        "prompt_empty": prompt_stats.get("empty", 0),
        "resize_count": resize_count,
        "num_pairs": num_pairs,
        "skipped_count": skipped_count,
    }
    if fid_error:
        summary_results["fid_kid_error"] = fid_error
    if pixel_metrics and pixel_metrics.get("dists_error"):
        summary_results["dists_error"] = pixel_metrics.get("dists_error")

    summary_path = os.path.join(args.output_dir, "results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n[OK] Results saved to: {metrics_path}")
    print(f"[OK] Legacy results saved to: {legacy_path}")
    print(f"[OK] Summary saved to: {summary_path}")
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
