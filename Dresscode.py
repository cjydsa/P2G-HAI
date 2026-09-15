import json
import os
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from transformers import CLIPImageProcessor

# Allow loading truncated/damaged images
ImageFile.LOAD_TRUNCATED_IMAGES = True

DRESSCODE_CATEGORIES = ["upper_body", "lower_body", "dresses"]
VALID_SPLITS = ["train", "test"]
VALID_TEST_ORDERS = ["paired", "unpaired"]
VALID_PROMPT_MODES = ["cache_only", "cache_or_fallback", "fallback_only"]


def make_pair_key(category, image_file, cloth_file):
    """
    Generate a consistent cache key for a (person, cloth) pair.

    Args:
        category: DressCode category
        image_file: Person image filename
        cloth_file: Cloth image filename

    Returns:
        str: Cache key in format "category|||image_file|||cloth_file"
    """
    return f"{category}|||{image_file}|||{cloth_file}"


def normalize_prompt_cache_key(key):
    """
    Normalize legacy cache keys to the canonical format using basenames.

    Legacy keys may include absolute/relative paths for image/cloth files.
    """
    if not isinstance(key, str):
        return key
    parts = key.split("|||")
    if len(parts) != 3:
        return key
    category = parts[0].strip()
    image_file = Path(parts[1].replace("\\", "/")).name
    cloth_file = Path(parts[2].replace("\\", "/")).name
    return make_pair_key(category, image_file, cloth_file)


class DressCodeDataset(Dataset):
    """
    DressCode Dataset loader for training garment extraction model.

    Args:
        root: DressCode root directory
        split: Dataset split ("train" or "test")
        tokenizer: CLIP tokenizer
        category: "all" or one of ["upper_body", "lower_body", "dresses"]
        test_order: "paired" or "unpaired" (only used for split="test")
        prompts_cache: Path to prompt cache file (json or jsonl), or None/empty string
        prompt_mode: Cache usage mode:
            - "cache_only": STRICT mode - only use cache, raise error on miss (for training)
            - "cache_or_fallback": Try cache first, use fallback on miss (for testing)
            - "fallback_only": Always use fallback (for prompt generation preprocessing)
        prompt_fallback: Default prompt when cache is missing (must be non-empty for fallback modes)
        size: Image size for VAE (default 512)
        verify_files: If True, verify all person/cloth files exist before training (default: True)
        strict_prompts_cache: If True, raise on cache miss even in cache_or_fallback mode
        enforce_cache_coverage: If True, raise when prompt_mode=cache_only and cache coverage < 100%

    Raises:
        FileNotFoundError: If required directories/files not found
        RuntimeError: If verify_files=True and files are missing
        RuntimeError: If prompt_mode=cache_only and cache is incomplete (when enforce_cache_coverage=True)
    """
    def __init__(
            self,
            root,
            split,
            tokenizer,
            category="all",
            test_order="paired",
            prompts_cache=None,
            prompt_mode="cache_or_fallback",
            prompt_fallback="",
            size=512,
            verify_files=True,
            strict_prompts_cache=False,
            enforce_cache_coverage=True,
    ):
        self.root = root
        self.split = split
        self.tokenizer = tokenizer
        self.category = category
        self.test_order = test_order
        self.size = size
        self.prompt_mode = prompt_mode
        self.prompt_fallback = prompt_fallback
        self.strict_prompts_cache = strict_prompts_cache
        self.enforce_cache_coverage = enforce_cache_coverage

        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}. Must be one of {VALID_SPLITS}")
        if category not in ["all"] + DRESSCODE_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of ['all'] + {DRESSCODE_CATEGORIES}")
        if test_order not in VALID_TEST_ORDERS:
            raise ValueError(f"Invalid test_order: {test_order}. Must be one of {VALID_TEST_ORDERS}")
        if prompt_mode not in VALID_PROMPT_MODES:
            raise ValueError(f"Invalid prompt_mode: {prompt_mode}. Must be one of {VALID_PROMPT_MODES}")
        if prompt_mode in ("cache_or_fallback", "fallback_only"):
            if prompt_fallback is None or str(prompt_fallback).strip() == "":
                raise RuntimeError(
                    "[PromptFallbackEmpty] prompt_fallback is empty or whitespace. "
                    "Provide a non-empty fallback prompt via --prompt_fallback \"...\", "
                    "or use --prompt_mode cache_only and ensure 100% prompt cache coverage."
                )
        if not os.path.isdir(root):
            raise FileNotFoundError(f"DressCode root not found: {root}")

        self.categories = DRESSCODE_CATEGORIES if category == "all" else [category]

        self.images_dir_map = {}
        for cat in self.categories:
            images_dir = os.path.join(self.root, cat, "images")
            if not os.path.isdir(images_dir):
                raise FileNotFoundError(f"Missing images directory: {images_dir}")
            self.images_dir_map[cat] = images_dir

        print(f"[DressCodeDataset] Loading DressCode from: {root}")
        print(f"  Split: {split}")
        print(f"  Categories: {', '.join(self.categories)}")
        print(f"  Prompt mode: {prompt_mode}")

        # Load pairs across categories
        self.data = []
        pairs_filename = self._pairs_filename()
        missing_categories = []

        for cat in self.categories:
            pairs_path = os.path.join(self.root, cat, pairs_filename)
            if os.path.exists(pairs_path):
                cat_pairs = self._load_pairs(
                    pairs_path,
                    expected_category=cat,
                    allow_category_prefix=True,
                    restrict_categories=None,
                    require_category_prefix=False,
                )
                self.data.extend(cat_pairs)
            else:
                missing_categories.append(cat)

        if missing_categories:
            root_pairs_path = os.path.join(self.root, pairs_filename)
            if not os.path.exists(root_pairs_path):
                missing_str = ", ".join(missing_categories)
                raise FileNotFoundError(
                    "Missing pairs for categories: "
                    f"{missing_str}. Expected {os.path.join(self.root, '<category>', pairs_filename)} "
                    f"or fallback {root_pairs_path}."
                )

            fallback_pairs = self._load_pairs(
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
                    f"{missing_str}. Provide per-category pairs under {self.root}."
                )

            loaded_fallback = {item["category"] for item in fallback_pairs}
            still_missing = [cat for cat in missing_categories if cat not in loaded_fallback]
            if still_missing:
                missing_str = ", ".join(still_missing)
                raise RuntimeError(
                    "Root-level pairs file missing category prefixes for required categories: "
                    f"{missing_str}. Provide per-category pairs under {self.root}."
                )

            self.data.extend(fallback_pairs)

        if not self.data:
            raise RuntimeError("No pairs loaded from DressCode dataset.")

        if verify_files:
            self._verify_files()

        # Load prompt cache
        self.prompt_cache = {}
        self._legacy_cache_keys_mapped = 0
        if prompts_cache and os.path.exists(prompts_cache):
            self._load_prompt_cache(prompts_cache)
            print(f"  Prompt cache loaded: {len(self.prompt_cache)} entries")

            if prompt_mode != "fallback_only":
                coverage = self._check_cache_coverage()
                print(f"  Cache coverage: {coverage:.1f}%")

                if prompt_mode == "cache_only" and coverage < 100.0 and self.enforce_cache_coverage:
                    missing_keys = self._get_missing_cache_keys()
                    covered = len(self.data) - len(missing_keys)
                    error_msg = (
                        "[PromptCacheIncomplete] prompt_mode=cache_only requires 100% cache coverage, "
                        f"but only {coverage:.1f}% ({covered}/{len(self.data)}) is cached.\n\n"
                        f"Missing prompts for {len(missing_keys)} pairs.\n"
                    )

                    show_count = min(20, len(missing_keys))
                    error_msg += f"\nFirst {show_count} missing cache keys:\n"
                    for i in range(show_count):
                        error_msg += f"  - {missing_keys[i]}\n"
                    if len(missing_keys) > show_count:
                        error_msg += f"  ... and {len(missing_keys) - show_count} more\n"

                    error_msg += (
                        "\nSolution: Run Stage-1 to generate prompts:\n"
                        "  python train_extractor_DC.py ... --auto_prepare_prompts --prepare_prompts_only\n"
                    )

                    raise RuntimeError(error_msg)

        else:
            if prompt_mode == "cache_only":
                raise FileNotFoundError(
                    "[PromptCacheMissing] prompt_mode=cache_only requires a valid cache file, "
                    f"but prompts_cache='{prompts_cache}' does not exist.\n\n"
                    "Solution: Run Stage-1 to generate prompts:\n"
                    "  python train_extractor_DC.py ... --auto_prepare_prompts --prepare_prompts_only"
                )
            if prompt_mode == "cache_or_fallback":
                print("  No cache file found, will use fallback prompt")
            print(f"  Fallback prompt: '{prompt_fallback}'")

        print(f"[DressCodeDataset] Loaded {len(self.data)} pairs")

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()

        # Stats tracking for cache misses
        self.cache_miss_count = 0
        self.cache_hit_count = 0

    def _pairs_filename(self):
        if self.split == "train":
            return "train_pairs.txt"
        return f"test_pairs_{self.test_order}.txt"

    def _ensure_image_ext(self, name):
        base, ext = os.path.splitext(name)
        if ext:
            return name
        return f"{name}.jpg"

    def _normalize_pair_path(self, path):
        normalized = path.replace("\\", "/").strip()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.lstrip("/")

    def _split_category_prefix(self, path):
        normalized = self._normalize_pair_path(path)
        parts = [p for p in normalized.split("/") if p]
        if not parts:
            return None, ""
        if parts[0] in DRESSCODE_CATEGORIES:
            category = parts[0]
            rest = parts[1:]
            if rest and rest[0] == "images":
                rest = rest[1:]
            if not rest:
                return category, ""
            return category, "/".join(rest)
        return None, normalized

    def _strip_images_prefix(self, path):
        normalized = self._normalize_pair_path(path)
        parts = [p for p in normalized.split("/") if p]
        if parts and parts[0] in ("images", "image"):
            parts = parts[1:]
        return "/".join(parts)

    def _load_pairs(
            self,
            pairs_path,
            expected_category=None,
            allow_category_prefix=False,
            restrict_categories=None,
            require_category_prefix=False,
    ):
        pairs = []
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    print(f"WARNING: Invalid line {line_num} in {pairs_path}: {line}")
                    continue

                person_token = parts[0]
                cloth_token = parts[1]

                if allow_category_prefix:
                    person_cat, person_path = self._split_category_prefix(person_token)
                    cloth_cat, cloth_path = self._split_category_prefix(cloth_token)
                else:
                    person_cat, person_path = None, self._normalize_pair_path(person_token)
                    cloth_cat, cloth_path = None, self._normalize_pair_path(cloth_token)

                if person_cat and cloth_cat and person_cat != cloth_cat:
                    raise RuntimeError(
                        f"Category mismatch in {pairs_path} line {line_num}: "
                        f"{person_cat} vs {cloth_cat}"
                    )

                line_category = person_cat or cloth_cat
                if line_category is None:
                    if require_category_prefix:
                        raise RuntimeError(
                            f"Root-level pairs file lacks category prefix at line {line_num} in {pairs_path}. "
                            "Provide per-category pairs files under each category directory."
                        )
                    line_category = expected_category

                if line_category is None:
                    raise RuntimeError(
                        f"Unable to determine category for line {line_num} in {pairs_path}."
                    )

                if expected_category and line_category != expected_category:
                    raise RuntimeError(
                        f"Unexpected category '{line_category}' in {pairs_path} line {line_num}; "
                        f"expected '{expected_category}'."
                    )

                if restrict_categories and line_category not in restrict_categories:
                    continue

                person_file = self._strip_images_prefix(person_path)
                cloth_file = self._strip_images_prefix(cloth_path)
                if not person_file or not cloth_file:
                    print(f"WARNING: Invalid paths in {pairs_path} line {line_num}: {line}")
                    continue

                person_file = self._ensure_image_ext(person_file)
                cloth_file = self._ensure_image_ext(cloth_file)
                person_file = Path(person_file).name
                cloth_file = Path(cloth_file).name

                pairs.append({
                    "image_file": person_file,
                    "cloth_file": cloth_file,
                    "category": line_category,
                })

        return pairs

    def _verify_files(self):
        missing = []
        for item in self.data:
            category = item["category"]
            images_dir = self.images_dir_map[category]
            person_path = os.path.join(images_dir, item["image_file"])
            cloth_path = os.path.join(images_dir, item["cloth_file"])
            if not os.path.exists(person_path):
                missing.append(("person", category, item["image_file"], person_path))
            if not os.path.exists(cloth_path):
                missing.append(("cloth", category, item["cloth_file"], cloth_path))

        if missing:
            error_msg = f"[FileMissing] Missing {len(missing)} files referenced in pairs.\n"
            error_msg += f"  Root: {self.root}\n"
            show_count = min(20, len(missing))
            error_msg += f"First {show_count} missing files:\n"
            for i in range(show_count):
                kind, category, filename, path = missing[i]
                error_msg += f"  - {kind} | {category} | {filename} | {path}\n"
            if len(missing) > show_count:
                error_msg += f"  ... and {len(missing) - show_count} more\n"
            raise RuntimeError(error_msg)

        print(f"  Verified {len(self.data)} pairs (person and cloth files exist)")

    def _load_prompt_cache(self, cache_path):
        """
        Load prompt cache from json or jsonl file.

        Formats supported:
        - JSON: {"key": "prompt", ...} or {"key": ["prompt1", "prompt2"], ...}
        - JSON: {"key": {"text": "...", ...}, ...}
        - JSONL: {"key": "...", "text": "..."} (one entry per line)
                 {"key": "...", "prompt": "..."}
                 {"key": "...", "prompts": ["..."]}

        Keys must be in format: "category|||image_file|||cloth_file"
        """
        ext = os.path.splitext(cache_path)[1].lower()
        legacy_mapped = 0

        if ext == ".json":
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
                for key, value in cache.items():
                    prompt_value = None
                    if isinstance(value, dict):
                        if "text" in value:
                            prompt_value = value["text"]
                        elif "prompt" in value:
                            prompt_value = value["prompt"]
                        elif "prompts" in value:
                            prompt_value = value["prompts"]
                    else:
                        prompt_value = value

                    if isinstance(prompt_value, str):
                        normalized_key = normalize_prompt_cache_key(key)
                        self.prompt_cache[key] = [prompt_value]
                        if normalized_key != key:
                            legacy_mapped += 1
                            if normalized_key not in self.prompt_cache:
                                self.prompt_cache[normalized_key] = [prompt_value]
                    elif isinstance(prompt_value, list):
                        normalized_key = normalize_prompt_cache_key(key)
                        self.prompt_cache[key] = prompt_value
                        if normalized_key != key:
                            legacy_mapped += 1
                            if normalized_key not in self.prompt_cache:
                                self.prompt_cache[normalized_key] = prompt_value
                    else:
                        print(f"WARNING: Invalid cache value for key {key}: {value}")

        elif ext == ".jsonl":
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = obj.get("key")
                    if not key:
                        continue

                    prompt_value = None
                    if "text" in obj:
                        prompt_value = obj["text"]
                    elif "prompt" in obj:
                        prompt_value = obj["prompt"]
                    elif "prompts" in obj:
                        prompt_value = obj["prompts"]

                    if isinstance(prompt_value, str):
                        normalized_key = normalize_prompt_cache_key(key)
                        self.prompt_cache[key] = [prompt_value]
                        if normalized_key != key:
                            legacy_mapped += 1
                            if normalized_key not in self.prompt_cache:
                                self.prompt_cache[normalized_key] = [prompt_value]
                    elif isinstance(prompt_value, list):
                        normalized_key = normalize_prompt_cache_key(key)
                        self.prompt_cache[key] = prompt_value
                        if normalized_key != key:
                            legacy_mapped += 1
                            if normalized_key not in self.prompt_cache:
                                self.prompt_cache[normalized_key] = prompt_value
        else:
            raise ValueError(f"Unsupported cache format: {ext} (expected .json or .jsonl)")

        if legacy_mapped > 0 and self._legacy_cache_keys_mapped == 0:
            self._legacy_cache_keys_mapped = legacy_mapped
            print(f"[PromptCache] Detected legacy keys with paths; mapped {legacy_mapped} entries to normalized keys.")

    def _check_cache_coverage(self):
        if not self.data:
            return 0.0

        covered = 0
        for item in self.data:
            key = make_pair_key(item["category"], item["image_file"], item["cloth_file"])
            if key in self.prompt_cache:
                covered += 1

        return (covered / len(self.data)) * 100.0

    def _get_missing_cache_keys(self):
        missing = []
        for item in self.data:
            key = make_pair_key(item["category"], item["image_file"], item["cloth_file"])
            if key not in self.prompt_cache:
                missing.append(key)
        return missing

    def _get_prompt_for_item(self, item):
        key = make_pair_key(item["category"], item["image_file"], item["cloth_file"])

        if self.prompt_mode == "fallback_only":
            return self.prompt_fallback
        elif self.prompt_mode == "cache_only":
            if key in self.prompt_cache:
                self.cache_hit_count += 1
                prompts = self.prompt_cache[key]
                return random.choice(prompts) if isinstance(prompts, list) else prompts

            raise RuntimeError(
                "[CacheMiss] Cache miss during training in cache_only mode.\n"
                f"  Key: {key}\n"
                f"  Category: {item['category']}\n"
                f"  Person: {item['image_file']}\n"
                f"  Cloth: {item['cloth_file']}\n\n"
                "This indicates the prompt cache is incomplete or corrupted.\n"
                "Please re-run Stage-1 to regenerate prompts:\n"
                "  python train_extractor_DC.py ... --auto_prepare_prompts --prepare_prompts_only"
            )
        elif self.prompt_mode == "cache_or_fallback":
            if key in self.prompt_cache:
                self.cache_hit_count += 1
                prompts = self.prompt_cache[key]
                return random.choice(prompts) if isinstance(prompts, list) else prompts
            if self.strict_prompts_cache:
                raise RuntimeError(
                    "[CacheMissStrict] strict_prompts_cache enabled but prompt is missing.\n"
                    f"  Key: {key}\n"
                    f"  Category: {item['category']}\n"
                    f"  Person: {item['image_file']}\n"
                    f"  Cloth: {item['cloth_file']}\n\n"
                    "Disable --strict_prompts_cache or regenerate cache with --auto_prepare_prompts."
                )
            self.cache_miss_count += 1
            return self.prompt_fallback
        else:
            raise ValueError(f"Invalid prompt_mode: {self.prompt_mode}")

    def _safe_open_rgb(self, path, max_side=2048):
        """
        Safely open and convert image to RGB with memory-efficient preprocessing.
        Copied from IGPair.py to prevent OOM on large images.
        """
        try:
            img = Image.open(path)

            if hasattr(img, "size"):
                w, h = img.size
                if w > 8192 or h > 8192:
                    img = img.resize((min(w, max_side), min(h, max_side)), Image.BILINEAR)

            try:
                img.draft("RGB", (max_side, max_side))
            except Exception:
                pass

            img.thumbnail((max_side, max_side), Image.BILINEAR)

            if img.mode != "RGB":
                img = img.convert("RGB")

            return img

        except MemoryError:
            with Image.open(path) as raw:
                small_img = raw.resize((self.size, self.size), Image.BILINEAR)
                return small_img.convert("RGB")
        except Exception as e:
            print(f"WARNING: Failed to load image {path}: {e}")
            return Image.new("RGB", (self.size, self.size), (0, 0, 0))

    def __getitem__(self, idx):
        item = self.data[idx]
        category = item["category"]
        images_dir = self.images_dir_map[category]

        person_path = os.path.join(images_dir, item["image_file"])
        cloth_path = os.path.join(images_dir, item["cloth_file"])

        person_img = self._safe_open_rgb(person_path, max_side=2048)
        clothes_img = self._safe_open_rgb(cloth_path, max_side=2048)

        text = self._get_prompt_for_item(item)
        text_with_prefix = text

        drop_image_embed = False
        rand_num = random.random()
        if rand_num < 0.05:
            drop_image_embed = True
        elif rand_num < 0.1:
            text_with_prefix = ""
        elif rand_num < 0.15:
            text_with_prefix = ""
            drop_image_embed = True

        text_input_ids = self.tokenizer(
            text_with_prefix,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids = self.tokenizer(
            "",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        vae_person = self.transform(person_img)
        vae_clothes = self.transform(clothes_img)

        # Use person image for CLIP
        clip_image = self.clip_image_processor(images=person_img, return_tensors="pt").pixel_values

        return {
            "vae_person": vae_person,
            "vae_clothes": vae_clothes,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "text": text_with_prefix,
            "text_input_ids": text_input_ids,
            "null_text_input_ids": null_text_input_ids,
            "image_file": item["image_file"],
            "cloth_file": item["cloth_file"],
            "category": category,
        }

    def __len__(self):
        return len(self.data)



def collate_fn(data):
    """
    Collate function for batching (identical to IGPair.py).
    """
    vae_person = torch.stack([example["vae_person"] for example in data]).to(
        memory_format=torch.contiguous_format).float()
    vae_clothes = torch.stack([example["vae_clothes"] for example in data]).to(
        memory_format=torch.contiguous_format).float()

    clip_image = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embed = [example["drop_image_embed"] for example in data]

    text = [example["text"] for example in data]
    input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    null_input_ids = torch.cat([example["null_text_input_ids"] for example in data], dim=0)

    # File paths for validation saving
    image_files = [example["image_file"] for example in data]
    cloth_files = [example["cloth_file"] for example in data]

    return {
        "vae_person": vae_person,
        "vae_clothes": vae_clothes,
        "clip_image": clip_image,
        "drop_image_embed": drop_image_embed,
        "text": text,
        "input_ids": input_ids,
        "null_input_ids": null_input_ids,
        "image_files": image_files,
        "cloth_files": cloth_files,
    }
