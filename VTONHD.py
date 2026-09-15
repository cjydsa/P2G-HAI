import json
import random
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from transformers import CLIPImageProcessor

# Allow loading truncated/damaged images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_pair_key(image_file, cloth_file):
    """
    Generate a consistent cache key for a (person, cloth) pair.

    Args:
        image_file: Person image filename
        cloth_file: Cloth image filename

    Returns:
        str: Cache key in format "image_file|||cloth_file"
    """
    return f"{image_file}|||{cloth_file}"


class VTONHDDataset(Dataset):
    """
    VTON-HD Dataset loader for training garment extraction model.

    Args:
        root: Dataset root directory (zalando-hd-resized)
        split: Dataset split ("train" or "test")
        tokenizer: CLIP tokenizer
        pairs_file: Path to pairs.txt (auto-detect if None)
        prompts_cache: Path to prompt cache file (json or jsonl), or None/empty string
        prompt_mode: Cache usage mode:
            - "cache_only": STRICT mode - only use cache, raise error on miss (for training)
            - "cache_or_fallback": Try cache first, use fallback on miss (for testing)
            - "fallback_only": Always use fallback (for prompt generation preprocessing)
        prompt_fallback: Default prompt when cache is missing (default: "" for safety)
        size: Image size for VAE (default 512)
        verify_files: If True, verify all person/cloth files exist before training (default: True)
        pairing_mode: Pairing strategy ("pairs_file" or "same_name")
            - "pairs_file": Read pairs from pairs_file (default)
            - "same_name": Generate pairs by matching filenames (00000.00.jpg <-> 00000.00.jpg)
        enforce_same_name_pairs: If True, validate that all pairs have matching basenames
            and raise error if not (default: False)

    Raises:
        FileNotFoundError: If required directories/files not found
        RuntimeError: If verify_files=True and files are missing
        RuntimeError: If prompt_mode=cache_only and cache is incomplete
        RuntimeError: If enforce_same_name_pairs=True and pairs have mismatched names
    """
    def __init__(
            self,
            root,
            split,
            tokenizer,
            pairs_file=None,
            prompts_cache=None,
            prompt_mode="cache_or_fallback",
            prompt_fallback="",
            size=512,
            verify_files=True,
            pairing_mode="pairs_file",
            enforce_same_name_pairs=False,
    ):
        self.root = root
        self.split = split
        self.tokenizer = tokenizer
        self.size = size
        self.prompt_mode = prompt_mode
        self.prompt_fallback = prompt_fallback
        self.pairing_mode = pairing_mode
        self.enforce_same_name_pairs = enforce_same_name_pairs

        # Validate pairing_mode
        valid_pairing_modes = ["pairs_file", "same_name"]
        if pairing_mode not in valid_pairing_modes:
            raise ValueError(f"Invalid pairing_mode: {pairing_mode}. Must be one of {valid_pairing_modes}")

        # Validate prompt_mode
        valid_modes = ["cache_only", "cache_or_fallback", "fallback_only"]
        if prompt_mode not in valid_modes:
            raise ValueError(f"Invalid prompt_mode: {prompt_mode}. Must be one of {valid_modes}")

        # Auto-detect directory structure
        self.person_dir = self._find_directory("person")
        self.cloth_dir = self._find_directory("cloth")

        # Auto-detect pairs.txt only when needed for pairs_file mode
        if self.pairing_mode == "pairs_file" and pairs_file is None:
            pairs_file = self._find_pairs_file()

        print(f"[VTONHDDataset] Loading VTON-HD from: {root}")
        print(f"  Split: {split}")
        print(f"  Person dir: {self.person_dir}")
        print(f"  Cloth dir: {self.cloth_dir}")
        print(f"  Pairing mode: {pairing_mode}")
        print(f"  Pairs file: {pairs_file}")
        print(f"  Prompt mode: {prompt_mode}")

        # Load pairs based on pairing_mode
        self.data = self._load_pairs(pairs_file)

        # Enforce same-name validation if requested
        if self.enforce_same_name_pairs:
            self._validate_same_name_pairs()

        # Verify files exist (optional but recommended)
        if verify_files:
            self._verify_files()

        # Load prompt cache
        self.prompt_cache = {}
        if prompts_cache and os.path.exists(prompts_cache):
            self._load_prompt_cache(prompts_cache)
            print(f"  Prompt cache loaded: {len(self.prompt_cache)} entries")

            # Check cache coverage if not in fallback_only mode
            if prompt_mode != "fallback_only":
                coverage = self._check_cache_coverage()
                print(f"  Cache coverage: {coverage:.1f}%")

                # STRICT: cache_only requires 100% coverage
                if prompt_mode == "cache_only" and coverage < 100.0:
                    missing_keys = self._get_missing_cache_keys()
                    error_msg = (
                        f"[PromptCacheIncomplete] prompt_mode=cache_only requires 100% cache coverage, "
                        f"but only {coverage:.1f}% ({len(self.prompt_cache)}/{len(self.data)}) is cached.\n\n"
                        f"Missing prompts for {len(missing_keys)} pairs.\n"
                    )

                    # Show first N missing examples
                    show_count = min(20, len(missing_keys))
                    error_msg += f"\nFirst {show_count} missing cache keys:\n"
                    for i in range(show_count):
                        error_msg += f"  - {missing_keys[i]}\n"
                    if len(missing_keys) > show_count:
                        error_msg += f"  ... and {len(missing_keys) - show_count} more\n"

                    error_msg += (
                        f"\n💡 Solution: Run Stage-1 to generate prompts:\n"
                        f"   python train_extractor_HD.py ... --auto_prepare_prompts --prepare_prompts_only\n"
                    )

                    raise RuntimeError(error_msg)

        else:
            # STRICT: cache_only requires cache file to exist
            if prompt_mode == "cache_only":
                raise FileNotFoundError(
                    f"[PromptCacheMissing] prompt_mode=cache_only requires a valid cache file, "
                    f"but prompts_cache='{prompts_cache}' does not exist.\n\n"
                    f"💡 Solution: Run Stage-1 to generate prompts:\n"
                    f"   python train_extractor_HD.py ... --auto_prepare_prompts --prepare_prompts_only"
                )
            elif prompt_mode == "cache_or_fallback":
                print(f"  No cache file found, will use fallback prompt")
            print(f"  Fallback prompt: '{prompt_fallback}'")

        print(f"[VTONHDDataset] Loaded {len(self.data)} pairs")

        # Transforms (consistent with IGPair)
        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()

        # Stats tracking for cache misses
        self.cache_miss_count = 0

    def _find_directory(self, dir_type):
        """
        Auto-detect person/cloth directory with multiple candidate paths.
        For person images, prioritize .../<split>/image structure.

        Args:
            dir_type: "person" or "cloth"

        Returns:
            str: Detected directory path
        """
        if dir_type == "person":
            # PRIORITY: .../<split>/image (user's data structure)
            candidates = [
                os.path.join(self.root, self.split, "image"),
                os.path.join(self.root, self.split, "images"),
                os.path.join(self.root, self.split, "img"),
                os.path.join(self.root, "image", self.split),
                os.path.join(self.root, "images", self.split),
            ]
        else:  # cloth
            candidates = [
                os.path.join(self.root, self.split, "cloth"),
                os.path.join(self.root, self.split, "cloths"),
                os.path.join(self.root, "cloth", self.split),
                os.path.join(self.root, "cloths", self.split),
            ]

        for path in candidates:
            if os.path.exists(path) and os.path.isdir(path):
                return path

        raise FileNotFoundError(
            f"Could not find {dir_type} directory. Tried:\n" +
            "\n".join(f"  - {p}" for p in candidates)
        )

    def _find_pairs_file(self):
        """
        Auto-detect pairs.txt file.

        Returns:
            str: Detected pairs file path
        """
        candidates = [
            os.path.join(self.root, f"{self.split}_pairs.txt"),
            os.path.join(self.root, self.split, "pairs.txt"),
            os.path.join(self.root, f"pairs_{self.split}.txt"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Could not find pairs file. Tried:\n" +
            "\n".join(f"  - {p}" for p in candidates)
        )

    def _load_pairs(self, pairs_path):
        """
        Load pairs based on pairing_mode.

        When pairing_mode='same_name': scan directories and match by filename
        When pairing_mode='pairs_file': read from pairs.txt file

        Format (pairs_file): person.jpg cloth.jpg (space or tab separated)

        Returns:
            list: List of dicts with image_file and cloth_file
        """
        if self.pairing_mode == "same_name":
            # Same-name pairing: scan directories and match filenames
            image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}

            # Scan person directory
            person_files = set()
            for fname in os.listdir(self.person_dir):
                if any(fname.endswith(ext) for ext in image_exts):
                    person_files.add(fname)

            # Scan cloth directory
            cloth_files = set()
            for fname in os.listdir(self.cloth_dir):
                if any(fname.endswith(ext) for ext in image_exts):
                    cloth_files.add(fname)

            # Find intersection (same-name pairs)
            common_files = sorted(person_files & cloth_files)

            if not common_files:
                raise RuntimeError(
                    f"[PairingError] pairing_mode=same_name found NO matching filenames!\n"
                    f"  Person dir has {len(person_files)} images\n"
                    f"  Cloth dir has {len(cloth_files)} images\n"
                    f"  Intersection: 0 files\n\n"
                    f"Please ensure person and cloth directories contain images with matching filenames."
                )

            pairs = [{"image_file": fname, "cloth_file": fname} for fname in common_files]

            print(f"  Found {len(pairs)} same-name pairs")
            print(f"  [same_name] Person images: {len(person_files)}, cloth images: {len(cloth_files)}")
            print(f"  [same_name] First 5 pairs:")
            for i, fname in enumerate(common_files[:5]):
                print(f"    {i+1}. {fname} <-> {fname}")

            # Optionally write pairs to pairs_file for reference
            if pairs_path and not os.path.exists(pairs_path):
                os.makedirs(os.path.dirname(pairs_path) or ".", exist_ok=True)
                with open(pairs_path, 'w') as f:
                    for pair in pairs:
                        f.write(f"{pair['image_file']} {pair['cloth_file']}\n")
                print(f"  [same_name] Saved pairs to: {pairs_path}")

            return pairs

        else:  # pairs_file mode
            pairs = []
            with open(pairs_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Split by whitespace (space or tab)
                    parts = line.split()
                    if len(parts) < 2:
                        print(f"WARNING: Invalid line {line_num} in {pairs_path}: {line}")
                        continue

                    person_file = parts[0]
                    cloth_file = parts[1]

                    # Store relative paths
                    pairs.append({
                        "image_file": person_file,
                        "cloth_file": cloth_file,
                    })

            return pairs

    def _validate_same_name_pairs(self):
        """
        Validate that all pairs have matching basenames.
        Raises RuntimeError if any pair has mismatched names.

        This ensures prompt cache keys will be consistent when using same-name pairing.
        """
        mismatches = []
        for item in self.data:
            person_basename = os.path.basename(item["image_file"])
            cloth_basename = os.path.basename(item["cloth_file"])

            if person_basename != cloth_basename:
                mismatches.append((item["image_file"], item["cloth_file"]))

        if mismatches:
            error_msg = f"[PairingValidationError] enforce_same_name_pairs=True but found {len(mismatches)} mismatched pairs!\n\n"
            error_msg += f"First {min(20, len(mismatches))} mismatched pairs:\n"
            for i, (img, cloth) in enumerate(mismatches[:20]):
                error_msg += f"  {i+1}. {img} != {cloth}\n"
            if len(mismatches) > 20:
                error_msg += f"  ... and {len(mismatches) - 20} more\n"

            error_msg += (
                f"\n💡 Solution:\n"
                f"  1. Use --pairing_mode same_name to generate correct same-name pairs, OR\n"
                f"  2. Fix the pairs file to use matching filenames, OR\n"
                f"  3. Remove --enforce_same_name_pairs if cross-pairing is intentional\n"
            )

            raise RuntimeError(error_msg)

        print(f"  ✓ All {len(self.data)} pairs validated (basenames match)")

    def _verify_files(self):
        """
        Verify that all person and cloth files referenced in pairs actually exist.
        Raises RuntimeError if any files are missing.
        """
        print(f"[VTONHDDataset] Verifying file existence for {len(self.data)} pairs...")

        missing_person = []
        missing_cloth = []

        for item in self.data:
            person_path = os.path.join(self.person_dir, item["image_file"])
            cloth_path = os.path.join(self.cloth_dir, item["cloth_file"])

            if not os.path.exists(person_path):
                missing_person.append(item["image_file"])

            if not os.path.exists(cloth_path):
                missing_cloth.append(item["cloth_file"])

        # Report findings
        total_missing = len(missing_person) + len(missing_cloth)

        if total_missing > 0:
            error_msg = f"[FileVerificationError] Found {total_missing} missing files:\n"

            if missing_person:
                error_msg += f"\n  Missing person images: {len(missing_person)}\n"
                # Show first N examples
                show_count = min(20, len(missing_person))
                for i in range(show_count):
                    error_msg += f"    - {missing_person[i]}\n"
                if len(missing_person) > show_count:
                    error_msg += f"    ... and {len(missing_person) - show_count} more\n"

            if missing_cloth:
                error_msg += f"\n  Missing cloth images: {len(missing_cloth)}\n"
                # Show first N examples
                show_count = min(20, len(missing_cloth))
                for i in range(show_count):
                    error_msg += f"    - {missing_cloth[i]}\n"
                if len(missing_cloth) > show_count:
                    error_msg += f"    ... and {len(missing_cloth) - show_count} more\n"

            error_msg += f"\nPlease ensure all files exist in:\n"
            error_msg += f"  Person dir: {self.person_dir}\n"
            error_msg += f"  Cloth dir: {self.cloth_dir}\n"

            raise RuntimeError(error_msg)

        print(f"  ✓ All {len(self.data)} pairs verified (person and cloth files exist)")

    def _load_prompt_cache(self, cache_path):
        """
        Load prompt cache from json or jsonl file.

        Formats supported:
        - JSON: {"key": "prompt", ...} or {"key": ["prompt1", "prompt2"], ...}
        - JSONL: {"key": "...", "prompt": "..."} (one entry per line)
                 {"key": "...", "prompts": ["..."]} (multi-prompt per line)

        Keys must be in format: "image_file|||cloth_file" (use make_pair_key())
        """
        ext = os.path.splitext(cache_path)[1].lower()

        if ext == '.json':
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                # Normalize to dict[key] -> list[str]
                for key, value in cache.items():
                    if isinstance(value, str):
                        self.prompt_cache[key] = [value]
                    elif isinstance(value, list):
                        self.prompt_cache[key] = value
                    else:
                        print(f"WARNING: Invalid cache value for key {key}: {value}")

        elif ext == '.jsonl':
            with open(cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = obj.get("key")
                    if not key:
                        continue

                    # Support both "prompt" (str) and "prompts" (list)
                    if "prompt" in obj:
                        prompt_value = obj["prompt"]
                        if isinstance(prompt_value, str):
                            self.prompt_cache[key] = [prompt_value]
                        elif isinstance(prompt_value, list):
                            self.prompt_cache[key] = prompt_value
                    elif "prompts" in obj:
                        prompts = obj["prompts"]
                        if isinstance(prompts, str):
                            self.prompt_cache[key] = [prompts]
                        elif isinstance(prompts, list):
                            self.prompt_cache[key] = prompts
        else:
            raise ValueError(f"Unsupported cache format: {ext} (expected .json or .jsonl)")

    def _check_cache_coverage(self):
        """
        Calculate what percentage of dataset pairs have cached prompts.

        Returns:
            float: Coverage percentage (0-100)
        """
        if not self.data:
            return 0.0

        covered = 0
        for item in self.data:
            key = make_pair_key(item['image_file'], item['cloth_file'])
            if key in self.prompt_cache:
                covered += 1

        return (covered / len(self.data)) * 100.0

    def _get_missing_cache_keys(self):
        """
        Get list of cache keys that are missing for current dataset.

        Returns:
            list: List of missing cache keys
        """
        missing = []
        for item in self.data:
            key = make_pair_key(item['image_file'], item['cloth_file'])
            if key not in self.prompt_cache:
                missing.append(key)
        return missing

    def _get_prompt_for_item(self, item):
        """
        Get prompt for item based on prompt_mode.

        Args:
            item: Data item dict

        Returns:
            str: Prompt text

        Raises:
            RuntimeError: If cache_only mode and cache miss
        """
        # Generate cache key using global function
        key = make_pair_key(item['image_file'], item['cloth_file'])

        if self.prompt_mode == "fallback_only":
            # Always use fallback (for preprocessing)
            return self.prompt_fallback

        elif self.prompt_mode == "cache_only":
            # STRICT: Must hit cache, no fallback allowed
            if key in self.prompt_cache:
                prompts = self.prompt_cache[key]
                return random.choice(prompts) if isinstance(prompts, list) else prompts

            # Cache miss - this should NOT happen if __init__ validation passed
            # But defensive check here in case of runtime issues
            raise RuntimeError(
                f"[CRITICAL] Cache miss during training in cache_only mode!\n"
                f"  Key: {key}\n"
                f"  Person: {item['image_file']}\n"
                f"  Cloth: {item['cloth_file']}\n\n"
                f"This indicates the prompt cache is incomplete or corrupted.\n"
                f"Please re-run Stage-1 to regenerate prompts:\n"
                f"  python train_extractor_HD.py ... --auto_prepare_prompts --prepare_prompts_only"
            )

        elif self.prompt_mode == "cache_or_fallback":
            # Try cache first, fallback on miss
            if key in self.prompt_cache:
                prompts = self.prompt_cache[key]
                return random.choice(prompts) if isinstance(prompts, list) else prompts

            # Cache miss - use fallback (with optional warning)
            return self.prompt_fallback

        else:
            raise ValueError(f"Invalid prompt_mode: {self.prompt_mode}")

    def _safe_open_rgb(self, path, max_side=2048):
        """
        Safely open and convert image to RGB with memory-efficient preprocessing.
        Copied from IGPair.py to prevent OOM on large images.

        Args:
            path: Image file path
            max_side: Maximum dimension to decode (reduces memory for large images)

        Returns:
            RGB PIL Image
        """
        try:
            img = Image.open(path)

            # Early size check - reject extremely large images
            if hasattr(img, 'size'):
                w, h = img.size
                if w > 8192 or h > 8192:
                    # Image too large, resize immediately
                    img = img.resize((min(w, max_side), min(h, max_side)), Image.BILINEAR)

            # Use draft mode for JPEG to decode at lower resolution (saves memory)
            try:
                img.draft("RGB", (max_side, max_side))
            except Exception:
                pass

            # Resize before conversion to avoid full-size decode
            img.thumbnail((max_side, max_side), Image.BILINEAR)

            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            return img

        except MemoryError:
            # Fallback: force small size immediately
            with Image.open(path) as raw:
                # Resize to target size directly without intermediate decode
                small_img = raw.resize((self.size, self.size), Image.BILINEAR)
                return small_img.convert("RGB")
        except Exception as e:
            # Log error and return a black placeholder to avoid crashing training
            print(f"WARNING: Failed to load image {path}: {e}")
            return Image.new("RGB", (self.size, self.size), (0, 0, 0))

    def __getitem__(self, idx):
        item = self.data[idx]

        # Construct full paths
        person_path = os.path.join(self.person_dir, item["image_file"])
        cloth_path = os.path.join(self.cloth_dir, item["cloth_file"])

        # Load images safely
        person_img = self._safe_open_rgb(person_path, max_side=2048)
        clothes_img = self._safe_open_rgb(cloth_path, max_side=2048)

        # Get prompt (from cache or fallback based on prompt_mode)
        text = self._get_prompt_for_item(item)

        # VTON-HD is upper-body dataset
        text_with_prefix = text

        # Apply text dropping logic
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < 0.05:
            drop_image_embed = 1
        elif rand_num < 0.1:
            # Drop the text
            text_with_prefix = ""
        elif rand_num < 0.15:
            # Drop both text and image embedding
            text_with_prefix = ""
            drop_image_embed = 1

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

        # CRITICAL: Use PERSON image for CLIP (reverse task: person → garment)
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
