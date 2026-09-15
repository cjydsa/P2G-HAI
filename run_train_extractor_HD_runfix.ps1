# run_train_extractor_HD.ps1
# Two-stage training script for VTON-HD:
#   Stage 1: Generate prompts using DashScope (Qwen) API
#   Stage 2: Train with cached prompts (no API calls)

conda activate IMAGDressing
if (-not $Env:PYTORCH_CUDA_ALLOC_CONF) {
    $Env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
}
if ($Env:PYTORCH_CUDA_ALLOC_CONF -notmatch "garbage_collection_threshold") {
    $Env:PYTORCH_CUDA_ALLOC_CONF = "$Env:PYTORCH_CUDA_ALLOC_CONF,garbage_collection_threshold:0.8"
}
$Env:XFORMERS_DISABLED = "1"
$Env:TOKENIZERS_PARALLELISM = "false"

# ===== Configuration =====
$DATA_ROOT = "E:\BaiduNetdiskDownload\VTON-HD\zalando-hd-resized"
$CACHE_FILE = "E:\BaiduNetdiskDownload\VTON-HD\zalando-hd-resized\prompts_vtonhd_train_paired.jsonl"
$PAIRS_FILE = "E:\BaiduNetdiskDownload\VTON-HD\zalando-hd-resized\train_pairs_paired.txt"
$CACHE_FILE_TEST = "E:\BaiduNetdiskDownload\VTON-HD\zalando-hd-resized\prompts_vtonhd_test_paired.jsonl"
$PAIRS_FILE_TEST = "E:\BaiduNetdiskDownload\VTON-HD\zalando-hd-resized\test_pairs_paired.txt"
$OUTPUT_DIR = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\outputs\extractor_HD"
$SCRIPT_PATH = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\train_extractor_HD_runfix.py"

# ===== Pairing Configuration (CRITICAL: Must be same for Stage-1 and Stage-2) =====
$PAIRING_MODE = "same_name"     # Options: "pairs_file" or "same_name" (recommended: same_name)
$ENFORCE_SAME_NAME = $true      # Validate that all pairs have matching basenames

# Resolve effective enforcement (always on for same_name)
$EFFECTIVE_ENFORCE_SAME_NAME = $ENFORCE_SAME_NAME
if ($PAIRING_MODE -eq "same_name") {
    if (-not $ENFORCE_SAME_NAME) {
        Write-Host "WARNING: pairing_mode is same_name; forcing enforce_same_name_pairs for cache safety." -ForegroundColor Yellow
    }
    $EFFECTIVE_ENFORCE_SAME_NAME = $true
}

# ===== Resume from Checkpoint =====
# Set to checkpoint path to resume training, or $null to start fresh
$RESUME_CHECKPOINT = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\outputs\extractor_HD\model_065000.pt"
# $RESUME_CHECKPOINT = $null  # Uncomment to start fresh training

# ===== DashScope (Qwen) API Configuration =====
$env:DASHSCOPE_API_KEY = "sk-bc487c556e864cddb9f4000d5b51a61a"
$env:OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$OPENAI_PROMPT_MODEL = "qwen3-vl-flash"

# Alternative: OpenAI API (uncomment to use)
# $env:OPENAI_API_KEY = "sk-proj-..."
# $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
# $OPENAI_PROMPT_MODEL = "gpt-4o-mini"

# ===== Prompt Generation Parameters =====
$PROMPT_WORD_MIN = 25
$PROMPT_WORD_MAX = 35
$PROMPT_MAX_RETRIES = 10
$PROMPT_SLEEP_BASE = 1.5

# Image optimization (reduces API cost and speeds up generation)
$PROMPT_IMAGE_MAX_SIDE = 512      # Max image dimension (0=no scaling)
$PROMPT_IMAGE_JPEG_QUALITY = 85   # JPEG quality (1-100)

# DashScope performance tuning (accelerates prompt generation)
$PROMPT_MAX_PIXELS = 147456        # Max pixels for vision: 147456 (384x384) or 65536 (256x256)
$PROMPT_MAX_TOKENS = 80           # Max tokens for response (lower = faster)
$PROMPT_WORKERS = 8               # Concurrent workers for parallel generation

# Sharding configuration (for parallel processing across multiple machines/GPUs)
$N_SHARDS = 1                      # Total number of shards (set to 1 for single machine)
$SHARD_ID = 0                      # Current shard ID (0 to N_SHARDS-1)

# ===== Model Paths =====
$MODEL_PATHS = @{
    pretrained_model = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing"
    vae = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing\sd-vae-ft-mse"
    adapter = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing\models\ip-adapter-plus_sd15.bin"
    image_encoder = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing\image_encoder"
}

# ===== Training Parameters =====
$TRAIN_BATCH_SIZE = 1
$GRADIENT_ACCUMULATION_STEPS = 4
$MAX_TRAIN_STEPS = 100000
$LEARNING_RATE = "1e-4"
$WEIGHT_DECAY = 0.01
$LR_SCHEDULER = "constant_with_warmup"
$NUM_WARMUP_STEPS = 1000
$NOISE_OFFSET = 0.05
$SNR_GAMMA = 3.0
$SAVE_STEPS = 5000
$MILESTONE_STEPS = 200000
$VALIDATION_STEPS = 5000
$VALIDATION_SAMPLES = 50
$VAL_PREVIEW_MAX = 4
$MAX_GRAD_NORM = 1.0

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "VTON-HD Extractor Training - Two Stage Mode" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration Summary:" -ForegroundColor White
Write-Host "  Model: $OPENAI_PROMPT_MODEL" -ForegroundColor Gray
Write-Host "  Prompt length: $PROMPT_WORD_MIN-$PROMPT_WORD_MAX words" -ForegroundColor Gray
Write-Host "  Image optimization: ${PROMPT_IMAGE_MAX_SIDE}px, Q$PROMPT_IMAGE_JPEG_QUALITY" -ForegroundColor Gray
Write-Host "  Performance: ${PROMPT_WORKERS} workers, ${PROMPT_MAX_TOKENS} tokens, ${PROMPT_MAX_PIXELS} max_pixels" -ForegroundColor Gray
Write-Host "  Pairing: mode=$PAIRING_MODE, enforce=$EFFECTIVE_ENFORCE_SAME_NAME" -ForegroundColor Gray
if ($N_SHARDS -gt 1) {
    Write-Host "  Sharding: Shard $SHARD_ID of $N_SHARDS" -ForegroundColor Gray
}
Write-Host ""

# ===== Stage 1: Generate Prompts =====
# NOTE: Set $RUN_STAGE1 = $true to regenerate prompts (requires API key)
#       Set $RUN_STAGE1 = $false to skip Stage-1 and use existing cache
$RUN_STAGE1 = $false  # Default: skip prompt generation
$RUN_STAGE1_TEST = $false
$EXIT_AFTER_STAGE1_TEST = $false

if ($RUN_STAGE1) {
    Write-Host "[Stage 1] Generating prompts using $OPENAI_PROMPT_MODEL..." -ForegroundColor Yellow
    Write-Host "Cache file: $CACHE_FILE" -ForegroundColor Gray
    if ($N_SHARDS -gt 1) {
        Write-Host "NOTE: Running shard $SHARD_ID of $N_SHARDS (parallel processing)" -ForegroundColor Cyan
    }
    Write-Host ""

    # Build Stage 1 command arguments
    $stage1_args = @(
        "--pretrained_model_name_or_path", $MODEL_PATHS.pretrained_model,
        "--pretrained_vae_model_path", $MODEL_PATHS.vae,
        "--pretrained_adapter_model_path", $MODEL_PATHS.adapter,
        "--image_encoder_path", $MODEL_PATHS.image_encoder,
        "--vtonhd_root", $DATA_ROOT,
        "--split", "train",
        "--prompts_cache", $CACHE_FILE,
        "--prompt_mode", "fallback_only",
        "--auto_prepare_prompts",
        "--prepare_prompts_only",
        "--openai_prompt_model", $OPENAI_PROMPT_MODEL,
        "--openai_prompt_max_retries", $PROMPT_MAX_RETRIES,
        "--openai_prompt_sleep_base", $PROMPT_SLEEP_BASE,
        "--openai_prompt_word_min", $PROMPT_WORD_MIN,
        "--openai_prompt_word_max", $PROMPT_WORD_MAX,
        "--openai_prompt_max_side", $PROMPT_IMAGE_MAX_SIDE,
        "--openai_prompt_jpeg_quality", $PROMPT_IMAGE_JPEG_QUALITY,
        "--prompt_max_pixels", $PROMPT_MAX_PIXELS,
        "--prompt_max_tokens", $PROMPT_MAX_TOKENS,
        "--prompt_workers", $PROMPT_WORKERS,
        "--output_dir", $OUTPUT_DIR
    )

    # Add pairing parameters (CRITICAL: must match Stage-2)
    if ($PAIRING_MODE -eq "pairs_file") {
        $stage1_args += "--pairs_file", $PAIRS_FILE
    }
    $stage1_args += "--pairing_mode", $PAIRING_MODE
    if ($EFFECTIVE_ENFORCE_SAME_NAME) {
        $stage1_args += "--enforce_same_name_pairs"
    }

    # Add sharding parameters if using multiple shards
    if ($N_SHARDS -gt 1) {
        $stage1_args += "--n_shards", $N_SHARDS
        $stage1_args += "--shard_id", $SHARD_ID
    }

    # Execute Stage 1: Generate prompts
    Write-Host "Running: python $SCRIPT_PATH $stage1_args" -ForegroundColor Cyan
    python $SCRIPT_PATH @stage1_args

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Stage-1 prompt generation failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host "[Stage 1] Prompt generation completed successfully!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[Stage 1] SKIPPED: Using existing prompt cache at $CACHE_FILE" -ForegroundColor Yellow
    Write-Host "  Set `$RUN_STAGE1 = `$true to regenerate prompts from API" -ForegroundColor Gray
    Write-Host ""
}

# ===== Stage 1-Test: Generate Prompts for Test Split =====
if ($RUN_STAGE1_TEST) {
    Write-Host "[Stage 1-Test] Generating test prompts using $OPENAI_PROMPT_MODEL..." -ForegroundColor Yellow
    Write-Host "Cache file: $CACHE_FILE_TEST" -ForegroundColor Gray
    if ($N_SHARDS -gt 1) {
        Write-Host "NOTE: Running shard $SHARD_ID of $N_SHARDS (parallel processing)" -ForegroundColor Cyan
    }
    Write-Host ""

    # Build Stage 1-Test command arguments
    $stage1_test_args = @(
        "--pretrained_model_name_or_path", $MODEL_PATHS.pretrained_model,
        "--pretrained_vae_model_path", $MODEL_PATHS.vae,
        "--pretrained_adapter_model_path", $MODEL_PATHS.adapter,
        "--image_encoder_path", $MODEL_PATHS.image_encoder,
        "--vtonhd_root", $DATA_ROOT,
        "--split", "test",
        "--prompts_cache", $CACHE_FILE_TEST,
        "--prompt_mode", "fallback_only",
        "--auto_prepare_prompts",
        "--prepare_prompts_only",
        "--openai_prompt_model", $OPENAI_PROMPT_MODEL,
        "--openai_prompt_max_retries", $PROMPT_MAX_RETRIES,
        "--openai_prompt_sleep_base", $PROMPT_SLEEP_BASE,
        "--openai_prompt_word_min", $PROMPT_WORD_MIN,
        "--openai_prompt_word_max", $PROMPT_WORD_MAX,
        "--openai_prompt_max_side", $PROMPT_IMAGE_MAX_SIDE,
        "--openai_prompt_jpeg_quality", $PROMPT_IMAGE_JPEG_QUALITY,
        "--prompt_max_pixels", $PROMPT_MAX_PIXELS,
        "--prompt_max_tokens", $PROMPT_MAX_TOKENS,
        "--prompt_workers", $PROMPT_WORKERS,
        "--output_dir", $OUTPUT_DIR
    )

    # Add pairing parameters (CRITICAL: must match Stage-2)
    if ($PAIRING_MODE -eq "pairs_file") {
        $stage1_test_args += "--pairs_file", $PAIRS_FILE_TEST
    }
    $stage1_test_args += "--pairing_mode", $PAIRING_MODE
    if ($EFFECTIVE_ENFORCE_SAME_NAME) {
        $stage1_test_args += "--enforce_same_name_pairs"
    }

    # Add sharding parameters if using multiple shards
    if ($N_SHARDS -gt 1) {
        $stage1_test_args += "--n_shards", $N_SHARDS
        $stage1_test_args += "--shard_id", $SHARD_ID
    }

    # Execute Stage 1-Test: Generate prompts
    Write-Host "Running: python $SCRIPT_PATH $stage1_test_args" -ForegroundColor Cyan
    python $SCRIPT_PATH @stage1_test_args

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Stage-1 test prompt generation failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host "[Stage 1-Test] Done! Cache written to: $CACHE_FILE_TEST" -ForegroundColor Green
    Write-Host ""

    if ($EXIT_AFTER_STAGE1_TEST) {
        exit 0
    }
} else {
    Write-Host "[Stage 1-Test] SKIPPED: Using existing test prompt cache at $CACHE_FILE_TEST" -ForegroundColor Yellow
    Write-Host "  Set `$RUN_STAGE1_TEST = `$true to regenerate test prompts from API" -ForegroundColor Gray
    Write-Host ""
}



# ===== Stage 2: Training with Cached Prompts =====
Write-Host "[Stage 2] Starting training with cached prompts..." -ForegroundColor Yellow
Write-Host "NOTE: No API calls will be made during training (cache_only mode)" -ForegroundColor Gray
Write-Host ""
Write-Host "IMPORTANT: Pairing Configuration" -ForegroundColor Cyan
Write-Host "  - pairing_mode: $PAIRING_MODE" -ForegroundColor Gray
Write-Host "  - enforce_same_name_pairs: $EFFECTIVE_ENFORCE_SAME_NAME" -ForegroundColor Gray
if ($PAIRING_MODE -ne "same_name") {
    Write-Host "  WARNING: Using pairs_file mode - ensure Stage-1 used the SAME pairs file!" -ForegroundColor Yellow
}
Write-Host "  If you change pairing rules, delete the old prompt cache file!" -ForegroundColor Yellow
Write-Host ""

# Check resume checkpoint
if ($RESUME_CHECKPOINT -and (Test-Path $RESUME_CHECKPOINT)) {
    Write-Host "Resume Training Configuration:" -ForegroundColor Green
    Write-Host "  Resuming from checkpoint: $RESUME_CHECKPOINT" -ForegroundColor Gray
    Write-Host ""
} elseif ($RESUME_CHECKPOINT) {
    Write-Host "ERROR: Checkpoint file not found: $RESUME_CHECKPOINT" -ForegroundColor Red
    Write-Host "  Please verify the checkpoint path or set `$RESUME_CHECKPOINT = `$null to start fresh" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "Training Mode: Starting fresh (no checkpoint)" -ForegroundColor Cyan
    Write-Host ""
}

# Build accelerate launch command arguments
$train_args = @(
    "--pretrained_model_name_or_path", $MODEL_PATHS.pretrained_model,
    "--pretrained_vae_model_path", $MODEL_PATHS.vae,
    "--pretrained_adapter_model_path", $MODEL_PATHS.adapter,
    "--image_encoder_path", $MODEL_PATHS.image_encoder,
    "--vtonhd_root", $DATA_ROOT,
    "--split", "train",
    "--pairing_mode", $PAIRING_MODE,
    "--prompts_cache", $CACHE_FILE,
    "--prompt_mode", "cache_only",
    "--train_batch_size", $TRAIN_BATCH_SIZE,
    "--gradient_accumulation_steps", $GRADIENT_ACCUMULATION_STEPS,
    "--max_train_steps", $MAX_TRAIN_STEPS,
    "--learning_rate", $LEARNING_RATE,
    "--weight_decay", $WEIGHT_DECAY,
    "--lr_scheduler", $LR_SCHEDULER,
    "--num_warmup_steps", $NUM_WARMUP_STEPS,
    "--output_dir", $OUTPUT_DIR,
    "--noise_offset", $NOISE_OFFSET,
    "--snr_gamma", $SNR_GAMMA,
    "--save_steps", $SAVE_STEPS,
    "--milestone_steps", $MILESTONE_STEPS,
    "--validation_steps", $VALIDATION_STEPS,
    "--validation_samples", $VALIDATION_SAMPLES,
    "--val_preview_max", $VAL_PREVIEW_MAX,
    "--max_grad_norm", $MAX_GRAD_NORM
)

# Add pairing parameters (CRITICAL: must match Stage-1)
if ($PAIRING_MODE -eq "pairs_file") {
    $train_args += "--pairs_file", $PAIRS_FILE
}
if ($EFFECTIVE_ENFORCE_SAME_NAME) {
    $train_args += "--enforce_same_name_pairs"
}

# Add checkpoint resumption parameter if specified
if ($RESUME_CHECKPOINT -and (Test-Path $RESUME_CHECKPOINT)) {
    $train_args += "--resume_from_checkpoint", $RESUME_CHECKPOINT
}

& accelerate launch `
  --mixed_precision "bf16" `
  --gpu_ids 0 `
  --num_processes 1 `
  $SCRIPT_PATH `
  @train_args

if ($LASTEXITCODE -eq 0) {
    Write-Host "" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
} else {
    Write-Host "" -ForegroundColor Red
    Write-Host "Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}