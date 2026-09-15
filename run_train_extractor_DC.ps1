# run_train_extractor_DC.ps1
# Two-stage training script for DressCode:
#   Stage 1: Generate prompts using DashScope (Qwen) API
#   Stage 2: Train with cached prompts (no API calls)

$ErrorActionPreference = "Stop"
conda activate IMAGDressing

function Add-AllocConfItem([string]$conf, [string]$key, [string]$value) {
    if ($conf -match "(^|,)${key}:") {
        return $conf
    }
    if ([string]::IsNullOrWhiteSpace($conf)) {
        return "${key}:${value}"
    }
    return "${conf},${key}:${value}"
}

$allocConf = $Env:PYTORCH_CUDA_ALLOC_CONF
$allocConf = Add-AllocConfItem $allocConf "max_split_size_mb" "128"
$allocConf = Add-AllocConfItem $allocConf "garbage_collection_threshold" "0.8"
$allocConf = Add-AllocConfItem $allocConf "expandable_segments" "True"
$Env:PYTORCH_CUDA_ALLOC_CONF = $allocConf

$ENABLE_XFORMERS = $false
if ($ENABLE_XFORMERS) {
    $Env:XFORMERS_DISABLED = "0"
} elseif (-not $Env:XFORMERS_DISABLED) {
    $Env:XFORMERS_DISABLED = "1"
}
$Env:TOKENIZERS_PARALLELISM = "false"

Write-Host "Environment:" -ForegroundColor White
Write-Host "  Python:" -ForegroundColor Gray
python -c "import sys; print(sys.version)"
Write-Host "  Torch:" -ForegroundColor Gray
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda)"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "  NVIDIA-SMI:" -ForegroundColor Gray
    nvidia-smi
} else {
    Write-Host "  nvidia-smi not found" -ForegroundColor Yellow
}

# ===== Configuration =====
$DRESSCODE_ROOT = "E:\BaiduNetdiskDownload\DressCode"
$PROMPTS_CACHE = "$DRESSCODE_ROOT\prompts_dresscode_cache.jsonl"
$CATEGORY = "all"
$TEST_ORDER = "paired"
$OUTPUT_DIR = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\outputs\extractor_DC"
$SCRIPT_PATH = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\train_extractor_DC.py"

# ===== Resume from Checkpoint =====
# Set to checkpoint path to resume training, or $null to start fresh
$RESUME_CHECKPOINT = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\outputs\extractor_DC\model_075000.pt"
# $RESUME_CHECKPOINT = $null  # Uncomment to start fresh training

# ===== DashScope (Qwen) API Configuration =====
# Set DASHSCOPE_API_KEY or OPENAI_API_KEY in your shell environment before running this script.
$env:DASHSCOPE_API_KEY = "sk-bc487c556e864cddb9f4000d5b51a61a"
$env:OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$OPENAI_PROMPT_MODEL = "qwen3-vl-flash"

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
$NUM_PROCESSES = 1
$DATALOADER_NUM_WORKERS = 0   # Windows stability: set to 2/4 for speed if stable
$GRADIENT_CHECKPOINTING = $true
$STRICT_PROMPTS_CACHE = $false
$SEED = 42
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
$TRAIN_PROMPT_MODE = "cache_only"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "DressCode Extractor Training - Two Stage Mode" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "" 
Write-Host "Configuration Summary:" -ForegroundColor White
Write-Host "  Dataset root: $DRESSCODE_ROOT" -ForegroundColor Gray
Write-Host "  Category: $CATEGORY" -ForegroundColor Gray
Write-Host "  Test order: $TEST_ORDER" -ForegroundColor Gray
Write-Host "  Prompt cache: $PROMPTS_CACHE" -ForegroundColor Gray
Write-Host "  Model: $OPENAI_PROMPT_MODEL" -ForegroundColor Gray
Write-Host "  Prompt length: $PROMPT_WORD_MIN-$PROMPT_WORD_MAX words" -ForegroundColor Gray
Write-Host "  Image optimization: ${PROMPT_IMAGE_MAX_SIDE}px, Q$PROMPT_IMAGE_JPEG_QUALITY" -ForegroundColor Gray
Write-Host "  Performance: ${PROMPT_WORKERS} workers, ${PROMPT_MAX_TOKENS} tokens, ${PROMPT_MAX_PIXELS} max_pixels" -ForegroundColor Gray
Write-Host "  Num processes: $NUM_PROCESSES" -ForegroundColor Gray
Write-Host "  DataLoader workers: $DATALOADER_NUM_WORKERS" -ForegroundColor Gray
Write-Host "  Xformers enabled: $ENABLE_XFORMERS" -ForegroundColor Gray
Write-Host "  Seed: $SEED" -ForegroundColor Gray
if ($N_SHARDS -gt 1) {
    Write-Host "  Sharding: Shard $SHARD_ID of $N_SHARDS" -ForegroundColor Gray
}
Write-Host ""

# ===== Stage 1: Generate Prompts =====
# NOTE: Set $RUN_STAGE1 = $true to regenerate prompts (requires API key)
#       Set $RUN_STAGE1 = $false to skip Stage-1 and use existing cache
$RUN_STAGE1 = $true  # Default: skip prompt generation

if ($RUN_STAGE1) {
    Write-Host "[Stage 1] Generating prompts using $OPENAI_PROMPT_MODEL..." -ForegroundColor Yellow
    Write-Host "Cache file: $PROMPTS_CACHE" -ForegroundColor Gray
    if (-not $env:DASHSCOPE_API_KEY -and -not $env:OPENAI_API_KEY) {
        Write-Host "WARNING: DASHSCOPE_API_KEY or OPENAI_API_KEY is not set; Stage 1 may fail." -ForegroundColor Yellow
    }
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
        "--dresscode_root", $DRESSCODE_ROOT,
        "--dresscode_category", $CATEGORY,
        "--dresscode_test_order", $TEST_ORDER,
        "--split", "train",
        "--prompts_cache", $PROMPTS_CACHE,
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
    Write-Host "[Stage 1] SKIPPED: Using existing prompt cache at $PROMPTS_CACHE" -ForegroundColor Yellow
    Write-Host "  Set `$RUN_STAGE1 = `$true to regenerate prompts from API" -ForegroundColor Gray
    Write-Host ""
}

if ($TRAIN_PROMPT_MODE -eq "cache_only" -and -not $RUN_STAGE1) {
    Write-Host "WARNING: prompt_mode=cache_only and prepare_prompts_only is disabled; missing cache will hard-fail." -ForegroundColor Yellow
    Write-Host "  For first run, set `$RUN_STAGE1 = `$true or run with --auto_prepare_prompts --prepare_prompts_only." -ForegroundColor Yellow
    Write-Host ""
}

# ===== Stage 2: Training with Cached Prompts =====
Write-Host "[Stage 2] Starting training with cached prompts..." -ForegroundColor Yellow
Write-Host "NOTE: No API calls will be made during training (cache_only mode)" -ForegroundColor Gray
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
    "--dresscode_root", $DRESSCODE_ROOT,
    "--dresscode_category", $CATEGORY,
    "--dresscode_test_order", $TEST_ORDER,
    "--split", "train",
    "--prompts_cache", $PROMPTS_CACHE,
    "--prompt_mode", $TRAIN_PROMPT_MODE,
    "--train_batch_size", $TRAIN_BATCH_SIZE,
    "--dataloader_num_workers", $DATALOADER_NUM_WORKERS,
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

if ($SEED -ne $null) {
    $train_args += "--seed", $SEED
}

if ($GRADIENT_CHECKPOINTING) {
    $train_args += "--gradient_checkpointing"
} else {
    $train_args += "--no_gradient_checkpointing"
}

if ($ENABLE_XFORMERS) {
    $train_args += "--enable_xformers_memory_efficient_attention"
}

if ($STRICT_PROMPTS_CACHE) {
    $train_args += "--strict_prompts_cache"
}

# Add checkpoint resumption parameter if specified
if ($RESUME_CHECKPOINT -and (Test-Path $RESUME_CHECKPOINT)) {
    $train_args += "--resume_from_checkpoint", $RESUME_CHECKPOINT
}

$accelerateLaunchArgs = @(
    "--mixed_precision", "bf16",
    "--gpu_ids", "0",
    "--num_processes", $NUM_PROCESSES,
    "$SCRIPT_PATH"
) + $train_args

$accelerateCmd = Get-Command accelerate -ErrorAction SilentlyContinue
if ($accelerateCmd) {
    & accelerate launch @accelerateLaunchArgs
} else {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Host "ERROR: python not found in PATH; cannot launch accelerate." -ForegroundColor Red
        exit 1
    }
    Write-Host "WARNING: accelerate command not found; falling back to python -m accelerate.commands.launch" -ForegroundColor Yellow
    & python -m accelerate.commands.launch @accelerateLaunchArgs
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
} else {
    Write-Host "" -ForegroundColor Red
    Write-Host "Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
