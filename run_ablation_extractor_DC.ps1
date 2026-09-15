# run_ablation_extractor_DC.ps1
# Eval-only ablations for DressCode extractor: S0/S1/S2 only
# It will run: train_extractor_DC.py in --eval_only mode for each S mode.

$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    $inEnv = ($env:CONDA_DEFAULT_ENV -eq "IMAGDressing")
    if ($inEnv -or -not $condaCmd) {
        return @{
            Exe = "python"
            PrefixArgs = @()
            UsingCondaRun = $false
        }
    }
    return @{
        Exe = "conda"
        PrefixArgs = @("run", "-n", "IMAGDressing", "python")
        UsingCondaRun = $true
    }
}

$pythonCmd = Resolve-PythonCommand
$PYTHON_EXE = $pythonCmd.Exe
$PYTHON_PREFIX = $pythonCmd.PrefixArgs

if (-not ($env:CONDA_DEFAULT_ENV -eq "IMAGDressing") -and (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Using: conda run -n IMAGDressing python" -ForegroundColor Gray
}

function Add-AllocConfItem([string]$conf, [string]$key, [string]$value) {
    if ($conf -match "(^|,)${key}:") { return $conf }
    if ([string]::IsNullOrWhiteSpace($conf)) { return "${key}:${value}" }
    return "${conf},${key}:${value}"
}

# ===== Environment (copied from run_train_extractor_DC.ps1 style) =====
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
& $PYTHON_EXE @PYTHON_PREFIX -c "import sys; print(sys.version)"
Write-Host "  Torch:" -ForegroundColor Gray
& $PYTHON_EXE @PYTHON_PREFIX -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda)"

# ===== Configuration (customize paths) =====
$PROJECT_ROOT   = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing"
$CHECKPOINT     = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\outputs\extractor_DC\model_075000.pt"
$OUTPUT_ROOT    = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\outputs\ablation_DC_evalonly"
$TEST_SET_JSON  = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\evaluation_results_DC\assets\test_subset_500.json"
$DRESSCODE_ROOT = "E:\BaiduNetdiskDownload\DressCode"
$PROMPTS_CACHE  = "E:\BaiduNetdiskDownload\DressCode\prompts_dresscode_cache.jsonl"
$CATEGORY       = "all"
$TEST_ORDER     = "paired"

$TRAIN_SCRIPT   = "$PROJECT_ROOT\train_extractor_DC.py"
$RUNNER_SCRIPT  = "$PROJECT_ROOT\run_ablation_extractor_DC.py"

# Ablation grid (S-only)
$S_MODES = @("S0", "S1", "S2")
$ONLY_RUN_INCOMPLETE = $true
# Evaluation dataset split and size
$SPLIT = "test"
$VALIDATION_SAMPLES = 500
$EVAL_DENOISE_STEPS = 12
$EVAL_CLEANUP_EVERY = 8
$EVAL_OOM_FALLBACK_STEPS = 6
# Optional subset JSON for fixed 500 pairs (overrides random sampling when set)

# Runner options
$SKIP_IF_EXISTS = $true
$DRY_RUN = $false
$PREPARE_PROMPTS = $false
$OPENAI_PROMPT_MODEL = "gpt-4o-mini"

$PRETRAINED_MODEL = "D:\models\IMAGDressing"
$VAE_PATH         = "D:\models\IMAGDressing\sd-vae-ft-mse"
$ADAPTER_PATH     = "D:\models\IP-Adapter\models\ip-adapter-plus_sd15.bin"
$IMAGE_ENCODER    = "D:\models\IMAGDressing\image_encoder"

function Assert-Path([string]$Path, [string]$Label, [string]$PathType = "Any") {
    $exists = $false
    if ($PathType -eq "Leaf") {
        $exists = Test-Path -Path $Path -PathType Leaf
    } elseif ($PathType -eq "Container") {
        $exists = Test-Path -Path $Path -PathType Container
    } else {
        $exists = Test-Path -Path $Path
    }
    if (-not $exists) {
        Write-Error "Missing ${Label}: ${Path}"
        exit 1
    }
}

function Test-RunHasEvalMetrics([string]$OutputRoot, [string]$SMode, [string]$MetricsFile = "eval_metrics.json") {
    $runDir = Join-Path $OutputRoot $SMode
    $metricsPath = Join-Path $runDir $MetricsFile
    return (Test-Path -Path $metricsPath -PathType Leaf)
}

Assert-Path $PROJECT_ROOT "PROJECT_ROOT" "Container"
Assert-Path $PRETRAINED_MODEL "pretrained model" "Container"
Assert-Path $VAE_PATH "pretrained VAE" "Container"
Assert-Path $ADAPTER_PATH "adapter model" "Leaf"
Assert-Path $IMAGE_ENCODER "image encoder" "Container"
Assert-Path $DRESSCODE_ROOT "DressCode root" "Container"
Assert-Path $PROMPTS_CACHE "prompts cache" "Leaf"
Assert-Path $TEST_SET_JSON "test subset JSON" "Leaf"
Assert-Path $TRAIN_SCRIPT "train_extractor_DC.py" "Leaf"
Assert-Path $RUNNER_SCRIPT "run_ablation_extractor_DC.py" "Leaf"
Assert-Path $CHECKPOINT "checkpoint" "Leaf"

if (-not (Test-Path $OUTPUT_ROOT)) {
    New-Item -ItemType Directory -Path $OUTPUT_ROOT -Force | Out-Null
}

if ($ONLY_RUN_INCOMPLETE) {
    $pendingModes = @()
    foreach ($mode in $S_MODES) {
        if (Test-RunHasEvalMetrics -OutputRoot $OUTPUT_ROOT -SMode $mode) {
            Write-Host "Skip completed mode: $mode (eval_metrics.json exists)" -ForegroundColor Gray
            continue
        }
        $pendingModes += $mode
    }
    $S_MODES = $pendingModes
}

if (-not $S_MODES -or $S_MODES.Count -eq 0) {
    Write-Host "All requested S modes are already completed. Nothing to run." -ForegroundColor Green
    exit 0
}

$S_MODES_STR = ($S_MODES -join ",")

# ===== Args forwarded to train_extractor_DC.py (after `--`) =====
$train_args = @(
    "--pretrained_model_name_or_path", $PRETRAINED_MODEL,
    "--pretrained_vae_model_path", $VAE_PATH,
    "--pretrained_adapter_model_path", $ADAPTER_PATH,
    "--image_encoder_path", $IMAGE_ENCODER,

    "--dresscode_root", $DRESSCODE_ROOT,
    "--dresscode_category", $CATEGORY,
    "--dresscode_test_order", $TEST_ORDER,
    "--split", $SPLIT,

    "--prompts_cache", $PROMPTS_CACHE,
    "--eval_only",
    "--checkpoint", $CHECKPOINT,
    "--test_set_json", $TEST_SET_JSON,
    "--save_val_pairs",
    "--validation_samples", $VALIDATION_SAMPLES,
    "--low_vram",
    "--eval_denoise_steps", $EVAL_DENOISE_STEPS,
    "--eval_cleanup_every", $EVAL_CLEANUP_EVERY,
    "--eval_oom_fallback_steps", $EVAL_OOM_FALLBACK_STEPS,
    "--eval_skip_existing_pairs",

    # Optional: prompt mode for eval.
    "--prompt_mode", "cache_or_fallback",

    "--mixed_precision", "bf16",
    "--seed", 42
)

if ($PREPARE_PROMPTS) {
    if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY) -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
        Write-Host "ERROR: Missing API key. Set `$env:DASHSCOPE_API_KEY or `$env:OPENAI_API_KEY in this PowerShell session before running." -ForegroundColor Red
        exit 1
    }
    Write-Host "Preparing prompts cache (auto_prepare_prompts)..." -ForegroundColor Cyan
    $prep_args = @(
        "--pretrained_model_name_or_path", $PRETRAINED_MODEL,
        "--image_encoder_path", $IMAGE_ENCODER,
        "--dresscode_root", $DRESSCODE_ROOT,
        "--dresscode_category", $CATEGORY,
        "--dresscode_test_order", $TEST_ORDER,
        "--split", $SPLIT,
        "--prompts_cache", $PROMPTS_CACHE,
        "--prompt_mode", "fallback_only",
        "--auto_prepare_prompts",
        "--prepare_prompts_only",
        "--openai_prompt_model", $OPENAI_PROMPT_MODEL
    )
    & $PYTHON_EXE @PYTHON_PREFIX $TRAIN_SCRIPT @prep_args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: prompt cache preparation failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ===== Args for run_ablation_extractor_DC.py =====
$runner_args = @(
    "--train_script", $TRAIN_SCRIPT,
    "--eval_only",
    "--checkpoint", $CHECKPOINT,
    "--output_root", $OUTPUT_ROOT,
    "--s_modes", $S_MODES_STR,
    "--no_p_grid",
    "--compute_visual_metrics",
    "--metrics_out", "eval_metrics.json"
)

if ($SKIP_IF_EXISTS) { $runner_args += "--skip_if_exists" }
if ($DRY_RUN) { $runner_args += "--dry_run" }
$runner_args += "--test_set_json"
$runner_args += $TEST_SET_JSON

Write-Host "Running ablations (eval-only)..." -ForegroundColor Cyan
Write-Host "  checkpoint: $CHECKPOINT" -ForegroundColor Gray
Write-Host "  output_root: $OUTPUT_ROOT" -ForegroundColor Gray
Write-Host "  grid: S=[$S_MODES_STR] (no P grid)" -ForegroundColor Gray
Write-Host "  eval_denoise_steps: $EVAL_DENOISE_STEPS (low_vram=on)" -ForegroundColor Gray
Write-Host ""

# Important: the `--` separator is required so the remaining args are passed through to train_extractor_DC.py
& $PYTHON_EXE @PYTHON_PREFIX $RUNNER_SCRIPT @runner_args -- @train_args

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: ablation runner failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "DONE. Check each run dir under: $OUTPUT_ROOT" -ForegroundColor Green
Write-Host "Expect: Sx\\eval_metrics.json (S0/S1/S2)" -ForegroundColor Green
