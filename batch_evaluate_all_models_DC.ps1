# batch_evaluate_all_models_DC.ps1
# Changes: ensure prompt cache completeness for subset_json and enforce cache_only
# with fixed DashScope OpenAI-compat configuration.
# Batch evaluate extractor_DC checkpoints on DressCode.

conda activate IMAGDressing

$ErrorActionPreference = "Continue"

# Fixed DashScope OpenAI-compat configuration (do not hardcode API key).
$env:OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPENAI_PROMPT_MODEL = "qwen-vl-plus-latest"
$env:DASHSCOPE_API_KEY = "sk-bc487c556e864cddb9f4000d5b51a61a"
$env:DRESSCODE_PROMPT_PROVIDER = "dashscope_openai_compat"

# Paths
$RepoRoot = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing"
$DressCodeRoot = "E:\BaiduNetdiskDownload\DressCode"
$CheckpointDir = Join-Path $RepoRoot "outputs\extractor_DC"
$EvalRoot = Join-Path $RepoRoot "evaluation_results_DC"
$AssetsDir = Join-Path $EvalRoot "assets"
$SubsetJson = Join-Path $AssetsDir "test_subset_500.json"
$PromptsCache = $env:DRESSCODE_PROMPTS_CACHE
if ([string]::IsNullOrWhiteSpace($PromptsCache)) {
    $PromptsCache = "E:\BaiduNetdiskDownload\DressCode\prompts_dresscode_cache.jsonl"
}

$PretrainedModel = Join-Path $RepoRoot "models\IMAGDressing"
$ImageEncoder = Join-Path $RepoRoot "models\IMAGDressing\image_encoder"
$VaePath = Join-Path $RepoRoot "models\IMAGDressing\sd-vae-ft-mse"

# Eval settings
$SubsetSize = 500
$SubsetSeed = 42
$Seed = 42
$Split = "test"
$Steps = 80
$InitMode = "from_noisy_gt"
$ReconstructT = 600
$MinEffectiveSteps = 10
$Scheduler = "dpmpp_2m_karras"
$TimestepSpacing = "trailing"
$Eta = 0.0
$DecodeFP32 = $true
$Guidance = 1.0
$ImageCfgScale = 1.0
$VaeDeterministic = $true
$RefinePass = $false
$RefineT = 150
$RefineSteps = 50
$SkipGeneration = $false
$Device = "cuda"

# Prompt settings
$LogFirstNPrompts = 10
$PromptMode = "cache_only"
$PromptProvider = $env:DRESSCODE_PROMPT_PROVIDER
$PromptApiBase = $env:OPENAI_BASE_URL
$PromptModel = $env:OPENAI_PROMPT_MODEL
$PromptApiKeyEnv = "DASHSCOPE_API_KEY"
$PromptTemperature = 0
$PromptMaxTokens = 80
$PromptImageMode = "cloth_only"
$PromptConcurrency = 8
$ActivePromptsCache = $PromptsCache

function Get-MissingCountFromOutput {
    param(
        [string[]]$OutputLines
    )
    $missingCount = $null
    foreach ($line in $OutputLines) {
        if ($line -match "missing\s*[:=]\s*(\d+)") {
            $missingCount = [int]$Matches[1]
        }
    }
    return $missingCount
}

function Ensure-PromptCacheComplete {
    param(
        [string]$DressCodeRoot,
        [string]$SubsetJson,
        [string]$PromptsCache,
        [string]$AssetsDir
    )

    $CheckArgs = @(
        "--check_prompts_only",
        "--prompt_mode", "cache_only",
        "--dresscode_root", $DressCodeRoot,
        "--subset_json", $SubsetJson,
        "--prompts_cache", $PromptsCache,
        "--output_dir", $AssetsDir,
        "--prompt_provider", $PromptProvider,
        "--prompt_api_base", $PromptApiBase,
        "--prompt_model", $PromptModel,
        "--prompt_api_key_env", $PromptApiKeyEnv,
        "--prompt_temperature", $PromptTemperature,
        "--prompt_max_tokens", $PromptMaxTokens,
        "--prompt_image_mode", $PromptImageMode
    )

    $checkOutput = & python -B .\evaluate_extractor_DC.py @CheckArgs 2>&1
    $checkExit = $LASTEXITCODE
    $checkOutput | ForEach-Object { Write-Host $_ }
    $missingCount = Get-MissingCountFromOutput -OutputLines $checkOutput
    $needsBuild = $false
    if ($checkExit -ne 0) { $needsBuild = $true }
    if ($missingCount -ne $null -and $missingCount -gt 0) { $needsBuild = $true }

    if ($needsBuild) {
        Write-Host "[WARN] prompts_cache incomplete; building missing prompts." -ForegroundColor Yellow

        $BuildArgs = @(
            "--build_prompts_only",
            "--dresscode_root", $DressCodeRoot,
            "--subset_json", $SubsetJson,
            "--prompts_cache", $PromptsCache,
            "--output_dir", $AssetsDir,
            "--prompt_provider", $PromptProvider,
            "--prompt_api_base", $PromptApiBase,
            "--prompt_model", $PromptModel,
            "--prompt_api_key_env", $PromptApiKeyEnv,
            "--prompt_temperature", $PromptTemperature,
            "--prompt_max_tokens", $PromptMaxTokens,
            "--prompt_image_mode", $PromptImageMode,
            "--prompt_concurrency", $PromptConcurrency
        )

        $buildOutput = & python -B .\evaluate_extractor_DC.py @BuildArgs 2>&1
        $buildExit = $LASTEXITCODE
        $buildOutput | ForEach-Object { Write-Host $_ }
        if ($buildExit -ne 0) {
            throw "build_prompts_only failed. Aborting before evaluation."
        }

        Write-Host "[INFO] Re-checking prompts_cache after build..." -ForegroundColor Yellow
        $checkOutput = & python -B .\evaluate_extractor_DC.py @CheckArgs 2>&1
        $checkExit = $LASTEXITCODE
        $checkOutput | ForEach-Object { Write-Host $_ }
        $missingCount = Get-MissingCountFromOutput -OutputLines $checkOutput
        if ($checkExit -ne 0 -or ($missingCount -ne $null -and $missingCount -gt 0)) {
            throw "prompts_cache still missing after build_prompts_only. Aborting before evaluation."
        }
    }

    if ($missingCount -eq 0) {
        Write-Host "[OK] prompts_cache complete for subset_json." -ForegroundColor Green
    } elseif ($missingCount -eq $null -and -not $needsBuild) {
        Write-Host "[OK] prompts_cache check completed (missing count not reported)." -ForegroundColor Green
    }
}

New-Item -ItemType Directory -Force -Path $AssetsDir | Out-Null
New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null

if (-not (Test-Path $DressCodeRoot)) {
    Write-Host "[ERROR] DressCode root not found: $DressCodeRoot" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $CheckpointDir)) {
    Write-Host "[ERROR] Checkpoint directory not found: $CheckpointDir" -ForegroundColor Red
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Preparing subset_json" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Test-Path $SubsetJson)) {
    $subsetArgs = @(
        "--dresscode_root", $DressCodeRoot,
        "--split", $Split,
        "--build_subset_only",
        "--subset_json", $SubsetJson,
        "--output_dir", $AssetsDir,
        "--subset_size", $SubsetSize,
        "--subset_seed", $SubsetSeed,
        "--prompts_cache", $PromptsCache,
        "--prompt_mode", $PromptMode
    )
    python -B .\evaluate_extractor_DC.py @subsetArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] build_subset_only failed." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[OK] subset_json already exists: $SubsetJson" -ForegroundColor Green
}

if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) {
    Write-Host "[ERROR] DASHSCOPE_API_KEY not set. Set the env var before running." -ForegroundColor Red
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Prompt Cache Precheck" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Ensure-PromptCacheComplete -DressCodeRoot $DressCodeRoot -SubsetJson $SubsetJson -PromptsCache $PromptsCache -AssetsDir $AssetsDir

$Checkpoints = Get-ChildItem -Path $CheckpointDir -Filter *.pt | Sort-Object Name
if ($Checkpoints.Count -eq 0) {
    Write-Host "[ERROR] No checkpoint files found under $CheckpointDir" -ForegroundColor Red
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Batch Evaluation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Checkpoints: $($Checkpoints.Count)" -ForegroundColor Yellow
Write-Host "Subset JSON: $SubsetJson" -ForegroundColor Yellow
Write-Host "Prompts cache: $ActivePromptsCache" -ForegroundColor Yellow
Write-Host "Scheduler: $Scheduler | Spacing: $TimestepSpacing | Eta: $Eta | Decode FP32: $DecodeFP32" -ForegroundColor Yellow
Write-Host "Steps: $Steps | Guidance: $Guidance | Image CFG: $ImageCfgScale | Init: $InitMode" -ForegroundColor Yellow
Write-Host "Reconstruct T: $ReconstructT | Min steps: $MinEffectiveSteps | Refine: $RefinePass" -ForegroundColor Yellow
Write-Host "Prompt mode: $PromptMode | VAE deterministic: $VaeDeterministic" -ForegroundColor Yellow
Write-Host "Skip generation: $SkipGeneration" -ForegroundColor Yellow
Write-Host ""

$Success = @()
$Failed = @()

foreach ($Ckpt in $Checkpoints) {
    $CkptName = $Ckpt.BaseName
    $OutDir = Join-Path $EvalRoot $CkptName
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Evaluating: $CkptName" -ForegroundColor Green
    Write-Host "Checkpoint: $($Ckpt.FullName)" -ForegroundColor Yellow
    Write-Host "Output: $OutDir" -ForegroundColor Yellow

    $EvalArgs = @(
        "--dresscode_root", $DressCodeRoot,
        "--checkpoint", $Ckpt.FullName,
        "--output_dir", $OutDir,
        "--subset_json", $SubsetJson,
        "--num_inference_steps", $Steps,
        "--init_mode", $InitMode,
        "--reconstruct_t", $ReconstructT,
        "--min_effective_steps", $MinEffectiveSteps,
        "--scheduler", $Scheduler,
        "--timestep_spacing", $TimestepSpacing,
        "--eta", $Eta,
        "--guidance_scale", $Guidance,
        "--image_cfg_scale", $ImageCfgScale,
        "--seed", $Seed,
        "--device", $Device,
        "--pretrained_model", $PretrainedModel,
        "--image_encoder", $ImageEncoder,
        "--vae_path", $VaePath,
        "--prompt_mode", $PromptMode,
        "--prompt_provider", $PromptProvider,
        "--prompt_api_base", $PromptApiBase,
        "--prompt_model", $PromptModel,
        "--prompt_api_key_env", $PromptApiKeyEnv,
        "--log_first_n_prompts", $LogFirstNPrompts
    )

    $EvalArgs += "--prompts_cache"; $EvalArgs += "$ActivePromptsCache"
    $EvalArgs += "--strict_empty_prompt"
    if ($VaeDeterministic) { $EvalArgs += "--vae_deterministic" }
    if ($DecodeFP32) { $EvalArgs += "--decode_fp32" } else { $EvalArgs += "--no-decode_fp32" }
    if ($RefinePass) {
        $EvalArgs += "--refine_pass"
        $EvalArgs += "--refine_t"; $EvalArgs += $RefineT
        $EvalArgs += "--refine_steps"; $EvalArgs += $RefineSteps
    }
    if ($SkipGeneration) { $EvalArgs += "--skip_generation" }

    python -B .\evaluate_extractor_DC.py @EvalArgs

    if ($LASTEXITCODE -eq 0) {
        $Success += $CkptName
        Write-Host "[OK] $CkptName evaluation completed" -ForegroundColor Green
    } else {
        $Failed += $CkptName
        Write-Host "[ERROR] $CkptName evaluation failed" -ForegroundColor Red
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Batch evaluation complete" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Successful checkpoints: $($Success.Count)" -ForegroundColor Green
if ($Success.Count -gt 0) { Write-Host ($Success -join ", ") }
Write-Host "Failed checkpoints: $($Failed.Count)" -ForegroundColor Yellow
if ($Failed.Count -gt 0) { Write-Host ($Failed -join ", ") }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "Summary of Results" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

$SummaryResults = @()
foreach ($Checkpoint in $Checkpoints) {
    $CheckpointName = $Checkpoint.BaseName
    $ResultsJson = Join-Path (Join-Path $EvalRoot $CheckpointName) "evaluation_results.json"

    if (Test-Path $ResultsJson) {
        $Results = Get-Content $ResultsJson | ConvertFrom-Json

        $StepsLabel = "N/A"
        if ($CheckpointName -match "model_(\d+)") {
            $StepsLabel = [int]$Matches[1]
        } elseif ($CheckpointName -match "milestone_(\d+)") {
            $StepsLabel = [int]$Matches[1]
        } elseif ($CheckpointName -eq "best") {
            $StepsLabel = "Best"
        } elseif ($CheckpointName -eq "last") {
            $StepsLabel = "Last"
        }

        $SummaryResults += [PSCustomObject]@{
            Checkpoint = $CheckpointName
            Steps = $StepsLabel
            Samples = $Results.num_successful
            PSNR = if ($Results.PSNR) { [math]::Round($Results.PSNR, 2) } else { "N/A" }
            SSIM = if ($Results.SSIM) { [math]::Round($Results.SSIM, 4) } else { "N/A" }
            LPIPS = if ($Results.LPIPS) { [math]::Round($Results.LPIPS, 4) } else { "N/A" }
            DISTS = if ($Results.DISTS) { [math]::Round($Results.DISTS, 4) } else { "N/A" }
            FID = if ($Results.FID) { [math]::Round($Results.FID, 2) } else { "N/A" }
            KIDx1000 = if ($Results.KID_x1000) { [math]::Round($Results.KID_x1000, 2) } else { "N/A" }
        }
    }
}

$SummaryResults = $SummaryResults | Sort-Object {
    if ($_.Steps -is [int]) { $_.Steps }
    elseif ($_.Steps -eq "Best") { [int]::MaxValue - 1 }
    elseif ($_.Steps -eq "Last") { [int]::MaxValue }
    else { 0 }
}

Write-Host "Evaluation Metrics Table:" -ForegroundColor Cyan
Write-Host ""
$SummaryResults | Format-Table -AutoSize

$SummaryPath = Join-Path $EvalRoot "summary.json"
$SummaryResults | ConvertTo-Json | Set-Content $SummaryPath -Encoding UTF8
Write-Host "Summary JSON saved to: $SummaryPath" -ForegroundColor Green

$SummaryCsvPath = Join-Path $EvalRoot "summary.csv"
$SummaryResults | Export-Csv -Path $SummaryCsvPath -NoTypeInformation -Encoding UTF8
Write-Host "Summary CSV saved to: $SummaryCsvPath" -ForegroundColor Green
