# batch_evaluate_all_models.ps1
# 批量评估 extractor_HD 目录中的所有 checkpoint

conda activate IMAGDressing

$ErrorActionPreference = "Continue"

# 定义参数
$TestSetJson = "test_subset_500.json"  # 使用500张图片的测试集
$ImageRoot = "E:\BaiduNetdiskDownload\VTON-HD\zalando-hd-resized"  # VTON-HD数据集
$CheckpointDir = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\outputs\extractor_HD"
$OutputBaseDir = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\evaluation_results_HD"
$SCHEDULER = "ddim"
$STEPS = 50
$RECONSTRUCT_T = 600
$GUIDANCE = 3.0
$MIN_EFFECTIVE_STEPS = 25
$VAE_DETERMINISTIC = $true
$DECODE_FP32 = $true
$ETA = 0.0               # 仅 DDIM 用
$PretrainedModel = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing"
$ImageEncoder = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing\image_encoder"
$VaePath = "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing\sd-vae-ft-mse"
$Seed = 42                    # Optional: set to an int for reproducible sampling
$ClothRoot = $ImageRoot       # Optional: separate cloth root; defaults to ImageRoot when not set
$PromptsCacheTrain = Join-Path $ImageRoot "prompts_vtonhd_train_paired.jsonl"
$PromptsCacheTest = Join-Path $ImageRoot "prompts_vtonhd_test_paired.jsonl"
$PromptsCache = $null
$CacheKeyMode = "basename_pair"
$InitMode = "from_noisy_gt"   # Optional: from_noise or from_noisy_gt
$StrictEmptyPrompt = $true   # fail fast if prompt empty
$LogFirstNPrompts = 10       # print first N prompts
$PromptFallback = "a high-quality photo of a garment on a plain background, studio lighting, detailed fabric texture"

# 选择 prompts cache（优先 test）
$TestSetName = [System.IO.Path]::GetFileName($TestSetJson)
if ($TestSetName.ToLower().Contains("test") -and (Test-Path $PromptsCacheTest)) {
    $PromptsCache = $PromptsCacheTest
} elseif (Test-Path $PromptsCacheTrain) {
    $PromptsCache = $PromptsCacheTrain
} else {
    $PromptsCache = $null
}

# 创建输出目录
New-Item -ItemType Directory -Force -Path $OutputBaseDir | Out-Null

# 检查测试集是否存在，不存在则创建
if (-not (Test-Path $TestSetJson)) {
    Write-Host "测试集 $TestSetJson 不存在，正在从原始数据创建..." -ForegroundColor Yellow
    Write-Host "请手动创建包含500个样本的测试集JSON文件，或使用 create_test_subset.py 脚本" -ForegroundColor Red
    # 可选：自动创建测试集
    # python create_test_subset.py --num_samples 500 --output_json $TestSetJson
}

# Prompts cache 命中率检查（命中为 0 则 warning）
if ($PromptsCache -and (Test-Path $PromptsCache) -and (Test-Path $TestSetJson)) {
    $TestDataRaw = Get-Content $TestSetJson | ConvertFrom-Json
    $TestItems = if ($TestDataRaw.PSObject.Properties.Name -contains "data") { $TestDataRaw.data } else { $TestDataRaw }
    $CacheKeys = @{}
    if ($PromptsCache.ToLower().EndsWith(".jsonl")) {
        Get-Content $PromptsCache | ForEach-Object {
            if ([string]::IsNullOrWhiteSpace($_)) { return }
            $m = [regex]::Match($_, '"key"\s*:\s*"([^"]+)"')
            if (-not $m.Success) { $m = [regex]::Match($_, '"pair_key"\s*:\s*"([^"]+)"') }
            if (-not $m.Success) { $m = [regex]::Match($_, '"id"\s*:\s*"([^"]+)"') }
            if ($m.Success) { $CacheKeys[$m.Groups[1].Value] = $true }
        }
    } else {
        $cacheObj = Get-Content $PromptsCache | ConvertFrom-Json
        if ($cacheObj -is [System.Collections.IDictionary]) {
            foreach ($k in $cacheObj.Keys) { $CacheKeys[$k] = $true }
        } else {
            foreach ($obj in $cacheObj) {
                $key = $obj.key
                if (-not $key) { $key = $obj.pair_key }
                if (-not $key) { $key = $obj.id }
                if ($key) { $CacheKeys[$key] = $true }
            }
        }
    }
    $HitCountRaw = 0
    foreach ($item in $TestItems) {
        $pairKey = "$($item.image_file)|||$($item.cloth_file)"
        if ($CacheKeys.ContainsKey($pairKey)) { $HitCountRaw++ }
    }
    $HitCountBase = 0
    foreach ($item in $TestItems) {
        $imgBase = Split-Path $item.image_file -Leaf
        $clothBase = Split-Path $item.cloth_file -Leaf
        $pairKeyBase = "$imgBase|||$clothBase"
        if ($CacheKeys.ContainsKey($pairKeyBase)) { $HitCountBase++ }
    }
    $HitCountEffective = if ($CacheKeyMode -eq "path_pair") { $HitCountRaw } elseif ($CacheKeyMode -eq "auto") { [Math]::Max($HitCountRaw, $HitCountBase) } else { $HitCountBase }
    if ($HitCountEffective -eq 0) {
        if ($HitCountBase -gt 0 -or $HitCountRaw -gt 0) {
            Write-Host "[WARN] Prompts cache key format mismatch. cache_key_mode=$CacheKeyMode" -ForegroundColor Yellow
        } else {
            Write-Host "[WARN] Prompts cache hit rate is 0: $PromptsCache" -ForegroundColor Yellow
        }
    }
}

# 获取所有要评估的checkpoints（所有 .pt 文件）
Write-Host "扫描checkpoint目录: $CheckpointDir" -ForegroundColor Cyan
$Checkpoints = Get-ChildItem "$CheckpointDir\*.pt" | Sort-Object Name

if ($Checkpoints.Count -eq 0) {
    Write-Host "错误: 未找到任何checkpoint文件！" -ForegroundColor Red
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Batch Evaluation of Extractor HD Models" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Found $($Checkpoints.Count) checkpoints to evaluate" -ForegroundColor Yellow
Write-Host "Test set: $TestSetJson" -ForegroundColor Yellow
Write-Host "Image root: $ImageRoot" -ForegroundColor Yellow
Write-Host "Scheduler: $SCHEDULER | Steps: $STEPS | Eta: $ETA | Decode FP32: $DECODE_FP32" -ForegroundColor Yellow
if ($PromptsCache -and (Test-Path $PromptsCache)) {
    Write-Host "Prompts cache: $PromptsCache" -ForegroundColor Yellow
} else {
    Write-Host "Prompts cache: (not used)" -ForegroundColor Yellow
}
Write-Host "Init mode: $InitMode | Guidance: $GUIDANCE | VAE deterministic: $VAE_DETERMINISTIC | Decode FP32: $DECODE_FP32" -ForegroundColor Yellow
Write-Host "Prompt cache key mode: $CacheKeyMode" -ForegroundColor Yellow
Write-Host ""

foreach ($Checkpoint in $Checkpoints) {
    $CheckpointName = $Checkpoint.BaseName
    $CheckpointPath = $Checkpoint.FullName
    $OutputDir = Join-Path $OutputBaseDir $CheckpointName

    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Evaluating: $CheckpointName" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Checkpoint: $CheckpointPath" -ForegroundColor Yellow
    Write-Host "Output: $OutputDir" -ForegroundColor Yellow
    Write-Host ""

    # 运行评估
    $EvalArgs = @(
        "--test_set_json", $TestSetJson,
        "--image_root", $ImageRoot,
        "--checkpoint", $CheckpointPath,
        "--output_dir", $OutputDir,
        "--num_inference_steps", $STEPS,
        "--scheduler", $SCHEDULER,
        "--device", "cuda",
        "--pretrained_model", $PretrainedModel,
        "--image_encoder", $ImageEncoder,
        "--vae_path", $VaePath,
        "--seed", $Seed,
        "--cloth_root", $ClothRoot
    )
    if ($PromptsCache -and (Test-Path $PromptsCache)) { $EvalArgs += "--prompts_cache"; $EvalArgs += $PromptsCache }
    if ($CacheKeyMode) { $EvalArgs += "--cache_key_mode"; $EvalArgs += $CacheKeyMode }
    if ($PromptFallback -ne "") { $EvalArgs += "--prompt_fallback"; $EvalArgs += $PromptFallback }
    if ($InitMode) { $EvalArgs += "--init_mode"; $EvalArgs += $InitMode }
    if ($null -ne $RECONSTRUCT_T) { $EvalArgs += "--reconstruct_t"; $EvalArgs += $RECONSTRUCT_T }
    if ($null -ne $GUIDANCE) { $EvalArgs += "--guidance_scale"; $EvalArgs += $GUIDANCE }
    if ($null -ne $MIN_EFFECTIVE_STEPS) { $EvalArgs += "--min_effective_steps"; $EvalArgs += $MIN_EFFECTIVE_STEPS }
    if ($null -ne $LogFirstNPrompts) { $EvalArgs += "--log_first_n_prompts"; $EvalArgs += $LogFirstNPrompts }
    if ($StrictEmptyPrompt) { $EvalArgs += "--strict_empty_prompt" }
    if ($VAE_DETERMINISTIC) { $EvalArgs += "--vae_deterministic" }
    if ($null -ne $ETA) { $EvalArgs += "--eta"; $EvalArgs += $ETA }
    if ($DECODE_FP32) { $EvalArgs += "--decode_fp32" }

    python evaluate_extractor.py @EvalArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] $CheckpointName evaluation completed" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] $CheckpointName evaluation failed" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host ""
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "All Evaluations Completed!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Results saved in: $OutputBaseDir" -ForegroundColor Green
Write-Host ""

# ===== 汇总结果并生成表格 =====
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "Summary of Results" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

$SummaryResults = @()
foreach ($Checkpoint in $Checkpoints) {
    $CheckpointName = $Checkpoint.BaseName
    $ResultsJson = Join-Path (Join-Path $OutputBaseDir $CheckpointName) "evaluation_results.json"

    if (Test-Path $ResultsJson) {
        $Results = Get-Content $ResultsJson | ConvertFrom-Json

        # 提取步数（从文件名中提取数字）
        $Steps = "N/A"
        if ($CheckpointName -match "model_(\d+)") {
            $Steps = [int]$Matches[1]
        } elseif ($CheckpointName -match "milestone_(\d+)") {
            $Steps = [int]$Matches[1]
        } elseif ($CheckpointName -eq "best") {
            $Steps = "Best"
        } elseif ($CheckpointName -eq "last") {
            $Steps = "Last"
        }

        $SummaryResults += [PSCustomObject]@{
            Checkpoint = $CheckpointName
            Steps = $Steps
            Samples = $Results.num_successful
            PSNR = if ($Results.PSNR) { [math]::Round($Results.PSNR, 2) } else { "N/A" }
            SSIM = if ($Results.SSIM) { [math]::Round($Results.SSIM, 4) } else { "N/A" }
            LPIPS = if ($Results.LPIPS) { [math]::Round($Results.LPIPS, 4) } else { "N/A" }
            FID = if ($Results.FID) { [math]::Round($Results.FID, 2) } else { "N/A" }
            "KID×1000" = if ($Results.KID_x1000) { [math]::Round($Results.KID_x1000, 2) } else { "N/A" }
        }
    }
}

# 按步数排序
$SummaryResults = $SummaryResults | Sort-Object {
    if ($_.Steps -is [int]) { $_.Steps }
    elseif ($_.Steps -eq "Best") { [int]::MaxValue - 1 }
    elseif ($_.Steps -eq "Last") { [int]::MaxValue }
    else { 0 }
}

# 显示表格
Write-Host "Evaluation Metrics Table:" -ForegroundColor Cyan
Write-Host ""
$SummaryResults | Format-Table -AutoSize

# 保存汇总结果（JSON格式）
$SummaryPath = Join-Path $OutputBaseDir "summary.json"
$SummaryResults | ConvertTo-Json | Set-Content $SummaryPath -Encoding UTF8
Write-Host "Summary JSON saved to: $SummaryPath" -ForegroundColor Green

# 保存汇总结果（CSV格式，便于Excel打开）
$SummaryCsvPath = Join-Path $OutputBaseDir "summary.csv"
$SummaryResults | Export-Csv -Path $SummaryCsvPath -NoTypeInformation -Encoding UTF8
Write-Host "Summary CSV saved to: $SummaryCsvPath" -ForegroundColor Green

# Display metrics explanation
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Metrics Explanation:" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pixel-level Metrics:" -ForegroundColor White
Write-Host "  PSNR: Peak Signal-to-Noise Ratio in dB" -ForegroundColor Gray
Write-Host "    Typical range 20-40 dB, higher is better" -ForegroundColor Gray
Write-Host ""
Write-Host "  SSIM: Structural Similarity Index" -ForegroundColor Gray
Write-Host "    Range 0-1, where 1 is identical, higher is better" -ForegroundColor Gray
Write-Host ""
Write-Host "  LPIPS: Learned Perceptual Image Patch Similarity" -ForegroundColor Gray
Write-Host "    Range 0-1, where 0 is identical, lower is better" -ForegroundColor Gray
Write-Host ""
Write-Host "Distribution Metrics:" -ForegroundColor White
Write-Host "  FID: Frechet Inception Distance" -ForegroundColor Gray
Write-Host "    Measures feature distribution distance, lower is better" -ForegroundColor Gray
Write-Host ""
Write-Host "  KID: Kernel Inception Distance scaled by 1000" -ForegroundColor Gray
Write-Host "    More robust than FID, kernel-based distance, lower is better" -ForegroundColor Gray
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
