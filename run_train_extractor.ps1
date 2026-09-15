# run_train_extractor.ps1

conda activate IMAGDressing
$Env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
# 如需更小碎片可改 64：
# $Env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:64,gc_threshold:0.6"

# 如你没装稳定的 xformers，建议加：
$Env:XFORMERS_DISABLED = "1"

$Env:TOKENIZERS_PARALLELISM="false"

& accelerate launch `
  --mixed_precision "bf16" `
  --gpu_ids 0 `
  --num_processes 1 `
  D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\train_extractor.py `
  --pretrained_model_name_or_path "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing" `
  --pretrained_vae_model_path "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing\sd-vae-ft-mse" `
  --pretrained_adapter_model_path "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing\models\ip-adapter-plus_sd15.bin" `
  --image_encoder_path "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\models\IMAGDressing\image_encoder" `
  --dataset_json_path "E:\BaiduNetdiskDownload\IGPair\IGPair\IGPair_clean.json" `
  --image_root_path "E:\BaiduNetdiskDownload\IGPair\" `
  --region_filter all `
  --train_batch_size 1 `
  --gradient_accumulation_steps 4 `
  --max_train_steps 100000 `
  --learning_rate 1e-4 `
  --weight_decay 0.01 `
  --lr_scheduler constant_with_warmup `
  --num_warmup_steps 1000 `
  --output_dir "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\outputs\extractor" `
  --noise_offset 0.05 `
  --snr_gamma 3.0 `
  --save_steps 5000 `
  --milestone_steps 200000 `
  --validation_steps 5000 `
  --validation_samples 24 `
  --val_preview_max 4 `
  --max_grad_norm 1.0 `
  --resume_from_checkpoint "D:\Users\14641\PycharmProjects\PythonProject\IMAGDressing\outputs\extractor\model_040000.pt"
