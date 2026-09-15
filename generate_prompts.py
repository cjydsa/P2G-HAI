"""
批量生成 VTON-HD 数据集的 text prompt

支持多种 API：
- OpenAI (GPT-4V, GPT-4o)
- Anthropic Claude (Claude 3.5 Sonnet)
- 本地 BLIP2/LLaVA 模型

用法:
    # 使用 OpenAI API
    python generate_prompts.py --api openai --model gpt-4o --api_key YOUR_KEY

    # 使用 Claude API
    python generate_prompts.py --api claude --model claude-3-5-sonnet-20241022 --api_key YOUR_KEY

    # 使用本地 BLIP2 模型
    python generate_prompts.py --api local --model blip2

    # 断点续传（跳过已生成的）
    python generate_prompts.py --api openai --resume
"""

import argparse
import json
import os
import base64
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io


# =============================================================================
# API 调用函数
# =============================================================================

def encode_image_to_base64(image_path, max_size=1024):
    """将图片编码为 base64（用于 API 调用）"""
    img = Image.open(image_path)

    # 压缩大图以减少 API 成本
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    # 转为 JPEG base64
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def call_openai_api(image_path, api_key, model="gpt-4o", prompt_template=None):
    """
    Call OpenAI GPT-4V/GPT-4o API to generate a detailed description of the clothing.

    Args:
        image_path: Path to the image file
        api_key: OpenAI API key
        model: Model name (default: gpt-4o)
        prompt_template: Custom prompt template

    Returns:
        str: Generated description text, None if failed
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    # Default prompt template if not provided
    if prompt_template is None:
        prompt_template = (
            "Describe the person's clothing in detail, including the style, color, patterns, and overall aesthetic. "
            "Focus on the clothing's design, fit, and any distinguishing features. "
            "Use concise, descriptive language suitable for generating an image prompt. "
            "Format: 'A [gender] wearing [clothing description], [pose/style details]'"
        )

    try:
        # Convert image to base64 format
        base64_image = encode_image_to_base64(image_path)

        # Create OpenAI client
        client = OpenAI(api_key=api_key)

        # Call OpenAI API to generate the description
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_template},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=150,
            temperature=0.7,
        )

        # Return the generated prompt
        prompt = response.choices[0].message.content.strip()
        return prompt

    except Exception as e:
        print(f"\nError calling OpenAI API: {e}")
        return None


def call_claude_api(image_path, api_key, model="claude-3-5-sonnet-20241022", prompt_template=None):
    """
    Call Claude API to generate a detailed description of the clothing.

    Args:
        image_path: Path to the image file
        api_key: Anthropic API key
        model: Model name (default: claude-3-5-sonnet-20241022)
        prompt_template: Custom prompt template

    Returns:
        str: Generated description text, None if failed
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("Please install anthropic: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    # Default prompt template if not provided
    if prompt_template is None:
        prompt_template = (
            "Describe the person's clothing in detail, focusing on the style, colors, patterns, and overall aesthetic. "
            "Include details about the fit, material, and design of the clothing. "
            "Format: 'A [gender] wearing [clothing description], [pose/style details]'"
        )

    # Read and convert image to base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Detect image format
    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.gif': 'image/gif'
    }
    media_type = media_type_map.get(ext, 'image/jpeg')

    try:
        # Call Claude API to generate the description
        message = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt_template
                        }
                    ],
                }
            ],
        )

        # Return the generated prompt
        prompt = message.content[0].text.strip()
        return prompt

    except Exception as e:
        print(f"\nError calling Claude API: {e}")
        return None


def call_local_blip2(image_path, model_name="blip2", device="cuda"):
    """调用本地 BLIP2 模型"""
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
    except ImportError:
        raise ImportError("请安装 transformers 和 torch: pip install transformers torch")

    # 加载模型（只在第一次调用时加载）
    if not hasattr(call_local_blip2, "model"):
        print("Loading BLIP2 model...")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.to(device)
        call_local_blip2.processor = processor
        call_local_blip2.model = model
        print("BLIP2 model loaded!")

    # 处理图片
    image = Image.open(image_path).convert("RGB")

    # 生成描述
    inputs = call_local_blip2.processor(image, return_tensors="pt").to(
        call_local_blip2.model.device,
        call_local_blip2.model.dtype
    )

    generated_ids = call_local_blip2.model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        temperature=0.7,
    )

    caption = call_local_blip2.processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    return caption


def call_local_llava(image_path, model_name="llava-1.5-7b", device="cuda"):
    """调用本地 LLaVA 模型"""
    try:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        import torch
    except ImportError:
        raise ImportError("请安装 transformers 和 torch")

    # 加载模型（只在第一次调用时加载）
    if not hasattr(call_local_llava, "model"):
        print("Loading LLaVA model...")
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        model.to(device)
        call_local_llava.processor = processor
        call_local_llava.model = model
        print("LLaVA model loaded!")

    # 处理图片
    image = Image.open(image_path).convert("RGB")

    prompt_text = (
        "USER: <image>\nDescribe this person's appearance in detail. "
        "Focus on clothing, pose, and style.\nASSISTANT:"
    )

    inputs = call_local_llava.processor(prompt_text, image, return_tensors="pt").to(
        call_local_llava.model.device,
        call_local_llava.model.dtype
    )

    generated_ids = call_local_llava.model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )

    caption = call_local_llava.processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    return caption


# =============================================================================
# 主处理流程
# =============================================================================

def load_existing_cache(cache_path):
    """加载已有的 prompt 缓存（支持断点续传）"""
    if not os.path.exists(cache_path):
        return {}

    ext = os.path.splitext(cache_path)[1].lower()

    if ext == '.json':
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    elif ext == '.jsonl':
        cache = {}
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("key")
                if key:
                    if "prompt" in obj:
                        cache[key] = obj["prompt"]
                    elif "prompts" in obj:
                        cache[key] = obj["prompts"]
        return cache

    else:
        raise ValueError(f"Unsupported cache format: {ext}")


def save_cache(cache, cache_path, format="json"):
    """保存 prompt 缓存"""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    if format == "json":
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    elif format == "jsonl":
        with open(cache_path, 'w', encoding='utf-8') as f:
            for key, prompt in cache.items():
                obj = {"key": key, "prompt": prompt}
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(cache)} prompts to {cache_path}")


def make_pair_key(image_file, cloth_file):
    """生成 pair 的唯一 key（与 VTONHD.py 保持一致）"""
    return f"{image_file}|||{cloth_file}"


def generate_prompts_for_dataset(
    vtonhd_root,
    split,
    pairs_file,
    output_path,
    api_type="openai",
    api_key=None,
    model=None,
    resume=True,
    max_samples=None,
    prompt_template=None,
    save_interval=50,
    device="cuda",
):
    """
    批量生成数据集的 text prompts

    Args:
        vtonhd_root: VTON-HD 数据集根目录
        split: 数据集分割 (train/test)
        pairs_file: pairs.txt 文件路径
        output_path: 输出缓存文件路径 (.json 或 .jsonl)
        api_type: API 类型 (openai/claude/blip2/llava)
        api_key: API 密钥（本地模型不需要）
        model: 模型名称
        resume: 是否断点续传
        max_samples: 最大处理样本数（用于测试）
        prompt_template: 自定义提示词模板
        save_interval: 每处理多少样本保存一次
        device: 设备 (cuda/cpu)
    """

    # 加载已有缓存（断点续传）
    cache = load_existing_cache(output_path) if resume else {}
    print(f"Loaded {len(cache)} existing prompts from cache")

    # 查找 person 图片目录
    person_dir_candidates = [
        os.path.join(vtonhd_root, split, "image"),
        os.path.join(vtonhd_root, split, "images"),
        os.path.join(vtonhd_root, "image", split),
    ]
    person_dir = None
    for path in person_dir_candidates:
        if os.path.exists(path):
            person_dir = path
            break

    if person_dir is None:
        raise FileNotFoundError(f"Could not find person image directory in {vtonhd_root}")

    print(f"Person image directory: {person_dir}")

    # 读取 pairs.txt
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append({
                    "image_file": parts[0],
                    "cloth_file": parts[1],
                })

    print(f"Loaded {len(pairs)} pairs from {pairs_file}")

    # 限制处理数量（用于测试）
    if max_samples:
        pairs = pairs[:max_samples]
        print(f"Limited to {max_samples} samples for testing")

    # 选择 API 调用函数
    if api_type == "openai":
        if not api_key:
            raise ValueError("OpenAI API requires --api_key")
        api_func = lambda img_path: call_openai_api(img_path, api_key, model or "gpt-4o", prompt_template)

    elif api_type == "claude":
        if not api_key:
            raise ValueError("Claude API requires --api_key")
        api_func = lambda img_path: call_claude_api(img_path, api_key, model or "claude-3-5-sonnet-20241022", prompt_template)

    elif api_type == "blip2":
        api_func = lambda img_path: call_local_blip2(img_path, device=device)

    elif api_type == "llava":
        api_func = lambda img_path: call_local_llava(img_path, device=device)

    else:
        raise ValueError(f"Unsupported API type: {api_type}")

    # 批量生成 prompts
    skipped = 0
    generated = 0
    failed = 0

    with tqdm(total=len(pairs), desc="Generating prompts") as pbar:
        for i, pair in enumerate(pairs):
            image_file = pair["image_file"]
            cloth_file = pair["cloth_file"]

            # 生成 key
            key = make_pair_key(image_file, cloth_file)

            # 跳过已生成的
            if key in cache:
                skipped += 1
                pbar.update(1)
                pbar.set_postfix({"generated": generated, "skipped": skipped, "failed": failed})
                continue

            # 构造图片路径
            image_path = os.path.join(person_dir, image_file)

            if not os.path.exists(image_path):
                print(f"\nWARNING: Image not found: {image_path}")
                failed += 1
                pbar.update(1)
                continue

            # 调用 API 生成 prompt
            try:
                prompt = api_func(image_path)

                # 只存储有效的 prompt（非 None）
                if prompt:
                    cache[key] = prompt
                    generated += 1

                    # 定期保存（避免中断丢失进度）
                    if generated % save_interval == 0:
                        save_cache(cache, output_path, format="json")
                        print(f"\n✓ Progress saved: {generated} new prompts generated")
                else:
                    # API 返回 None（调用失败）
                    print(f"\nWARNING: Failed to generate prompt for {image_file}")
                    failed += 1

                # API 限流（避免被 rate limit）
                if api_type in ["openai", "claude"]:
                    time.sleep(0.5)  # 每次调用间隔 0.5 秒

            except Exception as e:
                print(f"\nERROR processing {image_file}: {e}")
                failed += 1

            pbar.update(1)
            pbar.set_postfix({"generated": generated, "skipped": skipped, "failed": failed})

    # 最终保存
    save_cache(cache, output_path, format="json")

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total pairs: {len(pairs)}")
    print(f"Generated: {generated}")
    print(f"Skipped (cached): {skipped}")
    print(f"Failed: {failed}")
    print(f"Final cache size: {len(cache)}")
    print(f"Output: {output_path}")
    print("=" * 60)


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="批量生成 VTON-HD 数据集的 text prompts")

    # 数据集参数
    parser.add_argument("--vtonhd_root", type=str, required=True,
                        help="VTON-HD 数据集根目录")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="数据集分割")
    parser.add_argument("--pairs_file", type=str, default=None,
                        help="pairs.txt 文件路径（默认自动检测）")

    # 输出参数
    parser.add_argument("--output", type=str, default="prompts_cache.json",
                        help="输出缓存文件路径 (.json)")

    # API 参数
    parser.add_argument("--api", type=str, required=True,
                        choices=["openai", "claude", "blip2", "llava"],
                        help="API 类型")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API 密钥（openai/claude 需要）")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称（如 gpt-4o, claude-3-5-sonnet-20241022）")
    parser.add_argument("--prompt_template", type=str, default=None,
                        help="自定义提示词模板")

    # 运行参数
    parser.add_argument("--resume", action="store_true",
                        help="断点续传（跳过已生成的）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数（用于测试）")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="每处理多少样本保存一次")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")

    args = parser.parse_args()

    # 自动检测 pairs_file
    if args.pairs_file is None:
        candidates = [
            os.path.join(args.vtonhd_root, f"{args.split}_pairs.txt"),
            os.path.join(args.vtonhd_root, args.split, "pairs.txt"),
            os.path.join(args.vtonhd_root, f"pairs_{args.split}.txt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                args.pairs_file = path
                break

        if args.pairs_file is None:
            raise FileNotFoundError(
                f"Could not find pairs file. Tried:\n" +
                "\n".join(f"  - {p}" for p in candidates)
            )

    # 开始生成
    generate_prompts_for_dataset(
        vtonhd_root=args.vtonhd_root,
        split=args.split,
        pairs_file=args.pairs_file,
        output_path=args.output,
        api_type=args.api,
        api_key=args.api_key,
        model=args.model,
        resume=args.resume,
        max_samples=args.max_samples,
        prompt_template=args.prompt_template,
        save_interval=args.save_interval,
        device=args.device,
    )


if __name__ == "__main__":
    main()