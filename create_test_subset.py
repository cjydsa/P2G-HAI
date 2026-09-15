#!/usr/bin/env python3
"""
从IGPair或VTON-HD数据集中创建测试子集
"""
import json
import random
import os
import argparse

def create_test_subset_from_json(input_json, output_json, num_samples=50, seed=42):
    """
    从输入JSON中随机选择num_samples个样本创建测试集

    Args:
        input_json: 输入JSON文件路径
        output_json: 输出JSON文件路径
        num_samples: 测试样本数量
        seed: 随机种子
    """
    print(f"加载数据集: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总样本数: {len(data)}")

    # 设置随机种子以保证可复现
    random.seed(seed)

    # 随机选择样本
    if len(data) <= num_samples:
        test_subset = data
        print(f"警告: 数据集样本数({len(data)})少于请求数量({num_samples}), 使用全部样本")
    else:
        test_subset = random.sample(data, num_samples)
        print(f"随机选择 {num_samples} 个样本")

    # 保存测试子集
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(test_subset, f, indent=2, ensure_ascii=False)

    print(f"[OK] 测试子集已保存到: {output_json}")
    print(f"  样本数量: {len(test_subset)}")

    # 显示一些统计信息
    print("\n样本统计:")
    print(f"  前3个样本:")
    for i, item in enumerate(test_subset[:3]):
        print(f"    {i+1}. Person: {item.get('image_file', 'N/A')}")
        print(f"       Cloth: {item.get('cloth_file', 'N/A')}")


def create_test_subset_from_vtonhd(vtonhd_root, split, output_json, num_samples=500, seed=42):
    """
    从VTON-HD数据集创建测试子集（支持same_name模式）

    Args:
        vtonhd_root: VTON-HD数据集根目录
        split: 数据集分割 (train/test)
        output_json: 输出JSON文件路径
        num_samples: 测试样本数量
        seed: 随机种子
    """
    print(f"创建VTON-HD测试子集")
    print(f"  数据集根目录: {vtonhd_root}")
    print(f"  分割: {split}")
    print(f"  样本数量: {num_samples}")

    # 尝试使用same_name模式（从image目录扫描）
    image_dir_candidates = [
        os.path.join(vtonhd_root, split, "image"),
        os.path.join(vtonhd_root, split, "images"),
    ]

    image_dir = None
    for candidate in image_dir_candidates:
        if os.path.exists(candidate):
            image_dir = candidate
            break

    if not image_dir:
        raise FileNotFoundError(f"未找到image目录。尝试过: {image_dir_candidates}")

    print(f"使用image目录: {image_dir}")

    # 扫描目录获取所有图片文件
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}
    all_files = []
    for fname in os.listdir(image_dir):
        if any(fname.endswith(ext) for ext in image_exts):
            all_files.append(fname)

    print(f"找到 {len(all_files)} 个图片文件")

    # 创建same_name配对（包含完整的相对路径）
    data = []
    for fname in sorted(all_files):
        data.append({
            "image_file": f"{split}/image/{fname}",
            "cloth_file": f"{split}/cloth/{fname}",
        })

    print(f"总样本数: {len(data)}")

    # 设置随机种子
    random.seed(seed)

    # 随机选择样本
    if len(data) <= num_samples:
        test_subset = data
        print(f"警告: 数据集样本数({len(data)})少于请求数量({num_samples}), 使用全部样本")
    else:
        test_subset = random.sample(data, num_samples)
        print(f"随机选择 {num_samples} 个样本")

    # 保存测试子集
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(test_subset, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 测试子集已保存到: {output_json}")
    print(f"  样本数量: {len(test_subset)}")

    # 显示前几个样本
    print(f"\n前5个样本:")
    for i, item in enumerate(test_subset[:5]):
        print(f"  {i+1}. Person: {item['image_file']}")
        print(f"     Cloth: {item['cloth_file']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="创建测试子集（支持IGPair和VTON-HD）")

    # 数据集类型选择
    parser.add_argument("--dataset_type", type=str, default="vtonhd",
                       choices=["igpair", "vtonhd"],
                       help="数据集类型: igpair或vtonhd")

    # IGPair参数
    parser.add_argument("--input_json", type=str,
                       default="E:\\BaiduNetdiskDownload\\IGPair\\IGPair\\IGPair.json",
                       help="IGPair输入JSON文件路径")

    # VTON-HD参数
    parser.add_argument("--vtonhd_root", type=str,
                       default="E:\\BaiduNetdiskDownload\\VTON-HD\\zalando-hd-resized",
                       help="VTON-HD数据集根目录")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "test"],
                       help="VTON-HD数据集分割")

    # 通用参数
    parser.add_argument("--output_json", type=str,
                       default="test_subset_500_HD.json",
                       help="输出JSON文件路径")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="测试样本数量")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    if args.dataset_type == "igpair":
        create_test_subset_from_json(args.input_json, args.output_json, args.num_samples, args.seed)
    elif args.dataset_type == "vtonhd":
        create_test_subset_from_vtonhd(args.vtonhd_root, args.split, args.output_json, args.num_samples, args.seed)
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")
