#!/usr/bin/env python3
"""
标准 FID 和 KID 实现
严格按照定义实现:
- FID: Fréchet Inception Distance
- KID: Kernel Inception Distance (MMD^2 with polynomial kernel)
"""
# 设置环境变量避免OpenMP冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')


class InceptionV3FeatureExtractor:
    """
    标准 Inception V3 特征提取器
    提取 pool3 层的 2048 维特征
    """

    def __init__(self, device='cuda'):
        self.device = device

        try:
            # 尝试使用 pytorch-fid 的标准实现
            from pytorch_fid.inception import InceptionV3
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.model = InceptionV3([block_idx]).to(device)
            self.use_pytorch_fid = True
            print("使用 pytorch-fid 的标准 Inception V3")
        except ImportError:
            # 备用: torchvision 的 Inception V3
            print("使用 torchvision 的 Inception V3")
            from torchvision.models import inception_v3, Inception_V3_Weights
            model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)

            # 创建一个Hook来获取avgpool层的输出
            self.features = None
            def hook_fn(module, input, output):
                self.features = output

            # 注册hook在avgpool层
            model.avgpool.register_forward_hook(hook_fn)

            self.model = model
            self.model.to(device)
            self.use_pytorch_fid = False

        self.model.eval()

        # 标准 Inception V3 预处理
        # 输入应该是 [0, 1] 范围，然后归一化到 [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def extract_features(self, image_paths, batch_size=32):
        """
        提取图像的 Inception 特征

        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小

        Returns:
            features: numpy array of shape (N, 2048)
        """
        features_list = []

        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_tensors = []

                for img_path in batch_paths:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)

                batch = torch.stack(batch_tensors).to(self.device)

                # 前向传播获取特征
                if self.use_pytorch_fid:
                    features = self.model(batch)[0]
                else:
                    # torchvision Inception V3 - 通过hook获取avgpool输出
                    _ = self.model(batch)
                    features = self.features

                # 展平到 2048 维
                if len(features.shape) > 2:
                    features = features.squeeze(-1).squeeze(-1)

                features_list.append(features.cpu().numpy())

        # 合并所有批次
        all_features = np.concatenate(features_list, axis=0)

        return all_features


def calculate_fid_standard(features_real, features_generated):
    """
    标准 FID 计算

    FID = ||mu_r - mu_g||^2 + Trace(Sigma_r + Sigma_g - 2*(Sigma_r*Sigma_g)^(1/2))

    Args:
        features_real: 真实图像特征 (m, 2048)
        features_generated: 生成图像特征 (n, 2048)

    Returns:
        fid: FID score (float)
    """
    # 计算均值
    mu_real = np.mean(features_real, axis=0)
    mu_gen = np.mean(features_generated, axis=0)

    # 计算协方差矩阵 (使用无偏估计)
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_gen = np.cov(features_generated, rowvar=False)

    # 计算均值差的平方范数
    diff = mu_real - mu_gen
    mean_diff_squared = np.sum(diff ** 2)

    # 计算协方差项: Trace(Sigma_r + Sigma_g - 2*(Sigma_r*Sigma_g)^(1/2))
    # 首先计算 (Sigma_r * Sigma_g)^(1/2)
    covmean = linalg.sqrtm(sigma_real.dot(sigma_gen))

    # 检查数值稳定性
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"协方差矩阵平方根存在较大虚部: {m}")
        covmean = covmean.real

    # 计算 trace
    trace_sum = np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)

    # FID 公式
    fid = mean_diff_squared + trace_sum

    return fid


def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    """
    多项式核矩阵计算

    k(a, b) = ((a·b)/d + 1)^3

    Args:
        X: (m, d) features
        Y: (n, d) features
        degree: 多项式度数，默认3
        gamma: 缩放因子，默认 1/d
        coef0: 常数项，默认1

    Returns:
        K: (m, n) kernel matrix
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # 计算点积矩阵
    dot_product = X @ Y.T

    # 多项式核: (gamma * dot_product + coef0)^degree
    kernel_matrix = (gamma * dot_product + coef0) ** degree

    return kernel_matrix


def calculate_kid_standard(features_real, features_generated, subset_size=None, num_subsets=100):
    """
    标准 KID 计算 (无偏 MMD^2 估计)

    KID = term_xx + term_yy - 2*term_xy

    其中:
    - term_xx = (1/(m(m-1))) * sum_{i!=j} k(x_i, x_j)
    - term_yy = (1/(n(n-1))) * sum_{i!=j} k(y_i, y_j)
    - term_xy = (1/(mn)) * sum_{i,j} k(x_i, y_j)
    - k(a,b) = ((a·b)/d + 1)^3

    Args:
        features_real: 真实图像特征 (m, 2048)
        features_generated: 生成图像特征 (n, 2048)
        subset_size: 子集大小 (用于大数据集)，None表示使用全部数据
        num_subsets: 子集数量 (用于估计方差)

    Returns:
        kid_mean: KID 均值
        kid_std: KID 标准差
    """
    m = features_real.shape[0]
    n = features_generated.shape[0]
    d = features_real.shape[1]

    # 如果样本数太少，无法计算
    if m < 2 or n < 2:
        return None, None

    # 如果指定了子集大小，使用子集采样
    if subset_size is not None:
        subset_size = min(subset_size, m, n)

        if subset_size < 2:
            return None, None

        kid_values = []

        for _ in range(num_subsets):
            # 随机采样子集
            idx_real = np.random.choice(m, size=subset_size, replace=False)
            idx_gen = np.random.choice(n, size=subset_size, replace=False)

            X_subset = features_real[idx_real]
            Y_subset = features_generated[idx_gen]

            # 计算子集的 KID
            kid_val = _compute_kid_unbiased(X_subset, Y_subset, d)
            kid_values.append(kid_val)

        kid_mean = np.mean(kid_values)
        kid_std = np.std(kid_values)

    else:
        # 使用全部数据
        kid_mean = _compute_kid_unbiased(features_real, features_generated, d)
        kid_std = 0.0

    return kid_mean, kid_std


def _compute_kid_unbiased(X, Y, d):
    """
    计算无偏 KID (单次)

    Args:
        X: 真实特征 (m, d)
        Y: 生成特征 (n, d)
        d: 特征维度

    Returns:
        kid: KID value
    """
    m = X.shape[0]
    n = Y.shape[0]

    # 计算核矩阵
    gamma = 1.0 / d

    # K_XX: (m, m)
    K_XX = polynomial_kernel(X, X, degree=3, gamma=gamma, coef0=1)
    # K_YY: (n, n)
    K_YY = polynomial_kernel(Y, Y, degree=3, gamma=gamma, coef0=1)
    # K_XY: (m, n)
    K_XY = polynomial_kernel(X, Y, degree=3, gamma=gamma, coef0=1)

    # term_xx: 排除对角线 (i != j)
    term_xx = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))

    # term_yy: 排除对角线 (i != j)
    term_yy = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))

    # term_xy: 所有元素
    term_xy = np.sum(K_XY) / (m * n)

    # KID = term_xx + term_yy - 2*term_xy
    kid = term_xx + term_yy - 2 * term_xy

    return kid


def main():
    parser = argparse.ArgumentParser(description="标准 FID/KID 计算")
    parser.add_argument("--real_dir", type=str, required=True,
                       help="真实图片目录 (Ground Truth)")
    parser.add_argument("--generated_dir", type=str, required=True,
                       help="生成图片目录")
    parser.add_argument("--real_pattern", type=str, default="*.png",
                       help="真实图片文件模式")
    parser.add_argument("--gen_pattern", type=str, default="*.png",
                       help="生成图片文件模式")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="特征提取批处理大小")
    parser.add_argument("--kid_subset_size", type=int, default=1000,
                       help="KID子集大小")
    parser.add_argument("--kid_num_subsets", type=int, default=100,
                       help="KID子集数量")

    args = parser.parse_args()

    print("=" * 80)
    print("标准 FID/KID 计算工具")
    print("=" * 80)
    print("实现标准:")
    print("- FID: Fréchet Inception Distance")
    print("- KID: Kernel Inception Distance (unbiased MMD^2)")
    print("- Feature: Inception V3 pool3 (2048-dim)")
    print("- Kernel: 3rd-degree polynomial k(a,b) = ((a·b)/d + 1)^3")
    print("=" * 80)

    # 检查设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    # 初始化特征提取器
    print("\n[1] 初始化 Inception V3 特征提取器...")
    extractor = InceptionV3FeatureExtractor(device=device)
    print("[OK] 特征提取器已加载")

    # 加载真实图片
    print(f"\n[2] 加载真实图片...")
    real_paths = sorted(glob.glob(os.path.join(args.real_dir, args.real_pattern)))
    print(f"真实图片路径: {args.real_dir}")
    print(f"找到 {len(real_paths)} 张真实图片")

    if len(real_paths) == 0:
        print("[错误] 未找到真实图片!")
        return

    # 加载生成图片
    print(f"\n[3] 加载生成图片...")
    gen_paths = sorted(glob.glob(os.path.join(args.generated_dir, args.gen_pattern)))
    print(f"生成图片路径: {args.generated_dir}")
    print(f"找到 {len(gen_paths)} 张生成图片")

    if len(gen_paths) == 0:
        print("[错误] 未找到生成图片!")
        return

    # 提取真实图片特征
    print(f"\n[4] 提取真实图片特征...")
    print(f"处理 {len(real_paths)} 张图片...")
    features_real = extractor.extract_features(real_paths, batch_size=args.batch_size)
    print(f"[OK] 真实特征形状: {features_real.shape}")

    # 提取生成图片特征
    print(f"\n[5] 提取生成图片特征...")
    print(f"处理 {len(gen_paths)} 张图片...")
    features_gen = extractor.extract_features(gen_paths, batch_size=args.batch_size)
    print(f"[OK] 生成特征形状: {features_gen.shape}")

    # 计算 FID
    print(f"\n[6] 计算 FID...")
    fid_value = calculate_fid_standard(features_real, features_gen)
    print(f"[OK] FID = {fid_value:.4f}")

    # 计算 KID
    print(f"\n[7] 计算 KID...")
    print(f"使用子集大小={args.kid_subset_size}, 子集数量={args.kid_num_subsets}")

    # 检查是否需要使用子集
    if len(real_paths) > args.kid_subset_size or len(gen_paths) > args.kid_subset_size:
        kid_mean, kid_std = calculate_kid_standard(
            features_real, features_gen,
            subset_size=args.kid_subset_size,
            num_subsets=args.kid_num_subsets
        )
    else:
        # 样本数较少，直接计算
        kid_mean, kid_std = calculate_kid_standard(
            features_real, features_gen,
            subset_size=None,
            num_subsets=1
        )

    if kid_mean is not None:
        print(f"[OK] KID = {kid_mean:.6f} ± {kid_std:.6f}")
        print(f"[OK] KID* (×1000) = {kid_mean*1000:.3f} ± {kid_std*1000:.3f}")
    else:
        print("[警告] KID 无法计算 (样本数不足)")

    # 总结
    print("\n" + "=" * 80)
    print("结果总结")
    print("=" * 80)
    print(f"真实图片数量: {len(real_paths)}")
    print(f"生成图片数量: {len(gen_paths)}")
    print(f"FID: {fid_value:.4f}")
    if kid_mean is not None:
        print(f"KID: {kid_mean:.6f} ± {kid_std:.6f}")
        print(f"KID* (×1000): {kid_mean*1000:.3f} ± {kid_std*1000:.3f}")
    print("=" * 80)
    print("\n解释:")
    print("- FID 和 KID 越低越好")
    print("- FID 假设特征分布是高斯分布")
    print("- KID 是基于核方法的 MMD，不假设分布类型")
    print("=" * 80)


if __name__ == "__main__":
    main()
