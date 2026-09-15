#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DressCode extractor ablation runner (eval-only, S-only).

- Runs S0/S1/S2 only.
- Forwards unknown args to train_extractor_DC.py.
- Computes PSNR/SSIM/LPIPS/FID/KID from eval_images/real and eval_images/fake.
- Writes metrics JSON per run and a summary JSON at output_root.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

RUNNER_FLAGS_WITH_VALUE = {
    "--train_script",
    "--output_root",
    "--checkpoint",
    "--s_modes",
    "--p_modes",
    "--metrics_out",
}
RUNNER_FLAGS = {
    "--eval_only",
    "--no_p_grid",
    "--skip_if_exists",
    "--dry_run",
    "--compute_visual_metrics",
}


def _split_s_modes(value: str) -> List[str]:
    parts = re.split(r"[\s,]+", value.strip())
    modes = [p.strip().upper() for p in parts if p.strip()]
    if not modes:
        raise SystemExit("s_modes cannot be empty.")
    invalid = [m for m in modes if m not in ("S0", "S1", "S2")]
    if invalid:
        raise SystemExit(f"Invalid s_modes: {', '.join(invalid)}. Valid: S0,S1,S2")
    return modes


def _strip_runner_args(argv: List[str]) -> List[str]:
    cleaned: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            i += 1
            continue
        if arg in RUNNER_FLAGS_WITH_VALUE:
            i += 2
            continue
        if arg in RUNNER_FLAGS:
            i += 1
            continue
        if any(arg.startswith(f"{k}=") for k in RUNNER_FLAGS_WITH_VALUE):
            i += 1
            continue
        if any(arg.startswith(f"{k}=") for k in RUNNER_FLAGS):
            i += 1
            continue
        # Also strip ablation args if passed in train args (runner owns them)
        if arg in ("--ablation_s", "--ablation_p", "--ablation_spec", "--ablation_mode"):
            i += 2 if arg in ("--ablation_s", "--ablation_p") else 1
            continue
        if arg.startswith("--ablation_s=") or arg.startswith("--ablation_p="):
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _as_float_or_none(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if text in {"-inf", "-infinity"}:
            return float("-inf")
        try:
            return float(text)
        except Exception:
            return None
    return None


def _is_invalid_metrics_payload(metrics: Dict) -> bool:
    if not isinstance(metrics, dict) or not metrics:
        return True

    has_pixel_metrics = (
        all(k in metrics for k in ("psnr", "ssim", "lpips"))
        or all(k in metrics for k in ("psnr_mean", "ssim_mean", "lpips_mean"))
    )
    if not has_pixel_metrics:
        return True

    fid_value = _as_float_or_none(metrics.get("fid"))
    kid_value = _as_float_or_none(metrics.get("kid"))
    kid_std_value = _as_float_or_none(metrics.get("kid_std"))
    if fid_value is None or kid_value is None or kid_std_value is None:
        return True

    fid_error = metrics.get("fid_error")
    kid_error = metrics.get("kid_error")
    metrics_error = metrics.get("metrics_error")
    if (fid_value == -1.0 or kid_value == -1.0) and (
        bool(fid_error) or bool(kid_error) or bool(metrics_error)
    ):
        return True
    return False


def _collect_images(root: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = str(p.relative_to(root)).replace("\\", "/")
            files[rel] = p
    return files


def _match_pairs(real_dir: Path, fake_dir: Path) -> Tuple[List[Tuple[Path, Path, str]], List[str], List[str]]:
    real_map = _collect_images(real_dir)
    fake_map = _collect_images(fake_dir)
    common = sorted(set(real_map.keys()) & set(fake_map.keys()))
    missing_real = sorted(set(fake_map.keys()) - set(real_map.keys()))
    missing_fake = sorted(set(real_map.keys()) - set(fake_map.keys()))
    pairs = [(real_map[k], fake_map[k], k) for k in common]
    return pairs, missing_real, missing_fake


def _image_to_tensor(img, device):
    import numpy as np
    import torch

    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor.to(device)


def _create_ssim_window(window_size: int, channels: int, device, dtype):
    import torch

    sigma = 1.5
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def _ssim(x, y, window):
    import torch
    import torch.nn.functional as F

    channels = x.size(1)
    padding = window.size(2) // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / den
    return ssim_map.mean(dim=(1, 2, 3))


def _psnr(x, y, eps: float = 1e-8):
    import torch

    mse = torch.mean((x - y) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=eps)
    return 10.0 * torch.log10(1.0 / mse)


def _get_lpips_fn(device):
    try:
        import torch
        import lpips

        model = lpips.LPIPS(net="alex")
        model.to(device)
        model.eval()

        def _fn(fake, real):
            with torch.no_grad():
                f = fake * 2.0 - 1.0
                r = real * 2.0 - 1.0
                d = model(f, r)
                return d.view(d.shape[0])

        return _fn, "lpips"
    except Exception:
        try:
            import torch
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
            metric.eval()

            def _fn(fake, real):
                vals = []
                with torch.no_grad():
                    for i in range(fake.shape[0]):
                        v = metric(fake[i : i + 1], real[i : i + 1])
                        vals.append(v.view(-1))
                return torch.cat(vals, dim=0)

            return _fn, "torchmetrics"
        except Exception as exc:
            raise RuntimeError(
                "LPIPS requires the 'lpips' package (or torchmetrics) in the IMAGDressing env."
            ) from exc


def _compute_visual_metrics(real_dir: Path, fake_dir: Path, batch_size: int = 8) -> Dict:
    import torch

    from PIL import Image

    pairs, missing_real, missing_fake = _match_pairs(real_dir, fake_dir)
    if not pairs:
        raise RuntimeError("No matching image pairs found for visual metrics.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn, lpips_backend = _get_lpips_fn(device)
    window = _create_ssim_window(11, 3, device, torch.float32)

    total = len(pairs)
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    count = 0

    for i in range(0, total, batch_size):
        batch_pairs = pairs[i : i + batch_size]
        real_batch = []
        fake_batch = []
        for real_path, fake_path, _ in batch_pairs:
            real_img = Image.open(real_path).convert("RGB")
            fake_img = Image.open(fake_path).convert("RGB")
            if fake_img.size != real_img.size:
                fake_img = fake_img.resize(real_img.size, resample=Image.BILINEAR)
            real_batch.append(_image_to_tensor(real_img, device))
            fake_batch.append(_image_to_tensor(fake_img, device))

        real_t = torch.stack(real_batch, dim=0).to(device)
        fake_t = torch.stack(fake_batch, dim=0).to(device)

        with torch.no_grad():
            psnr_vals = _psnr(fake_t, real_t)
            ssim_vals = _ssim(fake_t, real_t, window)
            lpips_vals = lpips_fn(fake_t, real_t)

        psnr_sum += float(psnr_vals.sum().item())
        ssim_sum += float(ssim_vals.sum().item())
        lpips_sum += float(lpips_vals.sum().item())
        count += real_t.shape[0]

    metrics = {
        "psnr": psnr_sum / max(count, 1),
        "ssim": ssim_sum / max(count, 1),
        "lpips": lpips_sum / max(count, 1),
        "lpips_backend": lpips_backend,
        "num_pairs": count,
        "missing_real": len(missing_real),
        "missing_fake": len(missing_fake),
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
    }
    return metrics


def _pick_pattern(folder: Path) -> str:
    for ext in ("png", "jpg", "jpeg", "webp", "bmp"):
        if any(folder.glob(f"*.{ext}")):
            return f"*.{ext}"
    return "*.*"


def _parse_fid_kid_output(stdout: str) -> Tuple[float, float, float]:
    fid = None
    kid = None
    kid_std = None

    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("FID:"):
            try:
                fid = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
        if line.startswith("KID:"):
            rhs = line.split(":", 1)[1].strip()
            for sep in ("\u00b1", "\u5364", "+/-"):
                if sep in rhs:
                    a, b = rhs.split(sep, 1)
                    try:
                        kid = float(a.strip())
                        kid_std = float(b.strip())
                    except Exception:
                        pass
                    break

    if fid is None:
        m = re.search(r"FID\s*=\s*([0-9.+-eE]+)", stdout)
        if m:
            fid = float(m.group(1))
    if kid is None:
        m = re.search(r"KID\s*=\s*([0-9.+-eE]+)\s*(?:\u00b1|\u5364|\+/-)\s*([0-9.+-eE]+)", stdout)
        if m:
            kid = float(m.group(1))
            kid_std = float(m.group(2))

    if fid is None or kid is None or kid_std is None:
        raise RuntimeError("Failed to parse FID/KID from calculate_fid_kid_standard.py output.")
    return fid, kid, kid_std


def _run_fid_kid_standard(real_dir: Path, fake_dir: Path) -> Tuple[float, float, float]:
    script = Path(__file__).resolve().parent / "calculate_fid_kid_standard.py"
    if not script.exists():
        raise RuntimeError(f"Missing calculate_fid_kid_standard.py at: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--real_dir",
        str(real_dir),
        "--generated_dir",
        str(fake_dir),
        "--real_pattern",
        _pick_pattern(real_dir),
        "--gen_pattern",
        _pick_pattern(fake_dir),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
    if proc.returncode != 0:
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError(f"calculate_fid_kid_standard.py failed (code={proc.returncode}).\n{out}")

    stdout = proc.stdout or ""
    if not stdout:
        stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return _parse_fid_kid_output(stdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_ablation_extractor_DC.py",
        description="Run DressCode extractor ablations (S-only, eval-only) and compute metrics.",
    )
    parser.add_argument("--train_script", required=True, help="Path to train_extractor_DC.py")
    parser.add_argument("--output_root", required=True, help="Root dir to write ablation outputs")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to load for eval-only")
    parser.add_argument("--eval_only", action="store_true", help="Run eval-only (required)")
    parser.add_argument("--s_modes", default="S0,S1,S2", help="Comma-separated S modes")
    parser.add_argument("--p_modes", default="", help="Ignored (legacy)")
    parser.add_argument("--no_p_grid", action="store_true", help="Ignored (legacy)")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if metrics_out exists")
    parser.add_argument("--dry_run", action="store_true", help="Print commands but do not execute")
    parser.add_argument("--compute_visual_metrics", action="store_true", help="Compute PSNR/SSIM/LPIPS/FID/KID")
    parser.add_argument("--metrics_out", default="eval_metrics.json", help="Metrics JSON filename")

    args, unknown = parser.parse_known_args()

    if not args.eval_only:
        raise SystemExit("--eval_only is required for this runner.")
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required when using --eval_only.")

    train_script = Path(args.train_script)
    if not train_script.exists():
        raise SystemExit(f"train_script not found: {train_script}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    s_modes = _split_s_modes(args.s_modes)

    passthrough_args = _strip_runner_args([a for a in unknown if a != "--"])

    runs = []
    exit_code = 0

    for idx, s_mode in enumerate(s_modes, start=1):
        run_dir = output_root / s_mode
        metrics_path = run_dir / args.metrics_out

        if args.skip_if_exists and metrics_path.exists():
            existing_metrics = _read_json(metrics_path)
            if not _is_invalid_metrics_payload(existing_metrics):
                runs.append({
                    "s_mode": s_mode,
                    "run_dir": str(run_dir),
                    "status": "skipped",
                    "metrics_path": str(metrics_path),
                })
                print(f"[{idx}/{len(s_modes)}] SKIP {s_mode} (metrics exist): {metrics_path}")
                continue
            print(f"[{idx}/{len(s_modes)}] RE-RUN {s_mode} (invalid metrics found): {metrics_path}")

        cmd = [
            sys.executable,
            str(train_script),
            "--output_dir",
            str(run_dir),
            "--eval_only",
            "--checkpoint",
            args.checkpoint,
            "--ablation_s",
            s_mode,
        ]
        cmd += passthrough_args

        print(f"[{idx}/{len(s_modes)}] {s_mode} -> {run_dir}")
        print(subprocess.list2cmdline(cmd))

        if args.dry_run:
            runs.append({
                "s_mode": s_mode,
                "run_dir": str(run_dir),
                "status": "dry_run",
            })
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            runs.append({
                "s_mode": s_mode,
                "run_dir": str(run_dir),
                "status": f"train_failed({proc.returncode})",
            })
            exit_code = proc.returncode if exit_code == 0 else exit_code
            continue

        metrics = {}
        try:
            if args.compute_visual_metrics:
                existing_metrics = _read_json(metrics_path)
                if not _is_invalid_metrics_payload(existing_metrics):
                    metrics = existing_metrics
                    print(f"[{idx}/{len(s_modes)}] Reuse metrics: {metrics_path}")
                else:
                    real_dir = run_dir / "eval_images" / "real"
                    fake_dir = run_dir / "eval_images" / "fake"

                    if not real_dir.exists() or not fake_dir.exists():
                        raise RuntimeError("Missing eval_images/real or eval_images/fake for metrics.")

                    metrics.update(_compute_visual_metrics(real_dir, fake_dir))
                    fid, kid, kid_std = _run_fid_kid_standard(real_dir, fake_dir)
                    metrics.update({
                        "fid": fid,
                        "kid": kid,
                        "kid_std": kid_std,
                        "fid_kid_backend": "calculate_fid_kid_standard.py",
                    })
                    metrics.update({
                        "ablation_s": s_mode,
                        "checkpoint": args.checkpoint,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })

                    _write_json(metrics_path, metrics)
            else:
                print("[WARN] --compute_visual_metrics not set; metrics will not be written.")
        except Exception as exc:
            runs.append({
                "s_mode": s_mode,
                "run_dir": str(run_dir),
                "status": f"metrics_failed({exc})",
            })
            exit_code = exit_code or 1
            continue

        runs.append({
            "s_mode": s_mode,
            "run_dir": str(run_dir),
            "status": "ok",
            "metrics_path": str(metrics_path) if metrics_path.exists() else "",
            "metrics": metrics if metrics else None,
        })

    summary = {
        "output_root": str(output_root),
        "metrics_out": args.metrics_out,
        "s_modes": s_modes,
        "runs": runs,
    }
    summary_path = output_root / "ablation_summary.json"
    _write_json(summary_path, summary)
    print(f"Summary written: {summary_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
