"""
Second-generation CNN shape scan: masked patch reconstruction + patch AGOP.

Why this version exists
-----------------------
The first CNN autoencoder task was mostly local denoising, and input-pixel AGOP
nearly saturated across shapes. The convolutional AGOP literature suggests a
better diagnostic for CNNs: patch-based AGOP. This script therefore uses:

* masked patch reconstruction, not plain denoising;
* procedural images with local and mid-range structure;
* visual-token budget based on masked supervised spatial positions;
* patch-based AGOP computed from gradients with respect to input patches;
* patch-AGOP weighted mean cos² computed directly from the patch AGOP matrix;
* stem-convolution W^T W weighted mean cos² (flattened filters, auxiliary proxy).

Run:
    python3 cnn_masked_patch_feature_scan.py --dry-run-configs
    python3 cnn_masked_patch_feature_scan.py --target-params 3000000 --layers 2 24

Smoke test:
    python3 cnn_masked_patch_feature_scan.py --target-params 100000 --layers 2 5 --max-steps 20 --aofe-images 128
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CNNConfig:
    image_size: int = 32
    channels: int = 3
    width: int = 160
    blocks: int = 8
    activation: str = "gelu"
    norm: str = "group"
    bias: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: Optional[str]) -> torch.device:
    if requested:
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {requested!r} but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use --device cpu."
            )
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_compute_setup(device: torch.device) -> None:
    """One-time stdout so users can confirm GPU vs CPU."""
    print(f"Device selected: {device!s}  (torch.cuda.is_available()={torch.cuda.is_available()})")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        print(f"  GPU name: {torch.cuda.get_device_name(idx)}  |  CUDA runtime: {torch.version.cuda}")


def group_count(width: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, width), 0, -1):
        if width % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    def __init__(self, width: int, activation: str, norm: str, bias: bool):
        super().__init__()
        self.activation = activation
        if norm == "group":
            self.norm1 = nn.GroupNorm(group_count(width), width)
            self.norm2 = nn.GroupNorm(group_count(width), width)
        elif norm == "batch":
            self.norm1 = nn.BatchNorm2d(width)
            self.norm2 = nn.BatchNorm2d(width)
        elif norm == "none":
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self._act(self.norm1(x)))
        y = self.conv2(self._act(self.norm2(y)))
        return x + y / math.sqrt(2.0)


class MaskedPatchCNN(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        # One extra channel carries the binary mask so the model knows which
        # regions are missing.
        self.stem = nn.Conv2d(cfg.channels + 1, cfg.width, 3, padding=1, bias=cfg.bias)
        self.blocks = nn.ModuleList(ConvBlock(cfg.width, cfg.activation, cfg.norm, cfg.bias) for _ in range(cfg.blocks))
        if cfg.norm == "group":
            self.out_norm = nn.GroupNorm(group_count(cfg.width), cfg.width)
        elif cfg.norm == "batch":
            self.out_norm = nn.BatchNorm2d(cfg.width)
        elif cfg.norm == "none":
            self.out_norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm: {cfg.norm}")
        self.out = nn.Conv2d(cfg.width, cfg.channels, 3, padding=1, bias=cfg.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "gelu":
            return F.gelu(x)
        if self.cfg.activation == "relu":
            return F.relu(x)
        if self.cfg.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation: {self.cfg.activation}")

    def forward(self, corrupted: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.stem(torch.cat([corrupted, mask], dim=1))
        for block in self.blocks:
            h = block(h)
        return self.out(self._act(self.out_norm(h)))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ProceduralMaskedPatchTask:
    def __init__(
        self,
        *,
        image_size: int,
        channels: int,
        patch_size: int,
        mask_ratio: float,
        shapes_per_image: int,
        device: torch.device,
    ):
        if channels != 3:
            raise ValueError("Procedural generator currently expects RGB images.")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.shapes_per_image = shapes_per_image
        self.device = device
        coords = torch.linspace(-1.0, 1.0, image_size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.xx = xx.view(1, 1, image_size, image_size)
        self.yy = yy.view(1, 1, image_size, image_size)

    def sample_clean(self, batch_size: int) -> torch.Tensor:
        b = batch_size
        device = self.device
        img = torch.rand(b, 3, 1, 1, device=device) * 0.2
        img = img + 0.08 * torch.randn(b, 3, 1, 1, device=device) * self.xx
        img = img + 0.08 * torch.randn(b, 3, 1, 1, device=device) * self.yy

        for _ in range(self.shapes_per_image):
            cx = torch.empty(b, 1, 1, 1, device=device).uniform_(-0.85, 0.85)
            cy = torch.empty(b, 1, 1, 1, device=device).uniform_(-0.85, 0.85)
            sx = torch.empty(b, 1, 1, 1, device=device).uniform_(0.06, 0.30)
            sy = torch.empty(b, 1, 1, 1, device=device).uniform_(0.06, 0.30)
            color = torch.rand(b, 3, 1, 1, device=device)
            amp = torch.empty(b, 1, 1, 1, device=device).uniform_(0.25, 0.9)
            blob = torch.exp(-0.5 * (((self.xx - cx) / sx) ** 2 + ((self.yy - cy) / sy) ** 2))
            img = img + amp * color * blob

        # Oriented stripe fields encourage edge/texture features rather than
        # pure smoothing.
        theta = torch.rand(b, 1, 1, 1, device=device) * math.pi
        freq = torch.randint(2, 9, (b, 1, 1, 1), device=device, dtype=torch.float32)
        phase = torch.rand(b, 1, 1, 1, device=device) * 2 * math.pi
        coord = torch.cos(theta) * self.xx + torch.sin(theta) * self.yy
        img = img + 0.06 * torch.sin(freq * math.pi * coord + phase)
        return img.clamp(0.0, 1.0)

    def sample_mask(self, batch_size: int) -> torch.Tensor:
        p = self.patch_size
        grid = self.image_size // p
        patch_mask = (torch.rand(batch_size, 1, grid, grid, device=self.device) < self.mask_ratio).to(torch.float32)
        return patch_mask.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean = self.sample_clean(batch_size)
        mask = self.sample_mask(batch_size)
        fill = torch.full_like(clean, 0.5)
        corrupted = clean * (1.0 - mask) + fill * mask
        return corrupted, clean, mask

    @torch.no_grad()
    def make_fixed_set(self, n: int, batch_size: int = 512) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, ys, ms = [], [], []
        remaining = n
        while remaining > 0:
            b = min(batch_size, remaining)
            x, y, m = self.sample(b)
            xs.append(x.cpu())
            ys.append(y.cpu())
            ms.append(m.cpu())
            remaining -= b
        return torch.cat(xs), torch.cat(ys), torch.cat(ms)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(target)
    return ((pred - target) ** 2 * weight).sum() / weight.sum().clamp_min(1.0)


def count_params(cfg: CNNConfig) -> int:
    return MaskedPatchCNN(cfg).parameter_count()


def choose_width(args: argparse.Namespace, blocks: int) -> Tuple[CNNConfig, int]:
    best_cfg, best_params, best_err = None, None, None
    start = max(args.width_multiple, math.ceil(args.min_width / args.width_multiple) * args.width_multiple)
    for width in range(start, args.max_width + 1, args.width_multiple):
        cfg = CNNConfig(
            image_size=args.image_size,
            channels=args.channels,
            width=width,
            blocks=blocks,
            activation=args.activation,
            norm=args.norm,
            bias=True,
        )
        params = count_params(cfg)
        err = abs(params - args.target_params)
        if best_err is None or err < best_err:
            best_cfg, best_params, best_err = cfg, params, err
    if best_cfg is None or best_params is None:
        raise RuntimeError("No valid width found")
    return best_cfg, best_params


@torch.no_grad()
def evaluate(
    model: MaskedPatchCNN,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    m_cpu: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total, count = 0.0, 0.0
    for start in range(0, x_cpu.shape[0], batch_size):
        x = x_cpu[start : start + batch_size].to(device)
        y = y_cpu[start : start + batch_size].to(device)
        m = m_cpu[start : start + batch_size].to(device)
        pred = model(x, m)
        weight = m.expand_as(y)
        total += float((((pred - y) ** 2) * weight).sum().cpu())
        count += float(weight.sum().cpu())
    model.train()
    return total / max(count, 1.0)


def configure_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, nodecay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )


def cosine_lr(step: int, steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    t = min(max((step - warmup_steps) / max(1, steps - warmup_steps), 0.0), 1.0)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (lr - min_lr)


def train(
    model: MaskedPatchCNN,
    task: ProceduralMaskedPatchTask,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    m_val: torch.Tensor,
    args: argparse.Namespace,
    train_images: int,
    device: torch.device,
) -> Dict[str, float]:
    model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    opt = configure_optimizer(model, args.lr, args.weight_decay)
    images_per_step = args.batch_size * args.grad_accum
    planned_steps = math.ceil(train_images / images_per_step)
    steps = min(planned_steps, args.max_steps) if args.max_steps is not None else planned_steps
    best_val = float("inf")
    last_train = float("nan")
    t0 = time.time()
    p0 = next(model.parameters())
    print(f"  training steps={steps} planned_steps={planned_steps} images_per_step={images_per_step}  (model on {p0.device})")
    for step in range(steps):
        lr_now = cosine_lr(step, steps, args.lr, args.min_lr, args.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(args.grad_accum):
            x, y, m = task.sample(args.batch_size)
            loss = masked_mse(model(x, m), y, m) / args.grad_accum
            loss.backward()
            accum += float(loss.detach().cpu()) * args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last_train = accum
        do_eval = step == 0 or (step + 1) % args.eval_interval == 0 or step == steps - 1
        if do_eval:
            val = evaluate(model, x_val, y_val, m_val, args.eval_batch_size, device)
            best_val = min(best_val, val)
            print(
                f"    step={step + 1:5d}/{steps} train={last_train:.6f} val={val:.6f} "
                f"lr={lr_now:.2e} elapsed={(time.time() - t0) / 60:.1f}m"
            )
        elif args.train_log_interval > 0 and (step + 1) % args.train_log_interval == 0:
            print(
                f"    step={step + 1:5d}/{steps} train={last_train:.6f} "
                f"lr={lr_now:.2e} elapsed={(time.time() - t0) / 60:.1f}m  (no val)"
            )
    final_val = evaluate(model, x_val, y_val, m_val, args.eval_batch_size, device)
    best_val = min(best_val, final_val)
    return {
        "steps": float(steps),
        "planned_steps": float(planned_steps),
        "images_per_step": float(images_per_step),
        "effective_train_images": float(steps * images_per_step),
        "final_train_loss": float(last_train),
        "final_val_loss": float(final_val),
        "best_val_loss": float(best_val),
    }


@torch.no_grad()
def offdiag_energy(mat: torch.Tensor) -> float:
    diag = torch.diagonal(mat)
    off = mat - torch.diag(diag)
    return float((off ** 2).sum().item())


@torch.no_grad()
def offdiag_ratio(mat: torch.Tensor, eps: float = 1e-12) -> float:
    diag = torch.diagonal(mat)
    off = mat - torch.diag(diag)
    return float(((off ** 2).sum() / (mat ** 2).sum().clamp_min(eps)).item())


def compute_patch_agop_metrics(
    model: MaskedPatchCNN,
    x_cpu: torch.Tensor,
    m_cpu: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    k = args.agop_patch_size
    dim = args.channels * k * k
    agop = torch.zeros((dim, dim), device=device, dtype=torch.float32)
    total_patches = 0
    for start in range(0, x_cpu.shape[0], args.aofe_batch_size):
        base = x_cpu[start : start + args.aofe_batch_size].to(device)
        mask = m_cpu[start : start + args.aofe_batch_size].to(device)
        for _ in range(args.aofe_probes):
            x = base.detach().requires_grad_(True)
            with torch.enable_grad():
                out = model(x, mask)
                probe = torch.randn_like(out) * mask.expand_as(out)
                scalar = (out * probe).sum()
                grad = torch.autograd.grad(scalar, x, retain_graph=False, create_graph=False)[0]
            patches = F.unfold(grad, kernel_size=k, padding=k // 2).transpose(1, 2).reshape(-1, dim)
            agop += patches.T @ patches
            total_patches += patches.shape[0]
    agop /= max(1, total_patches)
    agop = 0.5 * (agop + agop.T)
    aofe = offdiag_energy(agop)
    ratio = offdiag_ratio(agop)
    patch_agop_mean_cos2 = weighted_mean_cos2_from_gram(agop, eps=1e-12)
    total_energy = float((agop ** 2).sum().item())
    model.train()
    return {
        "patch_aofe": aofe,
        "patch_log10_aofe": math.log10(max(aofe, 1e-30)),
        "patch_aofe_ratio": ratio,
        "patch_agop_mean_cos2": patch_agop_mean_cos2,
        "patch_aofe_total_energy": total_energy,
    }


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x, y = list(map(float, xs)), list(map(float, ys))
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx, vy = sum((v - mx) ** 2 for v in x), sum((v - my) ** 2 for v in y)
    den = math.sqrt(vx * vy)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den if den else float("nan")


@torch.no_grad()
def weighted_mean_cos2_from_gram(G: torch.Tensor, eps: float = 1e-12) -> float:
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square, got shape={tuple(G.shape)}")
    d = torch.diagonal(G)
    off_energy = (G ** 2).sum() - (d ** 2).sum()
    denom = (d.sum() ** 2 - (d ** 2).sum()).clamp_min(eps)
    return float((off_energy / denom).item())


@torch.no_grad()
def wt_w(W: torch.Tensor) -> torch.Tensor:
    if W.ndim != 2:
        raise ValueError(f"W must be 2D, got shape={tuple(W.shape)}")
    return W.T @ W


@torch.no_grad()
def weighted_mean_cos2_from_WtW(W: torch.Tensor, eps: float = 1e-12) -> float:
    return weighted_mean_cos2_from_gram(wt_w(W), eps=eps)


@torch.no_grad()
def compute_stem_wtw_metrics(model: MaskedPatchCNN) -> Dict[str, float]:
    """Stem Conv2d [out_ch, in_ch, k, k] → W [out_ch, in_ch*k*k] for G = W^T W."""
    was_training = model.training
    model.eval()
    W = model.stem.weight.reshape(model.stem.weight.shape[0], -1)
    cos2 = weighted_mean_cos2_from_WtW(W, eps=1e-12)
    model.train(was_training)
    return {"wtw_stem_mean_cos2": cos2}


def rankdata(xs: Iterable[float]) -> List[float]:
    vals = list(map(float, xs))
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = 0.5 * (i + j)
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def correlations(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    loss = [float(r["final_val_loss"]) for r in rows]
    keys = [
        "blocks",
        "width",
        "depth_width_ratio",
        "patch_aofe",
        "patch_log10_aofe",
        "patch_aofe_ratio",
        "patch_agop_mean_cos2",
        "wtw_stem_mean_cos2",
    ]
    return {
        f"final_val_loss_vs_{k}": {
            "pearson": pearson([float(r[k]) for r in rows], loss),
            "spearman": pearson(rankdata([float(r[k]) for r in rows]), rankdata(loss)),
        }
        for k in keys
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_layers(vals: List[int]) -> List[int]:
    if len(vals) == 2:
        lo, hi = vals
        return list(range(lo, hi + 1))
    return vals


def main() -> None:
    p = argparse.ArgumentParser(description="Masked patch CNN shape scan with patch AGOP.")
    p.add_argument("--target-params", type=int, default=3_000_000)
    p.add_argument("--visual-tokens-per-param", type=float, default=20.0)
    p.add_argument("--train-images", type=int, default=None)
    p.add_argument("--layers", type=int, nargs="+", default=[2, 24])
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--mask-ratio", type=float, default=0.65)
    p.add_argument("--shapes-per-image", type=int, default=8)
    p.add_argument("--activation", choices=["gelu", "relu", "silu"], default="gelu")
    p.add_argument("--norm", choices=["group", "batch", "none"], default="group")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Run validation and print val loss every N steps (also step 0 and last).",
    )
    p.add_argument(
        "--train-log-interval",
        type=int,
        default=10,
        help="Print train loss every N steps without validation (0=disable). Fills gaps between val logs.",
    )
    p.add_argument("--val-images", type=int, default=4096)
    p.add_argument("--aofe-images", type=int, default=1024)
    p.add_argument("--aofe-batch-size", type=int, default=16)
    p.add_argument("--aofe-probes", type=int, default=2)
    p.add_argument("--agop-patch-size", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--width-multiple", type=int, default=8)
    p.add_argument("--min-width", type=int, default=16)
    p.add_argument("--max-width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("./results_cnn_masked_patch_feature_scan"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dry-run-configs", action="store_true")
    args = p.parse_args()
    print("cnn_shape_scan: starting (stdout unbuffered recommended: python -u).", flush=True)

    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    layer_list = parse_layers(args.layers)
    print(
        f"Searching widths to match --target-params={args.target_params:,} "
        f"for {len(layer_list)} depth(s) (can take ~1–5+ minutes with --layers 2 24; progress below).",
        flush=True,
    )
    configs: List[Tuple[CNNConfig, int]] = []
    for i, blocks in enumerate(layer_list):
        print(f"  [{i + 1}/{len(layer_list)}] matching width for blocks={blocks} ...", flush=True)
        configs.append(choose_width(args, blocks))
    print("Parameter-matched masked-patch CNN configs:")
    for cfg, params in configs:
        print(f"  blocks={cfg.blocks:2d} width={cfg.width:4d} params={params:,} ({100*(params-args.target_params)/args.target_params:+.2f}%)")
    print_compute_setup(device)
    if args.dry_run_configs:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        dry_path = args.out_dir / "dry_run_configs.csv"
        dry_rows: List[Dict[str, object]] = []
        for cfg, params in configs:
            dry_rows.append(
                {
                    "blocks": cfg.blocks,
                    "width": cfg.width,
                    "depth_width_ratio": cfg.blocks / cfg.width,
                    "param_count": params,
                    "param_error": params - args.target_params,
                    "param_error_pct": 100.0 * (params - args.target_params) / args.target_params,
                    "target_params": args.target_params,
                }
            )
        write_csv(dry_path, dry_rows)
        print(f"Wrote dry-run table to {dry_path.resolve()}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        payload = vars(args).copy()
        payload["out_dir"] = str(args.out_dir)
        payload["device_resolved"] = str(device)
        json.dump(payload, f, indent=2)

    set_seed(args.seed)
    task = ProceduralMaskedPatchTask(
        image_size=args.image_size,
        channels=args.channels,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        shapes_per_image=args.shapes_per_image,
        device=device,
    )
    set_seed(args.seed + 17)
    print(f"Creating validation set: {args.val_images:,} images")
    x_val, y_val, m_val = task.make_fixed_set(args.val_images)
    set_seed(args.seed + 23)
    print(f"Creating patch-AGOP set: {args.aofe_images:,} images")
    x_aofe, _, m_aofe = task.make_fixed_set(args.aofe_images)

    supervised_positions_per_image = args.image_size * args.image_size * args.mask_ratio
    rows: List[Dict[str, object]] = []
    for cfg, params in configs:
        print("\n" + "=" * 80)
        print(f"student blocks={cfg.blocks} width={cfg.width} params={params:,}")
        set_seed(args.seed + cfg.blocks)
        model = MaskedPatchCNN(cfg)
        visual_tokens = int(round(args.visual_tokens_per_param * params))
        train_images = args.train_images or math.ceil(visual_tokens / supervised_positions_per_image)
        if args.train_images is not None:
            visual_tokens = int(round(train_images * supervised_positions_per_image))
        stats = train(model, task, x_val, y_val, m_val, args, train_images, device)
        agop = compute_patch_agop_metrics(model, x_aofe, m_aofe, args, device)
        wtw = compute_stem_wtw_metrics(model)
        row: Dict[str, object] = {
            "blocks": cfg.blocks,
            "width": cfg.width,
            "depth_width_ratio": cfg.blocks / cfg.width,
            "param_count": params,
            "param_error": params - args.target_params,
            "train_visual_token_budget": visual_tokens,
            "supervised_positions_per_image": supervised_positions_per_image,
            "train_image_budget": train_images,
            **agop,
            **stats,
            **wtw,
        }
        rows.append(row)
        write_csv(args.out_dir / "depth_scan_results.csv", rows)
        with (args.out_dir / "depth_scan_results.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(
            f"  loss={stats['final_val_loss']:.6f} patch_AOFE={agop['patch_aofe']:.3e} "
            f"patch_ratio={agop['patch_aofe_ratio']:.4f} "
            f"patch_AGOP_cos2={agop['patch_agop_mean_cos2']:.4f} "
            f"W^TW_cos2(stem)={wtw['wtw_stem_mean_cos2']:.4f}"
        )

    corr = correlations(rows)
    with (args.out_dir / "correlations.json").open("w", encoding="utf-8") as f:
        json.dump(corr, f, indent=2)
    print("\nCorrelation summary:")
    for k, v in corr.items():
        print(f"  {k}: Pearson={v['pearson']:.4f}, Spearman={v['spearman']:.4f}")
    print(f"\nSaved results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()