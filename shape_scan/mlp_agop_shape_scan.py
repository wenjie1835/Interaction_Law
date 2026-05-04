#!/usr/bin/env python3
"""
Parameter-matched MLP shape scan with AGOP/AOFE metrics.

Why this script exists
----------------------
The AGOP paper validates MLP feature learning primarily on vector/tabular tasks
(and also vectorized images), and the preprint includes low-rank polynomial
synthetic tasks that isolate feature learning in a clean, self-contained way.

This script turns that idea into a *shape scan* under a fixed parameter budget:
- keep total parameter count approximately fixed,
- vary depth,
- choose width to match the target parameter budget,
- train on a low-rank regression task with genuine feature interactions,
- measure validation/test loss together with AGOP-derived metrics:
    * AOFE            = off-diagonal AGOP energy
    * AOFE-ratio      = off-diagonal AGOP energy ratio
    * AGOP mean cos^2 = weighted mean pairwise cos^2 computed from the AGOP matrix

Default task
------------
The default synthetic task mixes two ingredients inspired by the AGOP preprint:
1) pairwise coordinate interaction (x0 * x1), and
2) a low-rank nonlinear component g(u^T x) with Hermite polynomials.
This makes the target depend on a small interacting subspace, which is exactly
where AGOP-style analysis is informative.

Run:
    python3 mlp_agop_shape_scan.py --dry-run-configs
    python3 mlp_agop_shape_scan.py --target-params 300000 --depths 1 12

Training stops when validation MSE fails to improve by ``--early-stop-min-delta`` for
``--early-stop-patience`` consecutive evaluations (after ``--min-steps-before-early-stop``),
or when ``--train-step-ceiling`` / ``--max-steps`` is reached. The checkpoint with the best
validation loss is always restored (plateau / "convergence" in the early-stop sense).

Smoke test:
    python3 mlp_agop_shape_scan.py --target-params 50000 --depths 1 4 --train-samples 2048 \
        --val-samples 512 --test-samples 1024 --agop-samples 256 --max-steps 50 \
        --train-step-ceiling 0 --early-stop-patience 0
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Reproducibility / utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def offdiag_energy(mat: torch.Tensor) -> float:
    diag = torch.diagonal(mat)
    off = mat - torch.diag(diag)
    return float((off ** 2).sum().item())


def offdiag_ratio(mat: torch.Tensor, eps: float = 1e-12) -> float:
    diag = torch.diagonal(mat)
    off = mat - torch.diag(diag)
    den = (mat ** 2).sum().clamp_min(eps)
    return float(((off ** 2).sum() / den).item())




@torch.no_grad()
def weighted_mean_cos2_from_gram(G: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Weighted mean cos^2 from a Gram-like matrix G.

    Interpreting G_ij = <v_i, v_j> and d_i = G_ii = ||v_i||^2,

        weighted_mean_cos2 = sum_{i != j} G_ij^2 / sum_{i != j} d_i d_j.

    For a PSD Gram matrix this is the weighted average of pairwise cos^2 values,
    with weights ||v_i||^2 ||v_j||^2. Here we apply it directly to AGOP.
    """
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be a square matrix, got shape={tuple(G.shape)}")
    G = G.to(torch.float64)
    G = 0.5 * (G + G.T)
    d = torch.diagonal(G).clamp_min(0.0)
    off_energy = (G ** 2).sum() - (d ** 2).sum()
    denom = (d.sum() ** 2 - (d ** 2).sum()).clamp_min(eps)
    return float((off_energy / denom).item())


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x, y = list(map(float, xs)), list(map(float, ys))
    if len(x) != len(y) or not x:
        return float("nan")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    den = math.sqrt(vx * vy)
    if den == 0.0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den


# -----------------------------
# Synthetic low-rank task
# -----------------------------

@dataclass(frozen=True)
class VectorTaskConfig:
    input_dim: int = 64
    latent_dim: int = 8
    target_noise: float = 0.03
    task: str = "hybrid"


class LowRankVectorTask:
    """
    Self-contained vector regression task with feature interactions.

    Modes:
      - product : target depends on x0 * x1 plus nuisance noise
      - hermite : target depends on low-rank projections via Hermite polynomials
      - hybrid  : combines both, which is the default and strongest signal
    """

    def __init__(self, cfg: VectorTaskConfig, device: torch.device, seed: int) -> None:
        self.cfg = cfg
        self.device = device
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        input_dim = cfg.input_dim
        latent_dim = cfg.latent_dim

        self.A = F.normalize(torch.randn(input_dim, latent_dim, generator=g, device=device), dim=0)
        self.B = F.normalize(torch.randn(input_dim, latent_dim, generator=g, device=device), dim=0)
        self.u = F.normalize(torch.randn(input_dim, generator=g, device=device), dim=0)
        self.v = F.normalize(torch.randn(input_dim, generator=g, device=device), dim=0)
        self.w = F.normalize(torch.randn(input_dim, generator=g, device=device), dim=0)
        self.mix = torch.randn(latent_dim, generator=g, device=device)

    @staticmethod
    def _he2(z: torch.Tensor) -> torch.Tensor:
        return z ** 2 - 1.0

    @staticmethod
    def _he4(z: torch.Tensor) -> torch.Tensor:
        return z ** 4 - 6.0 * z ** 2 + 3.0

    @torch.no_grad()
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, self.cfg.input_dim, device=self.device)

        # Inject a low-rank structured component.
        latent = x @ self.A
        interacted = (latent * (x @ self.B)) @ self.mix / math.sqrt(self.cfg.latent_dim)

        prod = x[:, 0] * x[:, 1]
        hermite = 0.8 * self._he2(x @ self.u) + 0.35 * self._he4(x @ self.v)
        ridge = 0.25 * torch.sin(2.0 * (x @ self.w))

        if self.cfg.task == "product":
            y = prod + 0.20 * ridge
        elif self.cfg.task == "hermite":
            y = hermite + 0.20 * ridge
        elif self.cfg.task == "hybrid":
            y = 0.55 * prod + 0.55 * hermite + 0.35 * interacted + 0.20 * ridge
        else:
            raise ValueError(f"Unknown task: {self.cfg.task}")

        y = y + self.cfg.target_noise * torch.randn_like(y)
        return x, y.unsqueeze(1)

    @torch.no_grad()
    def make_fixed_set(self, n: int, batch_size: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        remaining = n
        while remaining > 0:
            b = min(batch_size, remaining)
            x, y = self.sample(b)
            xs.append(x.cpu())
            ys.append(y.cpu())
            remaining -= b
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


# -----------------------------
# Model and parameter matching
# -----------------------------

class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, activation: str, bias: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width, bias=bias)
        self.fc2 = nn.Linear(width, width, bias=bias)
        self.activation = activation

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(self._act(x))
        y = self.fc2(self._act(y))
        return x + y / math.sqrt(2.0)


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    width: int
    depth: int
    activation: str = "gelu"
    bias: bool = True
    init_std: float = 0.05


class ResidualMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.input_dim, cfg.width, bias=cfg.bias)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(cfg.width, cfg.activation, cfg.bias) for _ in range(cfg.depth)
        )
        self.out_proj = nn.Linear(cfg.width, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(self._act(h))


def count_params(cfg: MLPConfig) -> int:
    w = cfg.width
    d = cfg.depth
    # in_proj + residual blocks + out_proj
    return (cfg.input_dim * w + w) + d * (2 * w * w + 2 * w) + (w + 1)


def parse_depths(depth_args: Sequence[int]) -> List[int]:
    if len(depth_args) == 1:
        return [int(depth_args[0])]
    if len(depth_args) == 2:
        lo, hi = int(depth_args[0]), int(depth_args[1])
        return list(range(lo, hi + 1))
    return [int(x) for x in depth_args]


def choose_width(args: argparse.Namespace, depth: int) -> Tuple[MLPConfig, int]:
    best_cfg: Optional[MLPConfig] = None
    best_params: Optional[int] = None
    best_err: Optional[int] = None
    start = max(args.width_multiple, math.ceil(args.min_width / args.width_multiple) * args.width_multiple)
    for width in range(start, args.max_width + 1, args.width_multiple):
        cfg = MLPConfig(
            input_dim=args.input_dim,
            width=width,
            depth=depth,
            activation=args.activation,
            bias=True,
            init_std=args.init_std,
        )
        params = count_params(cfg)
        err = abs(params - args.target_params)
        if best_err is None or err < best_err:
            best_cfg, best_params, best_err = cfg, params, err
    if best_cfg is None or best_params is None:
        raise RuntimeError("No valid width found")
    return best_cfg, best_params


# -----------------------------
# Training / evaluation / AGOP
# -----------------------------

def evaluate(
    model: ResidualMLP,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, x_cpu.shape[0], batch_size):
            x = x_cpu[start : start + batch_size].to(device)
            y = y_cpu[start : start + batch_size].to(device)
            pred = model(x)
            total += F.mse_loss(pred, y, reduction="sum").item()
            count += x.shape[0]
    model.train()
    return total / max(1, count)


def train_model(
    cfg: MLPConfig,
    x_train_cpu: torch.Tensor,
    y_train_cpu: torch.Tensor,
    x_val_cpu: torch.Tensor,
    y_val_cpu: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[ResidualMLP, Dict[str, float]]:
    model = ResidualMLP(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.max_steps
    if steps is None:
        base = max(1, math.ceil(args.train_samples / args.batch_size) * args.epochs)
        if args.train_step_ceiling > 0:
            steps = max(base, args.train_step_ceiling)
        else:
            steps = base

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=args.min_lr)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    last_train = float("inf")
    evals_without_improve = 0
    early_stopped = False

    n = x_train_cpu.shape[0]
    step_ceiling = steps
    for step in range(step_ceiling):
        idx = torch.randint(0, n, (args.batch_size,))
        x = x_train_cpu[idx].to(device)
        y = y_train_cpu[idx].to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        scheduler.step()
        last_train = float(loss.item())

        is_eval_step = (step + 1) % args.eval_interval == 0 or step + 1 == step_ceiling
        if is_eval_step:
            val = evaluate(model, x_val_cpu, y_val_cpu, args.eval_batch_size, device)
            improved = val < best_val - args.early_stop_min_delta
            if improved:
                best_val = val
                evals_without_improve = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            elif args.early_stop_patience > 0:
                evals_without_improve += 1

            can_stop = (
                args.early_stop_patience > 0
                and step + 1 >= args.min_steps_before_early_stop
                and evals_without_improve >= args.early_stop_patience
            )
            if can_stop:
                early_stopped = True
                break

    train_steps_done = step + 1

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate(model, x_val_cpu, y_val_cpu, args.eval_batch_size, device)
    return model, {
        "steps": float(step_ceiling),
        "train_steps_done": float(train_steps_done),
        "early_stopped": float(1.0 if early_stopped else 0.0),
        "final_train_loss": float(last_train),
        "final_val_loss": float(final_val),
        "best_val_loss": float(best_val),
    }


def compute_agop_metrics(
    model: ResidualMLP,
    x_cpu: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    dim = x_cpu.shape[1]
    agop = torch.zeros((dim, dim), device=device, dtype=torch.float32)
    total = 0

    for start in range(0, x_cpu.shape[0], args.agop_batch_size):
        base = x_cpu[start : start + args.agop_batch_size].to(device)
        for _ in range(args.agop_probes):
            x = base.detach().requires_grad_(True)
            with torch.enable_grad():
                out = model(x)
                probe = torch.randn_like(out)
                scalar = (out * probe).sum()
                grad = torch.autograd.grad(scalar, x, retain_graph=False, create_graph=False)[0]
            agop += grad.T @ grad
            total += grad.shape[0]

    agop /= max(1, total)
    agop = 0.5 * (agop + agop.T)
    aofe = offdiag_energy(agop)
    ratio = offdiag_ratio(agop)
    total_energy = float((agop ** 2).sum().item())
    diag_energy = float((torch.diagonal(agop) ** 2).sum().item())
    agop_mean_cos2 = weighted_mean_cos2_from_gram(agop, eps=1e-12)
    model.train()
    return {
        "aofe": aofe,
        "log10_aofe": math.log10(max(aofe, 1e-30)),
        "aofe_ratio": ratio,
        "aofe_total_energy": total_energy,
        "aofe_diag_energy": diag_energy,
        "agop_mean_cos2": agop_mean_cos2,
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-params", type=int, default=300_000)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 12])
    p.add_argument("--input-dim", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--task", choices=["product", "hermite", "hybrid"], default="hybrid")
    p.add_argument("--target-noise", type=float, default=0.03)
    p.add_argument("--activation", choices=["gelu", "relu", "silu"], default="gelu")
    p.add_argument("--init-std", type=float, default=0.03)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--train-step-ceiling",
        type=int,
        default=60_000,
        help=(
            "When --max-steps is unset: run at least max(epochs·batches, this value). "
            "Set 0 to use only the epochs-derived step count (legacy). "
            "Early stopping may finish earlier."
        ),
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Stop after this many consecutive validation checks without improvement (0 disables).",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-8,
        help="Minimum validation MSE decrease to count as an improvement.",
    )
    p.add_argument(
        "--min-steps-before-early-stop",
        type=int,
        default=0,
        help="Do not early-stop before this many steps (stabilizes first noisy evals).",
    )
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--train-samples", type=int, default=20_000)
    p.add_argument("--val-samples", type=int, default=4_096)
    p.add_argument("--test-samples", type=int, default=8_192)
    p.add_argument("--agop-samples", type=int, default=2_048)
    p.add_argument("--agop-batch-size", type=int, default=128)
    p.add_argument("--agop-probes", type=int, default=2)
    p.add_argument("--width-multiple", type=int, default=8)
    p.add_argument("--min-width", type=int, default=16)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("./results_mlp_agop_shape_scan"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dry-run-configs", action="store_true")
    args = p.parse_args()

    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_seed(args.seed)

    depths = parse_depths(args.depths)
    configs = [choose_width(args, depth) for depth in depths]
    print("Parameter-matched MLP configs:")
    for cfg, params in configs:
        pct = 100.0 * (params - args.target_params) / max(1, args.target_params)
        print(f"  depth={cfg.depth:2d} width={cfg.width:4d} params={params:,} ({pct:+.2f}%)")
    print(f"Device: {device}")
    if args.dry_run_configs:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    task = LowRankVectorTask(
        VectorTaskConfig(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            target_noise=args.target_noise,
            task=args.task,
        ),
        device=device,
        seed=args.seed + 11,
    )
    print("Creating fixed datasets...")
    x_train, y_train = task.make_fixed_set(args.train_samples)
    x_val, y_val = task.make_fixed_set(args.val_samples)
    x_test, y_test = task.make_fixed_set(args.test_samples)
    x_agop, _ = task.make_fixed_set(args.agop_samples)

    rows: List[Dict[str, float]] = []
    for cfg, params in configs:
        print(f"\n=== depth={cfg.depth} width={cfg.width} ===")
        set_seed(args.seed + cfg.depth)
        model, stats = train_model(cfg, x_train, y_train, x_val, y_val, args, device)
        test_loss = evaluate(model, x_test, y_test, args.eval_batch_size, device)
        agop = compute_agop_metrics(model, x_agop, args, device)
        row = {
            "depth": float(cfg.depth),
            "width": float(cfg.width),
            "depth_width_ratio": float(cfg.depth / cfg.width),
            "param_count": float(params),
            "param_error": float(params - args.target_params),
            **stats,
            "test_loss": float(test_loss),
            **agop,
        }
        rows.append(row)
        print(
            f"  val={stats['best_val_loss']:.6f} test={test_loss:.6f} "
            f"steps={int(stats['train_steps_done'])}/{int(stats['steps'])} "
            f"early_stop={int(stats['early_stopped'])} "
            f"AOFE={agop['aofe']:.3e} ratio={agop['aofe_ratio']:.4f} "
            f"AGOP_cos2={agop['agop_mean_cos2']:.4f}"
        )

    # Correlation summary
    losses = [r["best_val_loss"] for r in rows]
    aofe_vals = [r["aofe"] for r in rows]
    ratio_vals = [r["aofe_ratio"] for r in rows]
    agop_cos2_vals = [r["agop_mean_cos2"] for r in rows]
    print("\nCorrelations against validation loss:")
    print(f"  Pearson(val_loss, AOFE)           = {pearson(losses, aofe_vals):.4f}")
    print(f"  Pearson(val_loss, AOFE-ratio)     = {pearson(losses, ratio_vals):.4f}")
    print(f"  Pearson(val_loss, AGOP mean cos2) = {pearson(losses, agop_cos2_vals):.4f}")
    print(f"  Pearson(AOFE, AGOP mean cos2)     = {pearson(aofe_vals, agop_cos2_vals):.4f}")

    out_csv = args.out_dir / "mlp_shape_scan_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()
