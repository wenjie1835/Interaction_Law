"""
Second-generation MLP shape scan: hierarchical feature-learning task.

Why this version exists
-----------------------
The first MLP teacher-student task was too easy for shallow-wide students and
did not strongly require feature discovery. This script makes the task closer
to the AGOP / Neural Feature Ansatz picture:

* high-dimensional input with many nuisance dimensions;
* the teacher depends only on a hidden useful subspace;
* local useful groups are composed into multi-output targets;
* students are residual MLPs, reducing optimization failure in deep models;
* AGOP is measured on the input with Hutchinson probes for multi-output models.

The key extra diagnostic is whether the learned AGOP concentrates on useful
dimensions and whether shape changes alter AOFE / AOFE-ratio in the
feature-learning regime.

Run:
    python3 mlp_hierarchical_feature_scan.py --dry-run-configs
    python3 mlp_hierarchical_feature_scan.py --target-params 3000000 --layers 2 24

Smoke test:
    python3 mlp_hierarchical_feature_scan.py --target-params 100000 --layers 2 5 --max-steps 20 --val-samples 4096 --aofe-samples 1024
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
class StudentConfig:
    input_dim: int = 512
    output_dim: int = 32
    width: int = 768
    blocks: int = 8
    activation: str = "gelu"
    norm: str = "layer"
    bias: bool = True


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, activation: str, norm: str, bias: bool):
        super().__init__()
        self.activation = activation
        self.norm = nn.LayerNorm(width) if norm == "layer" else nn.Identity()
        self.fc1 = nn.Linear(width, 4 * width, bias=bias)
        self.fc2 = nn.Linear(4 * width, width, bias=bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(self._act(self.fc1(self.norm(x))))
        return x + y / math.sqrt(2.0)


class ResidualStudentMLP(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.input_dim, cfg.width, bias=cfg.bias)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(cfg.width, cfg.activation, cfg.norm, cfg.bias)
            for _ in range(cfg.blocks)
        )
        self.out_norm = nn.LayerNorm(cfg.width) if cfg.norm == "layer" else nn.Identity()
        self.out = nn.Linear(cfg.width, cfg.output_dim, bias=cfg.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "gelu":
            return F.gelu(x)
        if self.cfg.activation == "relu":
            return F.relu(x)
        if self.cfg.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation: {self.cfg.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._act(self.in_proj(x))
        for block in self.blocks:
            h = block(h)
        return self.out(self._act(self.out_norm(h)))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HierarchicalTeacher(nn.Module):
    """
    Teacher that ignores nuisance dimensions and composes local useful groups.
    """
    def __init__(
        self,
        *,
        useful_dim: int,
        groups: int,
        local_width: int,
        local_features: int,
        global_width: int,
        global_depth: int,
        output_dim: int,
        activation: str,
    ):
        super().__init__()
        if useful_dim % groups != 0:
            raise ValueError("useful_dim must be divisible by groups")
        self.useful_dim = useful_dim
        self.groups = groups
        self.group_dim = useful_dim // groups
        self.activation = activation
        self.local1 = nn.ModuleList(nn.Linear(self.group_dim, local_width) for _ in range(groups))
        self.local2 = nn.ModuleList(nn.Linear(local_width, local_features) for _ in range(groups))
        dims = [groups * local_features] + [global_width] * global_depth + [output_dim]
        self.global_layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features))
                nn.init.zeros_(module.bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        useful = x[:, : self.useful_dim]
        chunks = useful.split(self.group_dim, dim=1)
        local = []
        for i, chunk in enumerate(chunks):
            h = self._act(self.local1[i](chunk))
            local.append(self._act(self.local2[i](h)))
        h = torch.cat(local, dim=1)
        for layer in self.global_layers[:-1]:
            h = self._act(layer(h))
        return self.global_layers[-1](h)


class TeacherTask:
    def __init__(
        self,
        *,
        teacher: HierarchicalTeacher,
        input_dim: int,
        useful_dim: int,
        noise_std: float,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
    ):
        self.teacher = teacher
        self.input_dim = input_dim
        self.useful_dim = useful_dim
        self.noise_std = noise_std
        self.norm_mean = norm_mean.to(device)
        self.norm_std = norm_std.clamp_min(1e-8).to(device)
        self.device = device

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, self.input_dim, device=self.device)
        y = (self.teacher(x) - self.norm_mean) / self.norm_std
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        return x, y

    @torch.no_grad()
    def make_fixed_set(self, n: int, batch_size: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        remaining = n
        while remaining > 0:
            bsz = min(batch_size, remaining)
            x, y = self.sample(bsz)
            xs.append(x.cpu())
            ys.append(y.cpu())
            remaining -= bsz
        return torch.cat(xs), torch.cat(ys)


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


def build_task(args: argparse.Namespace, device: torch.device) -> TeacherTask:
    set_seed(args.teacher_seed)
    teacher = HierarchicalTeacher(
        useful_dim=args.useful_dim,
        groups=args.teacher_groups,
        local_width=args.teacher_local_width,
        local_features=args.teacher_local_features,
        global_width=args.teacher_global_width,
        global_depth=args.teacher_global_depth,
        output_dim=args.output_dim,
        activation=args.activation,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ys = []
    remaining = args.calibration_samples
    with torch.no_grad():
        while remaining > 0:
            bsz = min(8192, remaining)
            x = torch.randn(bsz, args.input_dim, device=device)
            ys.append(teacher(x))
            remaining -= bsz
    y = torch.cat(ys)
    mean = y.mean(dim=0, keepdim=True)
    std = y.std(dim=0, keepdim=True, unbiased=False)
    print(f"Teacher target std mean={float(std.mean().cpu()):.4f}, min={float(std.min().cpu()):.4f}")
    return TeacherTask(
        teacher=teacher,
        input_dim=args.input_dim,
        useful_dim=args.useful_dim,
        noise_std=args.noise_std,
        norm_mean=mean,
        norm_std=std,
        device=device,
    )


def count_params(cfg: StudentConfig) -> int:
    return ResidualStudentMLP(cfg).parameter_count()


def choose_width(args: argparse.Namespace, blocks: int) -> Tuple[StudentConfig, int]:
    best_cfg, best_params, best_err = None, None, None
    start = max(args.width_multiple, math.ceil(args.min_width / args.width_multiple) * args.width_multiple)
    for width in range(start, args.max_width + 1, args.width_multiple):
        cfg = StudentConfig(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
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
def evaluate(model: nn.Module, x_cpu: torch.Tensor, y_cpu: torch.Tensor, batch_size: int, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, x_cpu.shape[0], batch_size):
        x = x_cpu[start : start + batch_size].to(device)
        y = y_cpu[start : start + batch_size].to(device)
        pred = model(x)
        total += float(F.mse_loss(pred, y, reduction="sum").cpu())
        count += y.numel()
    model.train()
    return total / max(1, count)


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


def cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    t = min(max((step - warmup_steps) / max(1, max_steps - warmup_steps), 0.0), 1.0)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (lr - min_lr)


def train(
    model: ResidualStudentMLP,
    task: TeacherTask,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    train_samples: int,
    device: torch.device,
) -> Dict[str, float]:
    model.to(device)
    opt = configure_optimizer(model, args.lr, args.weight_decay)
    samples_per_step = args.batch_size * args.grad_accum
    planned_steps = math.ceil(train_samples / samples_per_step)
    steps = min(planned_steps, args.max_steps) if args.max_steps is not None else planned_steps
    best_val = float("inf")
    last_train = float("nan")
    t0 = time.time()
    print(f"  training steps={steps} planned_steps={planned_steps} samples_per_step={samples_per_step}")
    for step in range(steps):
        lr_now = cosine_lr(step, steps, args.lr, args.min_lr, args.warmup_steps)
        for group in opt.param_groups:
            group["lr"] = lr_now
        opt.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(args.grad_accum):
            x, y = task.sample(args.batch_size)
            loss = F.mse_loss(model(x), y) / args.grad_accum
            loss.backward()
            accum += float(loss.detach().cpu()) * args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last_train = accum
        if step == 0 or (step + 1) % args.eval_interval == 0 or step == steps - 1:
            val = evaluate(model, x_val, y_val, args.eval_batch_size, device)
            best_val = min(best_val, val)
            print(
                f"    step={step + 1:6d}/{steps} train={last_train:.6f} val={val:.6f} "
                f"lr={lr_now:.2e} elapsed={(time.time() - t0) / 60:.1f}m"
            )
    final_val = evaluate(model, x_val, y_val, args.eval_batch_size, device)
    best_val = min(best_val, final_val)
    return {
        "steps": float(steps),
        "planned_steps": float(planned_steps),
        "samples_per_step": float(samples_per_step),
        "effective_train_samples": float(steps * samples_per_step),
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
    return float((off ** 2).sum().div((mat ** 2).sum().clamp_min(eps)).item())


def compute_input_agop_metrics(
    model: ResidualStudentMLP,
    x_cpu: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    d = args.input_dim
    agop = torch.zeros((d, d), device=device, dtype=torch.float32)
    total = 0
    for start in range(0, x_cpu.shape[0], args.aofe_batch_size):
        base = x_cpu[start : start + args.aofe_batch_size].to(device)
        for _ in range(args.aofe_probes):
            x = base.detach().requires_grad_(True)
            with torch.enable_grad():
                out = model(x)
                probe = torch.randn_like(out)
                scalar = (out * probe).sum()
                grad = torch.autograd.grad(scalar, x, retain_graph=False, create_graph=False)[0]
            g = grad.to(torch.float32)
            agop += g.T @ g
            total += g.shape[0]
    agop /= max(1, total)
    agop = 0.5 * (agop + agop.T)

    useful = agop[: args.useful_dim, : args.useful_dim]
    nuisance = agop[args.useful_dim :, args.useful_dim :]
    cross = agop[: args.useful_dim, args.useful_dim :]
    total_energy = float((agop ** 2).sum().item())
    useful_energy = float((useful ** 2).sum().item())
    nuisance_energy = float((nuisance ** 2).sum().item()) if nuisance.numel() else 0.0
    cross_energy = float(2.0 * (cross ** 2).sum().item()) if cross.numel() else 0.0

    aofe = offdiag_energy(agop)
    ratio = offdiag_ratio(agop)
    model.train()
    return {
        "aofe": aofe,
        "log10_aofe": math.log10(max(aofe, 1e-30)),
        "aofe_ratio": ratio,
        "aofe_total_energy": total_energy,
        "useful_energy_frac": useful_energy / max(total_energy, 1e-30),
        "nuisance_energy_frac": nuisance_energy / max(total_energy, 1e-30),
        "cross_energy_frac": cross_energy / max(total_energy, 1e-30),
    }


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x, y = list(map(float, xs)), list(map(float, ys))
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx, vy = sum((v - mx) ** 2 for v in x), sum((v - my) ** 2 for v in y)
    den = math.sqrt(vx * vy)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den if den else float("nan")


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
        "blocks", "width", "depth_width_ratio", "aofe", "log10_aofe", "aofe_ratio",
        "useful_energy_frac", "nuisance_energy_frac", "cross_energy_frac",
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
    p = argparse.ArgumentParser(description="Hierarchical MLP feature-learning shape scan.")
    p.add_argument("--target-params", type=int, default=3_000_000)
    p.add_argument("--samples-per-param", type=float, default=20.0)
    p.add_argument("--train-samples", type=int, default=None)
    p.add_argument("--layers", type=int, nargs="+", default=[2, 24])
    p.add_argument("--input-dim", type=int, default=512)
    p.add_argument("--useful-dim", type=int, default=128)
    p.add_argument("--output-dim", type=int, default=32)
    p.add_argument("--teacher-groups", type=int, default=16)
    p.add_argument("--teacher-local-width", type=int, default=32)
    p.add_argument("--teacher-local-features", type=int, default=8)
    p.add_argument("--teacher-global-width", type=int, default=256)
    p.add_argument("--teacher-global-depth", type=int, default=6)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--activation", choices=["gelu", "relu", "silu"], default="gelu")
    p.add_argument("--norm", choices=["layer", "none"], default="layer")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--calibration-samples", type=int, default=50000)
    p.add_argument("--val-samples", type=int, default=50000)
    p.add_argument("--aofe-samples", type=int, default=8192)
    p.add_argument("--aofe-batch-size", type=int, default=256)
    p.add_argument("--aofe-probes", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--width-multiple", type=int, default=8)
    p.add_argument("--min-width", type=int, default=32)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--teacher-seed", type=int, default=2028)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("./results_mlp_hierarchical_feature_scan"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--dry-run-configs", action="store_true")
    args = p.parse_args()

    if args.useful_dim > args.input_dim:
        raise ValueError("useful_dim must be <= input_dim")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    configs = [choose_width(args, blocks) for blocks in parse_layers(args.layers)]
    print("Parameter-matched residual MLP configs:")
    for cfg, params in configs:
        print(f"  blocks={cfg.blocks:2d} width={cfg.width:4d} params={params:,} ({100*(params-args.target_params)/args.target_params:+.2f}%)")
    print(f"Device: {device}")
    if args.dry_run_configs:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        payload = vars(args).copy()
        payload["out_dir"] = str(args.out_dir)
        payload["device_resolved"] = str(device)
        json.dump(payload, f, indent=2)

    task = build_task(args, device)
    set_seed(args.seed + 17)
    print(f"Creating validation set: {args.val_samples:,}")
    x_val, y_val = task.make_fixed_set(args.val_samples)
    set_seed(args.seed + 23)
    print(f"Creating AOFE set: {args.aofe_samples:,}")
    x_aofe, _ = task.make_fixed_set(args.aofe_samples)

    rows: List[Dict[str, object]] = []
    for cfg, params in configs:
        print("\n" + "=" * 80)
        print(f"student blocks={cfg.blocks} width={cfg.width} params={params:,}")
        set_seed(args.seed + cfg.blocks)
        model = ResidualStudentMLP(cfg)
        train_samples = args.train_samples or int(round(args.samples_per_param * params))
        stats = train(model, task, x_val, y_val, args, train_samples, device)
        agop = compute_input_agop_metrics(model, x_aofe, args, device)
        row: Dict[str, object] = {
            "blocks": cfg.blocks,
            "width": cfg.width,
            "depth_width_ratio": cfg.blocks / cfg.width,
            "param_count": params,
            "param_error": params - args.target_params,
            "train_sample_budget": train_samples,
            **agop,
            **stats,
        }
        rows.append(row)
        write_csv(args.out_dir / "depth_scan_results.csv", rows)
        with (args.out_dir / "depth_scan_results.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(
            f"  loss={stats['final_val_loss']:.6f} AOFE={agop['aofe']:.3e} "
            f"ratio={agop['aofe_ratio']:.4f} useful_frac={agop['useful_energy_frac']:.4f}"
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