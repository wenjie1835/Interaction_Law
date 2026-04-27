"""
RNN/GRU shape scan with sequential teacher-student feature-learning task.

Purpose
-------
This is the recurrent analogue of the Transformer/MLP/CNN shape scans.
It tests whether a fixed-parameter RNN has an optimal depth-width ratio in a
controlled feature-learning regime, and whether loss relates to AOFE and
AOFE-ratio.

Task
----
A fixed teacher GRU receives only the first useful_dim channels of a high
dimensional input sequence. The remaining channels are nuisance noise. The
student GRU receives the full input sequence and must predict the teacher's
multi-output sequence.

    x_t in R^input_dim
    teacher sees x_t[:useful_dim]
    student sees x_t[:input_dim]
    loss = MSE over all time steps and output channels

Training budget
---------------
The data budget is counted in temporal supervised positions:

    temporal_tokens = num_sequences * sequence_length
    default temporal_tokens = 20 * parameter_count

AOFE
----
For sequence-to-sequence outputs, we estimate an input-channel AGOP using
Hutchinson probes:

    AGOP = E[(J^T r)(J^T r)^T]

where J is the Jacobian of all sequence outputs wrt all sequence inputs. The
gradient is reshaped from [B, T, input_dim] to [B*T, input_dim], so the AGOP is
over input channels, averaged over time and examples.

We also report useful/nuisance/cross AGOP energy fractions.

Examples
--------
Inspect parameter-matched configs:

    python3 rnn_sequence_feature_scan.py --dry-run-configs

Smoke test:

    python3 rnn_sequence_feature_scan.py --target-params 100000 --layers 1 4 --max-steps 20 --val-sequences 512 --aofe-sequences 256

Full-ish run:

    python3 rnn_sequence_feature_scan.py --target-params 3000000 --layers 1 12
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
class RNNConfig:
    input_dim: int = 128
    output_dim: int = 32
    hidden_size: int = 512
    num_layers: int = 3
    cell: str = "gru"
    dropout: float = 0.0
    bias: bool = True


class StudentRNN(nn.Module):
    def __init__(self, cfg: RNNConfig):
        super().__init__()
        self.cfg = cfg
        dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
        if cfg.cell == "gru":
            self.rnn = nn.GRU(
                cfg.input_dim,
                cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=dropout,
                batch_first=True,
                bias=cfg.bias,
            )
        elif cfg.cell == "lstm":
            self.rnn = nn.LSTM(
                cfg.input_dim,
                cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=dropout,
                batch_first=True,
                bias=cfg.bias,
            )
        elif cfg.cell == "rnn_tanh":
            self.rnn = nn.RNN(
                cfg.input_dim,
                cfg.hidden_size,
                num_layers=cfg.num_layers,
                nonlinearity="tanh",
                dropout=dropout,
                batch_first=True,
                bias=cfg.bias,
            )
        else:
            raise ValueError(f"Unknown recurrent cell: {cfg.cell}")
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.out = nn.Linear(cfg.hidden_size, cfg.output_dim, bias=cfg.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.out(self.norm(h))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TeacherGRU(nn.Module):
    def __init__(
        self,
        *,
        useful_dim: int,
        hidden_size: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.useful_dim = useful_dim
        self.rnn = nn.GRU(
            useful_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        useful = x[:, :, : self.useful_dim]
        h, _ = self.rnn(useful)
        return self.out(self.norm(h))


class SequentialTeacherTask:
    def __init__(
        self,
        *,
        teacher: TeacherGRU,
        input_dim: int,
        useful_dim: int,
        seq_len: int,
        ar_coef: float,
        event_prob: float,
        noise_std: float,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
    ):
        self.teacher = teacher
        self.input_dim = input_dim
        self.useful_dim = useful_dim
        self.seq_len = seq_len
        self.ar_coef = ar_coef
        self.event_prob = event_prob
        self.noise_std = noise_std
        self.norm_mean = norm_mean.to(device)
        self.norm_std = norm_std.clamp_min(1e-8).to(device)
        self.device = device

    def sample_inputs(self, batch_size: int) -> torch.Tensor:
        x = torch.randn(batch_size, self.seq_len, self.input_dim, device=self.device)

        # Make useful channels temporally structured so the teacher's recurrent
        # state has meaningful dynamics. Nuisance channels remain iid noise.
        useful = torch.randn(batch_size, self.seq_len, self.useful_dim, device=self.device)
        for t in range(1, self.seq_len):
            useful[:, t] = self.ar_coef * useful[:, t - 1] + math.sqrt(1 - self.ar_coef ** 2) * useful[:, t]

        if self.event_prob > 0:
            events = (torch.rand(batch_size, self.seq_len, 1, device=self.device) < self.event_prob).float()
            event_values = torch.randn(batch_size, self.seq_len, self.useful_dim, device=self.device)
            useful = useful + 1.5 * events * event_values

        x[:, :, : self.useful_dim] = useful
        return x

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.sample_inputs(batch_size)
        y = (self.teacher(x) - self.norm_mean) / self.norm_std
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        return x, y

    @torch.no_grad()
    def make_fixed_set(self, n: int, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
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


def build_task(args: argparse.Namespace, device: torch.device) -> SequentialTeacherTask:
    set_seed(args.teacher_seed)
    teacher = TeacherGRU(
        useful_dim=args.useful_dim,
        hidden_size=args.teacher_hidden_size,
        num_layers=args.teacher_layers,
        output_dim=args.output_dim,
        dropout=args.teacher_dropout,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ys = []
    remaining = args.calibration_sequences
    with torch.no_grad():
        while remaining > 0:
            bsz = min(512, remaining)
            # Use a temporary task-like input sampler before normalization exists.
            x = torch.randn(bsz, args.seq_len, args.input_dim, device=device)
            useful = torch.randn(bsz, args.seq_len, args.useful_dim, device=device)
            for t in range(1, args.seq_len):
                useful[:, t] = args.ar_coef * useful[:, t - 1] + math.sqrt(1 - args.ar_coef ** 2) * useful[:, t]
            if args.event_prob > 0:
                events = (torch.rand(bsz, args.seq_len, 1, device=device) < args.event_prob).float()
                useful = useful + 1.5 * events * torch.randn_like(useful)
            x[:, :, : args.useful_dim] = useful
            ys.append(teacher(x))
            remaining -= bsz
    y = torch.cat(ys, dim=0)
    mean = y.mean(dim=(0, 1), keepdim=True)
    std = y.std(dim=(0, 1), keepdim=True, unbiased=False)
    print(f"Teacher target std mean={float(std.mean().cpu()):.4f}, min={float(std.min().cpu()):.4f}")

    return SequentialTeacherTask(
        teacher=teacher,
        input_dim=args.input_dim,
        useful_dim=args.useful_dim,
        seq_len=args.seq_len,
        ar_coef=args.ar_coef,
        event_prob=args.event_prob,
        noise_std=args.noise_std,
        norm_mean=mean,
        norm_std=std,
        device=device,
    )


def count_params(cfg: RNNConfig) -> int:
    return StudentRNN(cfg).parameter_count()


def choose_width(args: argparse.Namespace, layers: int) -> Tuple[RNNConfig, int]:
    best_cfg, best_params, best_err = None, None, None
    start = max(args.width_multiple, math.ceil(args.min_width / args.width_multiple) * args.width_multiple)
    for width in range(start, args.max_width + 1, args.width_multiple):
        cfg = RNNConfig(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_size=width,
            num_layers=layers,
            cell=args.cell,
            dropout=args.dropout,
            bias=True,
        )
        params = count_params(cfg)
        err = abs(params - args.target_params)
        if best_err is None or err < best_err:
            best_cfg, best_params, best_err = cfg, params, err
    if best_cfg is None or best_params is None:
        raise RuntimeError("No valid hidden size found")
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


def cosine_lr(step: int, steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    t = min(max((step - warmup_steps) / max(1, steps - warmup_steps), 0.0), 1.0)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (lr - min_lr)


def train(
    model: StudentRNN,
    task: SequentialTeacherTask,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    train_sequences: int,
    device: torch.device,
) -> Dict[str, float]:
    model.to(device)
    opt = configure_optimizer(model, args.lr, args.weight_decay)
    seqs_per_step = args.batch_size * args.grad_accum
    planned_steps = math.ceil(train_sequences / seqs_per_step)
    steps = min(planned_steps, args.max_steps) if args.max_steps is not None else planned_steps
    best_val = float("inf")
    last_train = float("nan")
    t0 = time.time()
    print(f"  training steps={steps} planned_steps={planned_steps} sequences_per_step={seqs_per_step}")
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
        "sequences_per_step": float(seqs_per_step),
        "effective_train_sequences": float(steps * seqs_per_step),
        "effective_train_temporal_tokens": float(steps * seqs_per_step * args.seq_len),
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


def compute_input_channel_agop(
    model: StudentRNN,
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
            g = grad.reshape(-1, d).to(torch.float32)
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
        "num_layers",
        "hidden_size",
        "depth_width_ratio",
        "aofe",
        "log10_aofe",
        "aofe_ratio",
        "useful_energy_frac",
        "nuisance_energy_frac",
        "cross_energy_frac",
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
    p = argparse.ArgumentParser(description="RNN/GRU sequential feature-learning shape scan.")
    p.add_argument("--target-params", type=int, default=3_000_000)
    p.add_argument("--temporal-tokens-per-param", type=float, default=20.0)
    p.add_argument("--train-sequences", type=int, default=None)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 12])
    p.add_argument("--cell", choices=["gru", "lstm", "rnn_tanh"], default="gru")
    p.add_argument("--input-dim", type=int, default=128)
    p.add_argument("--useful-dim", type=int, default=24)
    p.add_argument("--output-dim", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--ar-coef", type=float, default=0.85)
    p.add_argument("--event-prob", type=float, default=0.04)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--teacher-hidden-size", type=int, default=192)
    p.add_argument("--teacher-layers", type=int, default=3)
    p.add_argument("--teacher-dropout", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--calibration-sequences", type=int, default=8192)
    p.add_argument("--val-sequences", type=int, default=4096)
    p.add_argument("--aofe-sequences", type=int, default=1024)
    p.add_argument("--aofe-batch-size", type=int, default=32)
    p.add_argument("--aofe-probes", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--width-multiple", type=int, default=8)
    p.add_argument("--min-width", type=int, default=16)
    p.add_argument("--max-width", type=int, default=2048)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--teacher-seed", type=int, default=2029)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("./results_rnn_sequence_feature_scan"))
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

    configs = [choose_width(args, layers) for layers in parse_layers(args.layers)]
    print("Parameter-matched recurrent configs:")
    for cfg, params in configs:
        rel = 100 * (params - args.target_params) / args.target_params
        print(f"  layers={cfg.num_layers:2d} hidden={cfg.hidden_size:4d} params={params:,} ({rel:+.2f}%)")
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
    print(f"Creating validation set: {args.val_sequences:,} sequences")
    x_val, y_val = task.make_fixed_set(args.val_sequences)
    set_seed(args.seed + 23)
    print(f"Creating AOFE set: {args.aofe_sequences:,} sequences")
    x_aofe, _ = task.make_fixed_set(args.aofe_sequences)

    rows: List[Dict[str, object]] = []
    for cfg, params in configs:
        print("\n" + "=" * 80)
        print(f"student layers={cfg.num_layers} hidden={cfg.hidden_size} params={params:,}")
        set_seed(args.seed + cfg.num_layers)
        model = StudentRNN(cfg)
        temporal_tokens = int(round(args.temporal_tokens_per_param * params))
        train_sequences = args.train_sequences or math.ceil(temporal_tokens / args.seq_len)
        if args.train_sequences is not None:
            temporal_tokens = train_sequences * args.seq_len
        stats = train(model, task, x_val, y_val, args, train_sequences, device)
        agop = compute_input_channel_agop(model, x_aofe, args, device)
        row: Dict[str, object] = {
            "num_layers": cfg.num_layers,
            "hidden_size": cfg.hidden_size,
            "depth_width_ratio": cfg.num_layers / cfg.hidden_size,
            "param_count": params,
            "param_error": params - args.target_params,
            "train_temporal_token_budget": temporal_tokens,
            "train_sequence_budget": train_sequences,
            "seq_len": args.seq_len,
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