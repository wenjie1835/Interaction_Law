#!/usr/bin/env python3
"""Plot optimal-frontier CE vs WtW AOFE_ratio, projected AGOP AOFE_ratio, or raw AOFE."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _float(x: str | None) -> float | None:
    if x is None or str(x).strip() == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


@dataclass(frozen=True)
class FrontierPack:
    x_wtw_ratio: np.ndarray
    x_aofe_ratio: np.ndarray
    x_aofe: np.ndarray
    test_ce: np.ndarray
    budgets: np.ndarray
    depths: List[int]
    widths: List[int]


def load_frontier(csv_path: Path) -> FrontierPack:
    """For each ``target_n``, smallest ``test_ce`` among rows with CE + WtW + projected ``aofe_ratio``."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_n: Dict[int, Dict[str, str]] = {}
    for r in rows:
        try:
            n = int(float(r["target_n"]))
        except (KeyError, ValueError):
            continue
        te = _float(r.get("test_ce"))
        if te is None or _float(r.get("wtw_aofe_ratio")) is None or _float(r.get("aofe_ratio")) is None:
            continue
        prev = by_n.get(n)
        if prev is None or te < (_float(prev["test_ce"]) or 1e99):
            by_n[n] = r

    ns = sorted(by_n.keys())
    wtw_list: List[float] = []
    ar_list: List[float] = []
    a_list: List[float] = []
    ce_list: List[float] = []
    d_list: List[int] = []
    w_list: List[int] = []
    for n in ns:
        r = by_n[n]
        ce = _float(r["test_ce"])
        wtw = _float(r["wtw_aofe_ratio"])
        ar = _float(r.get("aofe_ratio"))
        ao = _float(r.get("aofe"))
        assert ce is not None and wtw is not None and ar is not None
        if ao is None or not np.isfinite(ao) or ao <= 0:
            raise ValueError(f"Frontier row for N={n} missing positive aofe: {r!r}")
        if not np.isfinite(ar) or ar <= 0:
            raise ValueError(f"Frontier row for N={n} invalid aofe_ratio: {r!r}")
        ce_list.append(ce)
        wtw_list.append(wtw)
        ar_list.append(ar)
        a_list.append(ao)
        d_list.append(int(float(r["depth"])))
        w_list.append(int(float(r["d_model"])))

    return FrontierPack(
        x_wtw_ratio=np.asarray(wtw_list, dtype=np.float64),
        x_aofe_ratio=np.asarray(ar_list, dtype=np.float64),
        x_aofe=np.asarray(a_list, dtype=np.float64),
        test_ce=np.asarray(ce_list, dtype=np.float64),
        budgets=np.asarray(ns, dtype=np.int64),
        depths=d_list,
        widths=w_list,
    )


def fit_powerlaw(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """y ≈ a * x**b ; return (a, b, R^2 on **original CE** scale)."""
    logx = np.log(x)
    logy = np.log(y)
    b, c = np.polyfit(logx, logy, 1)
    a = float(np.exp(c))
    pred = a * (x**b)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, float(b), r2_lin


def _scatter_frontier_plot(
    x: np.ndarray,
    ce: np.ndarray,
    budgets: np.ndarray,
    depths: List[int],
    widths: List[int],
    x_label_plain: str,
    x_fit_symbol: str,
    title: str,
    left_x_log: bool,
    a: float,
    b: float,
    r2: float,
    xf: np.ndarray,
    yf: np.ndarray,
) -> plt.Figure:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def decorate(ax: plt.Axes, xlog: bool, ylog: bool) -> None:
        ax.scatter(x, ce, s=70, color="tab:blue", zorder=3, edgecolors="white", linewidths=0.5)
        ax.plot(
            xf,
            yf,
            "--",
            color="tab:red",
            lw=2.0,
            zorder=2,
            label=r"power law: CE = %.3f $\cdot$ %s$^{%.3f}$" % (a, x_fit_symbol, b),
        )
        for n_t, xv, yi, di, wi in zip(budgets, x, ce, depths, widths):
            ax.annotate(
                f"{n_t/1e6:.1f}M (d={di}, w={wi})",
                (xv, yi),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                alpha=0.9,
            )
        xlab = x_label_plain + (" (log)" if xlog else "")
        ax.set_xlabel(xlab, fontsize=11)
        ylab = "Best test cross-entropy (nats/byte)"
        if ylog:
            ylab += " (log)"
        ax.set_ylabel(ylab, fontsize=11)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.4, which="both" if (xlog or ylog) else "major")
        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")

    ttl_left = (
        f"Semi-log frontier ($R^2={r2:.3f}$, log $x$)"
        if left_x_log
        else f"Power-law fit on frontier ($R^2={r2:.3f}$)"
    )
    ax_left.set_title(ttl_left, fontsize=11)
    decorate(ax_left, xlog=left_x_log, ylog=False)

    ax_right.set_title("Log-log view", fontsize=11)
    decorate(ax_right, xlog=True, ylog=True)

    plt.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "ntp_sweep_merged_by_budget_and_shape.csv",
    )
    ap.add_argument(
        "--plots",
        choices=("all", "both", "wtw", "aofe", "agop_ratio"),
        default="all",
        help=(
            "all: WtW + AGOP AOFE_ratio + raw AOFE; both: WtW + raw AOFE (legacy); "
            "or one of wtw | aofe | agop_ratio."
        ),
    )
    args = ap.parse_args()

    fp = load_frontier(args.csv)
    out_dir = args.csv.resolve().parent
    do_wtw = args.plots in ("all", "both", "wtw")
    do_raw = args.plots in ("all", "both", "aofe")
    do_ar = args.plots in ("all", "agop_ratio")

    if do_wtw:
        path = out_dir / "optimal_loss_vs_wtw_aofe_ratio.png"
        x = fp.x_wtw_ratio
        ce = fp.test_ce
        a, b, r2 = fit_powerlaw(x, ce)
        xf = np.linspace(float(x.min()), float(x.max()), 200)
        yf = a * (xf**b)
        fig = _scatter_frontier_plot(
            x, ce, fp.budgets, fp.depths, fp.widths,
            x_label_plain="Best-shape WtW AOFE_ratio",
            x_fit_symbol=r"\mathrm{WtW}",
            title="Optimal Frontier: Best Loss vs. Best WtW AOFE_ratio",
            left_x_log=False,
            a=a, b=b, r2=r2, xf=xf, yf=yf,
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}  WtW: n={len(x)}, R^2={r2:.6f}, a={a:.6f}, b={b:.6f}")

    if do_ar:
        path = out_dir / "optimal_loss_vs_agop_aofe_ratio.png"
        x = fp.x_aofe_ratio
        ce = fp.test_ce
        a, b, r2 = fit_powerlaw(x, ce)
        xf = np.linspace(float(x.min()), float(x.max()), 200)
        yf = a * (xf**b)
        fig = _scatter_frontier_plot(
            x, ce, fp.budgets, fp.depths, fp.widths,
            x_label_plain="Best-shape AOFE_ratio (64× projected AGOP)",
            x_fit_symbol=r"\mathrm{AOFE_{ratio}}",
            title="Optimal Frontier: Best Loss vs. AOFE_ratio (projected AGOP)",
            left_x_log=False,
            a=a, b=b, r2=r2, xf=xf, yf=yf,
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}  AGOP AOFE_ratio: n={len(x)}, R^2={r2:.6f}, a={a:.6f}, b={b:.6f}")

    if do_raw:
        path = out_dir / "optimal_loss_vs_aofe.png"
        x = fp.x_aofe
        ce = fp.test_ce
        a, b, r2 = fit_powerlaw(x, ce)
        xf = np.geomspace(float(x.min()), float(x.max()), 200)
        yf = a * (xf**b)
        fig = _scatter_frontier_plot(
            x, ce, fp.budgets, fp.depths, fp.widths,
            x_label_plain="AOFE (embedding-channel AGOP, best shape)",
            x_fit_symbol=r"\mathrm{AOFE}",
            title="Optimal Frontier: Best Loss vs. AOFE",
            left_x_log=True,
            a=a, b=b, r2=r2, xf=xf, yf=yf,
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}  AOFE: n={len(x)}, R^2={r2:.6f}, a={a:.6f}, b={b:.6f}")


if __name__ == "__main__":
    main()
