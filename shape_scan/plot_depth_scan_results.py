#!/usr/bin/env python3
"""
Visualize depth-scan CSVs from transformer_shape_scan.py, mlp_shape_scan.py,
cnn_shape_scan.py, or rnn_shape_scan.py: loss, AOFE, AOFE-ratio vs depth; optional
fourth panel and scatter figure for W$^\\top$W weighted mean cos$^2$ when CSV
contains wtw_* columns (mlp_agop_shape_scan, transformer_shape_agop, cnn_shape_scan).

Schema is auto-detected from column names.

Usage:
  python plot_depth_scan_results.py \\
    --csv results_tiny_gpt_depth_aofe/depth_scan_results.csv \\
    --out-dir results_tiny_gpt_depth_aofe/figures

  # ``agop_mean_cos2`` columns (Tiny-GPT depth scan) add a 4th depth panel plus
  # ``depth_scan_agop_mean_cos2_vs_metrics.{png,pdf}`` (WtW scans still use ``wtw_*``).

  python plot_depth_scan_results.py \\
    --csv results_mlp_teacher_student_depth_scan/depth_scan_results.csv \\
    --out-dir results_mlp_teacher_student_depth_scan/figures

  # Residual MLP (mlp_shape_scan.py) uses blocks/width in CSV; auto-detected via train_sample_budget:
  python plot_depth_scan_results.py \\
    --csv results_mlp_scan_3M/depth_scan_results.csv \\
    --out-dir results_mlp_scan_3M/figures

  # MLP + AGOP (mlp_agop_shape_scan.py: columns depth, width):
  python plot_depth_scan_results.py \\
    --csv results_mlp_agop_shape_scan/mlp_shape_scan_results.csv \\
    --out-dir results_mlp_agop_shape_scan/figures

  python plot_depth_scan_results.py \\
    --csv results_cnn_autoencoder_depth_scan/depth_scan_results.csv \\
    --out-dir results_cnn_autoencoder_depth_scan/figures

  # Masked-patch CNN shape scan (cnn_shape_scan.py, ~3M params, patch AOFE columns):
  python plot_depth_scan_results.py \\
    --csv results_cnn_scan_3M/depth_scan_results.csv \\
    --out-dir results_cnn_scan_3M/figures

  # Masked-patch feature sweep (includes wtw_stem_mean_cos2 → 4th depth panel + W^T W scatter):
  python plot_depth_scan_results.py \\
    --csv results_cnn_masked_patch_feature_scan/depth_scan_results.csv \\
    --out-dir results_cnn_masked_patch_feature_scan/figures

  # RNN/GRU shape scan (rnn_shape_scan.py, num_layers × hidden_size, input-channel AOFE):
  python plot_depth_scan_results.py \\
    --csv results_rnn_scan_3M/depth_scan_results.csv \\
    --out-dir results_rnn_scan_3M/figures

  # GRU + AGOP shape scan (rnn_agop_shape_scan.py uses columns layers, hidden_size):
  python plot_depth_scan_results.py \\
    --csv results_rnn_agop_shape_scan/rnn_shape_scan_results.csv \\
    --out-dir results_rnn_agop_shape_scan/figures

  # Dry-run table from cnn_shape_scan.py --dry-run-configs (no loss columns):
  python plot_depth_scan_results.py \\
    --csv results_cnn_masked_patch_feature_scan/dry_run_configs.csv \\
    --out-dir results_cnn_masked_patch_feature_scan/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def to_int(row: Dict[str, Any], key: str) -> int:
    """Parse a CSV numeric cell as int (handles strings like \"2.0\")."""
    return int(round(float(row[key])))


def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.shape != b.shape:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else float("nan")


def detect_scan_schema(rows: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    """Return (kind, depth_col, width_col) with kind in {'gpt','mlp','cnn','rnn'}."""
    keys = set(rows[0].keys())
    if "num_layers" in keys and "hidden_size" in keys:
        return "rnn", "num_layers", "hidden_size"
    # rnn_agop_shape_scan.py writes depth as "layers" (not num_layers)
    if "layers" in keys and "hidden_size" in keys:
        return "rnn", "layers", "hidden_size"
    if "hidden_layers" in keys and "hidden_width" in keys:
        return "mlp", "hidden_layers", "hidden_width"
    # Residual MLP (mlp_shape_scan.py) also uses blocks×width; CNN uses the same
    # column names but has train_visual_token_budget instead of train_sample_budget.
    if "blocks" in keys and "width" in keys:
        if "train_sample_budget" in keys:
            return "mlp", "blocks", "width"
        return "cnn", "blocks", "width"
    # mlp_agop_shape_scan.py: columns depth × width
    if "depth" in keys and "width" in keys:
        return "mlp", "depth", "width"
    if "n_layer" in keys and "n_embd" in keys:
        return "gpt", "n_layer", "n_embd"
    raise SystemExit(
        "Unknown CSV schema: need (n_layer, n_embd), (hidden_layers, hidden_width), "
        "(num_layers or layers, hidden_size), (depth, width), or (blocks, width) with MLP vs CNN disambiguation."
    )


def detect_aofe_columns(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Return (aofe_col, ratio_col) for CNN masked-patch vs legacy pixel AOFE."""
    keys = set(rows[0].keys())
    if "patch_aofe" in keys and "patch_aofe_ratio" in keys:
        return "patch_aofe", "patch_aofe_ratio"
    if "aofe" in keys and "aofe_ratio" in keys:
        return "aofe", "aofe_ratio"
    raise SystemExit("CSV missing AOFE columns (expected aofe/aofe_ratio or patch_aofe/patch_aofe_ratio).")


def detect_wtw_cos2_column(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Column from mlp_agop / transformer_shape_agop / cnn_shape_scan if present."""
    keys = set(rows[0].keys())
    for k in (
        "wtw_in_proj_mean_cos2",
        "wtw_head_out_mean_cos2",
        "wtw_stem_mean_cos2",
        "wtw_wte_mean_cos2",
        "wtw_wpe_mean_cos2",
    ):
        if k in keys:
            return k
    # transformer_shape_scan.py (byte LM depth scan)
    if "agop_mean_cos2" in keys:
        return "agop_mean_cos2"
    return None


def wtw_cos2_ylabel(column: str) -> str:
    if column == "wtw_in_proj_mean_cos2":
        return r"W$^\top$W mean cos$^2$ (in_proj)"
    if column == "wtw_head_out_mean_cos2":
        return r"W$^\top$W mean cos$^2$ (head_out)"
    if column == "wtw_stem_mean_cos2":
        return r"W$^\top$W mean cos$^2$ (stem)"
    if column == "wtw_wte_mean_cos2":
        return r"W$^\top$W mean cos$^2$ (wte)"
    if column == "wtw_wpe_mean_cos2":
        return r"W$^\top$W mean cos$^2$ (wpe)"
    if column == "agop_mean_cos2":
        return r"AGOP weighted mean cos$^2$"
    return column.replace("_", " ")


def scatter_cos2_vs_metrics_basenames(metric_col: Optional[str]) -> Tuple[str, str]:
    """PNG/PDF base names for the three-panel scatter (cos2-like vs loss / AOFE / ratio)."""
    if metric_col == "agop_mean_cos2":
        return "depth_scan_agop_mean_cos2_vs_metrics.png", "depth_scan_agop_mean_cos2_vs_metrics.pdf"
    return "depth_scan_wtw_cos2_vs_metrics.png", "depth_scan_wtw_cos2_vs_metrics.pdf"


def scatter_cos2_suptitle(metric_col: Optional[str]) -> str:
    if metric_col == "agop_mean_cos2":
        return (
            r"AGOP weighted mean cos$^2$ vs validation loss / AOFE / ratio"
        )
    return (
        r"W$^\top$W weighted mean cos$^2$ vs validation loss / AOFE / ratio"
    )


def plot_config_only(
    rows: List[Dict[str, Any]],
    *,
    kind: str,
    depth_k: str,
    width_k: str,
    out_dir: Path,
) -> None:
    """Dry-run style table: blocks/width/param_count only (no trained loss)."""
    L = np.array([to_int(r, depth_k) for r in rows])
    C = np.array([to_int(r, width_k) for r in rows])
    P = np.array([to_int(r, "param_count") for r in rows])
    tgt = int(float(rows[0]["target_params"])) if rows and "target_params" in rows[0] else int(np.median(P))

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    axes[0].plot(L, C, "o-", color="#0072B2", lw=1.4, ms=7)
    w_ylabel = "Channel width W" if kind == "cnn" else ("hidden_size H" if kind == "rnn" else "Width")
    axes[0].set_ylabel(w_ylabel)
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].set_title("Param-matched sweep: width chosen vs depth (fixed target params)")

    axes[1].axhline(tgt, color="0.5", ls="--", lw=1.2, label=f"target ≈ {tgt:,}")
    axes[1].plot(L, P, "s-", color="#D55E00", lw=1.4, ms=6)
    axes[1].set_ylabel("Parameter count")
    depth_lbl = "Depth (blocks)" if kind == "cnn" else ("Depth (RNN layers)" if kind == "rnn" else "Depth")
    axes[1].set_xlabel(depth_lbl)
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle=":", alpha=0.7)
    axes[1].set_title("Total params vs depth (should hug target line)")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "dry_run_width_params_vs_depth.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "dry_run_width_params_vs_depth.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(6.8, 4.8))
    err_pct = np.array([float(r.get("param_error_pct", 0.0)) for r in rows])
    b = ax.bar(L.astype(float), err_pct, width=0.65, color="#009E73", edgecolor="0.25", linewidth=0.6)
    ax.axhline(0.0, color="0.4", lw=1.0)
    ax.set_xlabel("Depth (blocks)" if kind == "cnn" else ("Depth (RNN layers)" if kind == "rnn" else "Depth"))
    ax.set_ylabel("Param count vs target (%)")
    ax.set_title("Mismatch from exact target (parameter-matching grid search)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    fig2.tight_layout()
    fig2.savefig(out_dir / "dry_run_param_error_pct.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig2.savefig(out_dir / "dry_run_param_error_pct.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot depth scan CSV from transformer / MLP / CNN shape scans"
    )
    parser.add_argument("--csv", type=Path, required=True, help="depth_scan_results.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Figure output directory (default: csv parent / figures)")
    parser.add_argument("--loss-key", choices=("final_val_loss", "best_val_loss"), default="final_val_loss")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("Empty CSV")

    kind, depth_k, width_k = detect_scan_schema(rows)
    keys0 = set(rows[0].keys())

    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.loss_key not in keys0:
        if "param_count" not in keys0:
            raise SystemExit(f"CSV has no {args.loss_key} and no param_count; cannot plot.")
        plot_config_only(rows, kind=kind, depth_k=depth_k, width_k=width_k, out_dir=out_dir)
        summary_path = out_dir / "depth_scan_summary.txt"
        summary_path.write_text(
            "\n".join(
                [
                    f"CSV: {args.csv.resolve()}",
                    "Mode: dry-run / config table only (no validation loss in file).",
                    "",
                    f"Figures: width & param sweep vs depth, param error %.",
                    f"Output dir: {out_dir.resolve()}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Config-only CSV: wrote figures to {out_dir.resolve()}")
        print(f"Wrote summary   {summary_path}")
        return

    L = np.array([to_int(r, depth_k) for r in rows])
    C = np.array([to_int(r, width_k) for r in rows])
    loss = np.array([to_float(r, args.loss_key) for r in rows])
    aofe_k, ratio_k = detect_aofe_columns(rows)
    aofe = np.array([to_float(r, aofe_k) for r in rows])
    ratio = np.array([to_float(r, ratio_k) for r in rows])
    wtw_k = detect_wtw_cos2_column(rows)
    wtw = np.array([to_float(r, wtw_k) for r in rows]) if wtw_k is not None else None

    # Depth/width aspect ratios (both common conventions)
    depth_over_width = L / np.maximum(C, 1)  # "深/宽"
    width_over_depth = C / np.maximum(L, 1)  # "宽/深"

    i_best = int(np.argmin(loss))
    best = rows[i_best]

    if kind == "mlp":
        if depth_k == "blocks":
            depth_xlabel = "Depth (residual MLP blocks)"
            cbar_depth_label = "blocks"
            ann = lambda i: f"B{L[i]}×W{C[i]}"

            def xl_depth_over_width() -> str:
                return r"Blocks/width ratio  ($B / W$)"

            def xl_width_over_depth() -> str:
                return r"Width/blocks ratio  ($W / B$)"

            if "param_count" in rows[0]:
                pm = int(np.median([to_int(r, "param_count") for r in rows]))
                aspect_title = f"Residual MLP: blocks/width (median ~{pm:,} params)"
            else:
                aspect_title = "Residual MLP: blocks vs width"
        elif depth_k == "depth":
            depth_xlabel = "Depth (residual MLP blocks)"
            cbar_depth_label = "depth"
            ann = lambda i: f"D{L[i]}×W{C[i]}"

            def xl_depth_over_width() -> str:
                return r"Depth/width ratio  ($D / W$)"

            def xl_width_over_depth() -> str:
                return r"Width/depth ratio  ($W / D$)"

            if "param_count" in rows[0]:
                pm = int(np.median([to_int(r, "param_count") for r in rows]))
                aspect_title = f"Residual MLP (AGOP): depth/width (median ~{pm:,} params)"
            else:
                aspect_title = "Residual MLP: depth vs width"
        else:
            depth_xlabel = "Depth (hidden layers)"
            cbar_depth_label = "hidden_layers"
            ann = lambda i: f"L{L[i]}×W{C[i]}"

            def xl_depth_over_width() -> str:
                return r"Depth/width ratio  ($L_{\mathrm{hid}} / W_{\mathrm{hid}}$)"

            def xl_width_over_depth() -> str:
                return r"Width/depth ratio  ($W_{\mathrm{hid}} / L_{\mathrm{hid}}$)"

            if "param_count" in rows[0]:
                pm = int(np.median([to_int(r, "param_count") for r in rows]))
                aspect_title = f"Find shape by depth/width (median ~{pm:,} trainable params)"
            else:
                aspect_title = "Find shape by depth/width ratio"
    elif kind == "cnn":
        depth_xlabel = "Depth (residual blocks)"
        cbar_depth_label = "blocks"
        ann = lambda i: f"B{L[i]}×W{C[i]}"

        def xl_depth_over_width() -> str:
            return r"Blocks/width ratio  ($B / W_{\mathrm{ch}}$)"

        def xl_width_over_depth() -> str:
            return r"Width/blocks ratio  ($W_{\mathrm{ch}} / B$)"

        if "param_count" in rows[0]:
            pm = int(np.median([to_int(r, "param_count") for r in rows]))
            task_hint = "masked-patch CNN" if aofe_k == "patch_aofe" else "CNN autoencoder"
            aspect_title = f"{task_hint}: shape by blocks/width (median ~{pm:,} params)"
        else:
            aspect_title = "CNN: blocks vs channel width"
    elif kind == "rnn":
        depth_xlabel = "Depth (RNN layers)"
        cbar_depth_label = depth_k.replace("_", " ")
        ann = lambda i: f"L{L[i]}×H{C[i]}"

        def xl_depth_over_width() -> str:
            return r"Depth/width ratio  ($L / H_{\mathrm{hid}}$)"

        def xl_width_over_depth() -> str:
            return r"Width/depth ratio  ($H_{\mathrm{hid}} / L$)"

        if "param_count" in rows[0]:
            pm = int(np.median([to_int(r, "param_count") for r in rows]))
            aspect_title = f"RNN/GRU: layers vs hidden (median ~{pm:,} params)"
        else:
            aspect_title = "RNN: depth vs hidden size"
    else:
        depth_xlabel = "Depth (n_layer)"
        cbar_depth_label = "n_layer"
        ann = lambda i: f"L{L[i]}×C{C[i]}"

        def xl_depth_over_width() -> str:
            return r"Depth/width ratio  ($n_{\mathrm{layer}} / n_{\mathrm{embd}}$)"

        def xl_width_over_depth() -> str:
            return r"Width/depth ratio  ($n_{\mathrm{embd}} / n_{\mathrm{layer}}$)"

        aspect_title = "Find shape by depth/width ratio (fixed ~3M params)"

    x = L
    # --- Figure 1: loss, AOFE, ratio [, W^T W cos2] vs depth ---
    n_panels = 4 if wtw_k is not None else 3
    fig1, axes1 = plt.subplots(n_panels, 1, figsize=(7.5, 4.0 + 2.6 * n_panels), sharex=True)
    axes = np.atleast_1d(axes1).ravel().tolist()

    c_loss = "#0072B2"
    c_aofe = "#D55E00"
    c_ar = "#009E73"

    axes[0].plot(x, loss, "o-", color=c_loss, lw=1.4, ms=7)
    axes[0].scatter([L[i_best]], [loss[i_best]], s=140, c="none", edgecolors="#E69F00", linewidths=2.5, zorder=5)
    axes[0].set_ylabel(args.loss_key.replace("_", " "))
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].set_title("Validation loss vs depth (orange ring = lowest loss in scan)")

    axes[1].plot(x, aofe, "s-", color=c_aofe, lw=1.4, ms=6)
    axes[1].set_ylabel("AOFE" if aofe_k == "aofe" else "patch AOFE")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.7)
    axes[1].set_title("AOFE (off-diagonal Frobenius energy) vs depth")

    axes[2].plot(x, ratio, "^-", color=c_ar, lw=1.4, ms=6)
    axes[2].set_ylabel("AOFE-ratio" if ratio_k == "aofe_ratio" else "patch AOFE-ratio")
    axes[2].set_ylim(0, max(1.0, float(ratio.max()) * 1.05))
    axes[2].grid(True, linestyle=":", alpha=0.7)
    axes[2].set_title("AOFE-ratio vs depth")

    if wtw_k is not None:
        assert wtw is not None
        c_wtw = "#CC79A7"
        axes[3].plot(x, wtw, "D-", color=c_wtw, lw=1.4, ms=5)
        axes[3].scatter([L[i_best]], [wtw[i_best]], s=140, c="none", edgecolors="#E69F00", linewidths=2.5, zorder=5)
        axes[3].set_ylabel(wtw_cos2_ylabel(wtw_k))
        axes[3].set_ylim(0, max(1.0, float(wtw.max()) * 1.05))
        axes[3].grid(True, linestyle=":", alpha=0.7)
        axes[3].set_title(
            (
                r"AGOP weighted mean cos$^2$ vs depth (ring = lowest loss)"
                if wtw_k == "agop_mean_cos2"
                else r"W$^\top$W weighted mean cos$^2$ vs depth (ring = lowest loss)"
            )
        )
        axes[3].set_xlabel(depth_xlabel)
    else:
        axes[2].set_xlabel(depth_xlabel)

    fig1.tight_layout()
    p1 = out_dir / "depth_scan_loss_aofe_vs_depth.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
    fig1.savefig(out_dir / "depth_scan_loss_aofe_vs_depth.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    # --- W^T W cos2 vs loss / AOFE / ratio (scatter, color = depth) ---
    if wtw_k is not None and wtw is not None:
        fig_w, axs = plt.subplots(1, 3, figsize=(12.0, 4.0), sharex=False)
        y_specs = [
            (loss, args.loss_key.replace("_", " "), "linear"),
            (aofe, "AOFE" if aofe_k == "aofe" else "patch AOFE", "log"),
            (ratio, "AOFE-ratio" if ratio_k == "aofe_ratio" else "patch AOFE-ratio", "linear"),
        ]
        cbar_sc = None
        for ax, (yv, ylab, yscale) in zip(axs, y_specs):
            cbar_sc = ax.scatter(
                wtw, yv, c=L, cmap="viridis", s=72, edgecolors="0.3", linewidths=0.55, vmin=L.min(), vmax=L.max()
            )
            if yscale == "log":
                ax.set_yscale("log")
            for i in range(len(rows)):
                ax.annotate(
                    ann(i),
                    (wtw[i], yv[i]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=7,
                    alpha=0.85,
                )
            ax.scatter(
                [wtw[i_best]],
                [yv[i_best]],
                s=185,
                facecolors="none",
                edgecolors="#E69F00",
                linewidths=2.3,
                zorder=6,
            )
            ax.set_xlabel(wtw_cos2_ylabel(wtw_k))
            ax.set_ylabel(ylab)
            ax.grid(True, linestyle=":", alpha=0.7)
        if cbar_sc is not None:
            fig_w.colorbar(cbar_sc, ax=axs[2], shrink=0.88).set_label(f"{cbar_depth_label} (depth)")
        fig_w.suptitle(scatter_cos2_suptitle(wtw_k), y=1.02, fontsize=11)
        fig_w.tight_layout()
        fn_png, fn_pdf = scatter_cos2_vs_metrics_basenames(wtw_k)
        fig_w.savefig(out_dir / fn_png, dpi=200, bbox_inches="tight", facecolor="white")
        fig_w.savefig(out_dir / fn_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig_w)

    # --- Figure 2: aspect ratio vs loss ---
    fig2, ax = plt.subplots(figsize=(6.8, 5))
    sc = ax.scatter(depth_over_width, loss, c=L, cmap="viridis", s=80, edgecolors="0.3", linewidths=0.6)
    cbar = fig2.colorbar(sc, ax=ax)
    cbar.set_label(f"{cbar_depth_label} (depth)")
    for i, r in enumerate(rows):
        ax.annotate(
            ann(i),
            (depth_over_width[i], loss[i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            alpha=0.85,
        )
    ax.scatter(
        [depth_over_width[i_best]],
        [loss[i_best]],
        s=200,
        facecolors="none",
        edgecolors="#E69F00",
        linewidths=2.5,
        zorder=6,
        label=f"Lowest {args.loss_key}",
    )
    ax.set_xlabel(xl_depth_over_width())
    ax.set_ylabel(args.loss_key.replace("_", " "))
    ax.set_title(aspect_title)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(out_dir / "depth_scan_loss_vs_depth_over_width.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig2.savefig(out_dir / "depth_scan_loss_vs_depth_over_width.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    # --- Figure 3: width/depth vs loss (alternative aspect view) ---
    fig3, ax = plt.subplots(figsize=(6.8, 5))
    sc = ax.scatter(width_over_depth, loss, c=L, cmap="viridis", s=80, edgecolors="0.3", linewidths=0.6)
    fig3.colorbar(sc, ax=ax).set_label(f"{cbar_depth_label} (depth)")
    for i in range(len(rows)):
        ax.annotate(ann(i), (width_over_depth[i], loss[i]), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.85)
    ax.scatter(
        [width_over_depth[i_best]],
        [loss[i_best]],
        s=200,
        facecolors="none",
        edgecolors="#E69F00",
        linewidths=2.5,
        zorder=6,
    )
    ax.set_xlabel(xl_width_over_depth())
    ax.set_ylabel(args.loss_key.replace("_", " "))
    ax.set_title("Same scan: width/depth vs loss")
    ax.grid(True, linestyle=":", alpha=0.7)
    fig3.tight_layout()
    fig3.savefig(out_dir / "depth_scan_loss_vs_width_over_depth.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig3.savefig(out_dir / "depth_scan_loss_vs_width_over_depth.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig3)

    # --- Summary text ---
    summary_path = out_dir / "depth_scan_summary.txt"
    d_b, w_b = to_int(best, depth_k), to_int(best, width_k)
    lines = [
        f"CSV: {args.csv.resolve()}",
        f"Schema: {kind}",
        f"Loss column: {args.loss_key}",
        "",
    ]
    if kind == "mlp":
        if "param_count" in best:
            pc_m = to_int(best, "param_count")
        else:
            pc_m = best.get("param_count", "")
        if depth_k == "blocks":
            lines.append(f"Lowest loss row: blocks={d_b}, width={w_b}, param_count={pc_m}")
        elif depth_k == "depth":
            lines.append(f"Lowest loss row: depth={d_b}, width={w_b}, param_count={pc_m}")
        else:
            lines.append(f"Lowest loss row: hidden_layers={d_b}, hidden_width={w_b}, param_count={pc_m}")
    elif kind == "cnn":
        pc = best.get("param_count", "")
        lines.append(
            f"Lowest loss row: blocks={best[depth_k]}, width={best[width_k]}, param_count={pc}"
        )
    elif kind == "rnn":
        if "param_count" in best:
            pc = to_int(best, "param_count")
        else:
            pc = ""
        lines.append(f"Lowest loss row: {depth_k}={d_b}, hidden_size={w_b}, param_count={pc}")
    else:
        n_head = best.get("n_head", "N/A")
        lines.append(
            f"Lowest loss row: n_layer={best['n_layer']}, n_embd={best['n_embd']}, n_head={n_head}"
        )
    lines.extend(
        [
            f"  depth/width (L/W) = {d_b / max(w_b, 1):.6f}",
            f"  width/depth (W/L) = {w_b / max(d_b, 1):.4f}",
            f"  {args.loss_key} = {to_float(best, args.loss_key):.6f}",
            f"  {aofe_k} = {to_float(best, aofe_k):.6e}, {ratio_k} = {to_float(best, ratio_k):.6f}",
        ]
    )
    if wtw_k is not None and wtw is not None:
        if "wtw_wte_mean_cos2" in best and "wtw_wpe_mean_cos2" in best:
            lines.append(f"  wtw_wte_mean_cos2 = {to_float(best, 'wtw_wte_mean_cos2'):.6f}")
            lines.append(f"  wtw_wpe_mean_cos2 = {to_float(best, 'wtw_wpe_mean_cos2'):.6f}")
            wpe_arr = np.array([to_float(r, "wtw_wpe_mean_cos2") for r in rows])
            lines.extend(
                [
                    "",
                    "Pearson r (W^T W mean cos2 vs metrics):",
                    f"  wte vs {args.loss_key}: {pearson_np(wtw, loss):.4f}",
                    f"  wte vs {aofe_k}: {pearson_np(wtw, aofe):.4f}",
                    f"  wte vs {ratio_k}: {pearson_np(wtw, ratio):.4f}",
                    f"  wpe vs {args.loss_key}: {pearson_np(wpe_arr, loss):.4f}",
                    f"  wpe vs {aofe_k}: {pearson_np(wpe_arr, aofe):.4f}",
                    f"  wpe vs {ratio_k}: {pearson_np(wpe_arr, ratio):.4f}",
                ]
            )
        else:
            lines.append(f"  {wtw_k} = {to_float(best, wtw_k):.6f}")
            pear_hdr = (
                "Pearson r (AGOP mean cos² vs metrics):"
                if wtw_k == "agop_mean_cos2"
                else "Pearson r (W^T W mean cos2 vs metrics):"
            )
            lines.extend(
                [
                    "",
                    pear_hdr,
                    f"  vs {args.loss_key}: {pearson_np(wtw, loss):.4f}",
                    f"  vs {aofe_k}: {pearson_np(wtw, aofe):.4f}",
                    f"  vs {ratio_k}: {pearson_np(wtw, ratio):.4f}",
                ]
            )
    lines.extend(["", "Sorted by loss (best first):"])
    order = np.argsort(loss)
    for j in order:
        r = rows[int(j)]
        d_i, w_i = to_int(r, depth_k), to_int(r, width_k)
        if "wtw_wte_mean_cos2" in r and "wtw_wpe_mean_cos2" in r:
            wtw_tail = (
                f"  wte_cos2={to_float(r, 'wtw_wte_mean_cos2'):.4f}"
                f"  wpe_cos2={to_float(r, 'wtw_wpe_mean_cos2'):.4f}"
            )
        else:
            wtw_tail = f"  {wtw_k}={to_float(r, wtw_k):.4f}" if wtw_k is not None else ""
        if kind == "mlp":
            if depth_k == "blocks":
                lines.append(
                    f"  B={d_i:>2} W={w_i:>4}  B/W={d_i / max(w_i, 1):.5f}  "
                    f"{args.loss_key}={to_float(r, args.loss_key):.6f}{wtw_tail}"
                )
            elif depth_k == "depth":
                lines.append(
                    f"  D={d_i:>2} W={w_i:>4}  D/W={d_i / max(w_i, 1):.5f}  "
                    f"{args.loss_key}={to_float(r, args.loss_key):.6f}{wtw_tail}"
                )
            else:
                lines.append(
                    f"  L={d_i:>2} W={w_i:>4}  L/W={d_i / max(w_i, 1):.5f}  "
                    f"{args.loss_key}={to_float(r, args.loss_key):.6f}{wtw_tail}"
                )
        elif kind == "cnn":
            lines.append(
                f"  B={d_i:>2} W={w_i:>4}  B/W={d_i / max(w_i, 1):.5f}  "
                f"{args.loss_key}={to_float(r, args.loss_key):.6f}{wtw_tail}"
            )
        elif kind == "rnn":
            lines.append(
                f"  L={d_i:>2} H={w_i:>4}  L/H={d_i / max(w_i, 1):.5f}  "
                f"{args.loss_key}={to_float(r, args.loss_key):.6f}{wtw_tail}"
            )
        else:
            lines.append(
                f"  L={d_i:>2} C={w_i:>3}  L/C={d_i / max(w_i, 1):.5f}  "
                f"{args.loss_key}={to_float(r, args.loss_key):.6f}{wtw_tail}"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote figures to {out_dir.resolve()}")
    print(f"Wrote summary   {summary_path}")
    try:
        idx_sort = lines.index("Sorted by loss (best first):")
    except ValueError:
        idx_sort = len(lines)
    for ln in lines[4:min(idx_sort, 15)]:
        print(ln)
    if wtw_k is not None and wtw is not None:
        fn_png, _ = scatter_cos2_vs_metrics_basenames(wtw_k)
        lbl = "AGOP mean cos²" if wtw_k == "agop_mean_cos2" else "W^T W cos²"
        print(
            f"Pearson ({lbl} vs {args.loss_key}, {aofe_k}, {ratio_k}): "
            f"{pearson_np(wtw, loss):.4f}, {pearson_np(wtw, aofe):.4f}, {pearson_np(wtw, ratio):.4f}"
        )
        print(f"Also: {fn_png} (+ .pdf)")


if __name__ == "__main__":
    main()
