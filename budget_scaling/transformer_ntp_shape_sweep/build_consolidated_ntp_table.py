#!/usr/bin/env python3
"""
Merge NTP shape-sweep CSV shards into one table (per budget × shape: loss, α, AOFE).

Reads:
  - results_ntp_shape_sweep.csv (main)
  - results_ntp_shape_sweep（5M）.csv, （10M）.csv
  - ../results_ntp_10M_parallel/**/results_ntp_shape_sweep.csv (full runs for 10M slice)
  - ../results_transformer_agop_10M_d12/results.csv (teacher–student + embedding AGOP; MSE columns)

Deduplicate by (target_n, depth, d_model), keeping the row with more filled metrics.

Usage:
  python build_consolidated_ntp_table.py
  python build_consolidated_ntp_table.py --out /path/to/ntp_sweep_merged.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
# ROOT = .../Interaction_Law/results/transformer_ntp_shape_sweep
INTERACTION_LAW = ROOT.parents[1]
# Path: Interaction_Law/results_ntp_10M_parallel (sibling to results/ folder).
PARALLEL_10M = INTERACTION_LAW / "results_ntp_10M_parallel"

OUT_FIELDNAMES = [
    "target_n",
    "depth",
    "d_model",
    "n_heads",
    "d_ff",
    "active_n",
    "depth_over_d_model",
    "d_model_over_depth",
    "train_ce",
    "val_ce",
    "test_ce",
    "aofe",
    "log10_aofe",
    "aofe_ratio",
    "wtw_aofe_ratio",
    "steps_run",
    "elapsed_s",
    "param_budget_label",
    # Teacher–student regression (transformer_shape_agop.py); empty for WikiText-NTP rows.
    "train_mse",
    "val_mse",
    "test_mse",
    "row_source",
]

SOURCES: List[Path] = [
    ROOT / "results_ntp_shape_sweep.csv",
    ROOT / "results_ntp_shape_sweep（5M）.csv",
    ROOT / "results_ntp_shape_sweep（10M）.csv",
    # Single-shape full runs under Interaction_Law/ (e.g. N=10M depth=14 only).
    INTERACTION_LAW / "results_ntp_10M_d14" / "results_ntp_shape_sweep.csv",
]

# Optional: frozen-teacher MSE task + embedding-space AGOP (same budget/depth semantics).
TRANSFORMER_AGOP_RESULTS = INTERACTION_LAW / "results_transformer_agop_10M_d12" / "results.csv"


def _completeness(row: Dict[str, str]) -> int:
    src = (row.get("row_source") or "").strip()
    if src == "teacher_student_agop":
        score = 0
        for k in ("train_mse", "val_mse", "test_mse", "steps_run", "aofe_ratio"):
            v = (row.get(k) or "").strip()
            if v:
                score += 1
        return score
    score = 0
    for k in ("train_ce", "val_ce", "test_ce", "steps_run", "elapsed_s"):
        v = (row.get(k) or "").strip()
        if v:
            score += 1
    return score


def _num(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    v = (row.get(key) or "").strip()
    if not v:
        return default
    return float(v)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_teacher_student_agop_rows(path: Path) -> List[Dict[str, str]]:
    """Map transformer_shape_agop ``results.csv`` rows onto NTP-merge schema."""
    if not path.is_file():
        return []
    out: List[Dict[str, str]] = []
    for r in load_csv(path):
        if not r.get("total_params"):
            continue
        out.append({
            "target_n": str(int(float(r["total_params"]))),
            "depth": str(int(float(r["depth"]))),
            "d_model": str(int(float(r["d_model"]))),
            "n_heads": str(int(float(r["n_heads"]))),
            "d_ff": str(int(float(r["d_ff"]))),
            "active_n": str(int(float(r["active_params"]))),
            "train_ce": "",
            "val_ce": "",
            "test_ce": "",
            "aofe": r.get("agop_offdiag_energy", "") or "",
            "aofe_ratio": r.get("agop_offdiag_ratio", "") or "",
            "wtw_aofe_ratio": "",
            "steps_run": str(int(float(r["steps_run"]))) if (r.get("steps_run") or "").strip() else "",
            "elapsed_s": "",
            "train_mse": r.get("train_mse", "") or "",
            "val_mse": r.get("val_mse", "") or "",
            "test_mse": r.get("test_mse", "") or "",
            "row_source": "teacher_student_agop",
        })
    return out


def key_row(r: Dict[str, str]) -> Tuple[int, int, int]:
    return (int(float(r["target_n"])), int(float(r["depth"])), int(float(r["d_model"])))


def merge_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[int, int, int], Dict[str, str]]:
    best: Dict[Tuple[int, int, int], Dict[str, str]] = {}
    for r in rows:
        if not r.get("target_n"):
            continue
        k = key_row(r)
        prev = best.get(k)
        if prev is None or _completeness(r) > _completeness(prev):
            best[k] = r
    return best


def expand(sources: List[Path], parallel_glob: Path) -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    for p in sources:
        if p.is_file():
            all_rows.extend(load_csv(p))
    if parallel_glob.is_dir():
        for p in sorted(parallel_glob.glob("**/results_ntp_shape_sweep.csv")):
            all_rows.extend(load_csv(p))
    all_rows.extend(normalize_teacher_student_agop_rows(TRANSFORMER_AGOP_RESULTS))
    return all_rows


def to_output_row(r: Dict[str, str]) -> Dict[str, Any]:
    d = int(float(r["depth"]))
    w = int(float(r["d_model"]))
    tn = int(float(r["target_n"]))
    aofe = _num(r, "aofe")
    log_a = math.log10(aofe) if aofe == aofe and aofe > 0 else float("nan")
    return {
        "target_n": tn,
        "depth": d,
        "d_model": w,
        "n_heads": int(float(r.get("n_heads") or 0)),
        "d_ff": int(float(r.get("d_ff") or 0)),
        "active_n": int(float(r.get("active_n") or 0)),
        "depth_over_d_model": d / max(w, 1),
        "d_model_over_depth": w / max(d, 1),
        "train_ce": _num(r, "train_ce") if (r.get("train_ce") or "").strip() else None,
        "val_ce": _num(r, "val_ce") if (r.get("val_ce") or "").strip() else None,
        "test_ce": _num(r, "test_ce") if (r.get("test_ce") or "").strip() else None,
        "aofe": aofe if aofe == aofe else None,
        "log10_aofe": log_a if log_a == log_a else None,
        "aofe_ratio": _num(r, "aofe_ratio") if (r.get("aofe_ratio") or "").strip() else None,
        "wtw_aofe_ratio": _num(r, "wtw_aofe_ratio") if (r.get("wtw_aofe_ratio") or "").strip() else None,
        "steps_run": int(float(r["steps_run"])) if (r.get("steps_run") or "").strip() else None,
        "elapsed_s": _num(r, "elapsed_s") if (r.get("elapsed_s") or "").strip() else None,
        "param_budget_label": f"N={tn:,}",
        "train_mse": _num(r, "train_mse") if (r.get("train_mse") or "").strip() else None,
        "val_mse": _num(r, "val_mse") if (r.get("val_mse") or "").strip() else None,
        "test_mse": _num(r, "test_mse") if (r.get("test_mse") or "").strip() else None,
        "row_source": (r.get("row_source") or "").strip() or None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build merged NTP shape sweep table.")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "ntp_sweep_merged_by_budget_and_shape.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    raw = expand(SOURCES, PARALLEL_10M)
    merged = merge_rows(raw)
    keys_sorted = sorted(merged.keys(), key=lambda t: (t[0], t[1], t[2]))
    out_rows = [to_output_row(merged[k]) for k in keys_sorted]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for row in out_rows:
            w.writerow({k: ("" if row.get(k) is None else row[k]) for k in OUT_FIELDNAMES})

    readme = args.out.with_suffix(".列说明.txt")
    readme.write_text(
        "列说明（与 ntp_sweep_merged_by_budget_and_shape.csv 对应）\n"
        "============================================\n"
        "target_n          目标参数量/参数预算 N（与脚本 target_n 一致）\n"
        "depth              Transformer 层数 L（深度）\n"
        "d_model            模型宽度（通道维）\n"
        "n_heads, d_ff      注意力头数与 FFN 维（与主实验一致）\n"
        "active_n           与实现相关的有效/活跃参数量（原表字段）\n"
        "depth_over_d_model 深/宽 = L / d_model（与 CSV 中 alpha 一致）\n"
        "d_model_over_depth 宽/深 = d_model / L\n"
        "train_ce, val_ce, test_ce  各划分交叉熵（主要 loss 可看 val_ce / test_ce）\n"
        "aofe, log10_aofe   输入维 AGOP 非对角能量及其 log10\n"
        "aofe_ratio         非对角 Frobenius 能量占 AGOP 总能量的比例\n"
        "wtw_aofe_ratio     原表中的 wtw 相关 AOFE 比例\n"
        "steps_run, elapsed_s  训练步数与墙钟时间（秒）\n"
        "param_budget_label 人类可读的参数量标签\n"
        "train_mse, val_mse, test_mse  教师-学生回归 MSE（仅 transformer_shape_agop 任务有值）\n"
        "row_source  数据来源：空=WikiText 字节 NTP；teacher_student_agop=transformer_shape_agop.py\n"
        "\n"
        "合并规则：同 (target_n, depth, d_model) 多来源时，保留「字段更完整」的一行；\n"
        "10M：depth 4,5,6,8 以 results_ntp_10M_parallel 下完整运行为准（若存在）；"
        "depth=14（d_model=240）以 results_ntp_10M_d14 为准。\n"
        "results_transformer_agop_10M_d12：N=10M、L=12 的教师-学生形状（CE 列为空，用 MSE 列）。\n"
        f"生成脚本: {Path(__file__).name}\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out} ({len(out_rows)} rows)")
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
