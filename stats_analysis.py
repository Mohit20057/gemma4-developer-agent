"""
stats_analysis.py: Publication-ready statistical analysis for context selection (Step 2C).

Reads results.jsonl (and optional sweep_results.jsonl) and computes rigorous metrics:
  1. Main Comparison Table:
     - Mean & std of func_recall, func_hit, file_recall per method & budget.
     - Paired bootstrap (10,000 resamples) vs baseline (topk_bm25).
     - 95% Confidence Intervals [lo, hi] and empirical p-values.
  2. Stratified Analysis:
     - By repository (parsed from instance_id).
     - By patch complexity: single-file (gold_files == 1) vs multi-file (gold_files > 1).
     - By budget level (accuracy-vs-budget frontier).
  3. Effect Sizes:
     - Paired Cohen's d (mean difference / std of difference) with 95% CI.
     - Cliff's delta (non-parametric ordinal dominance metric) with 95% CI.
  4. Statistical Power Analysis:
     - Required N for 80% power at alpha=0.05 given observed effect size.
     - Assessment of whether evaluation sample size is sufficient.
  5. Sensitivity Sweep Analysis:
     - Ablation of gamma, tau, dmax parameters from sweep_results.jsonl.
  6. Frontier Plot Data:
     - Generates clean machine-readable frontier_data.json for plotting.
  7. Instance-Level Win/Loss Analysis:
     - Top 10 cases where fl_graph outperforms topk_bm25.
     - Top 10 cases where fl_graph underperforms topk_bm25.

Dependencies: Python 3.8+, numpy only. No scipy, no network.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------------
# statistical utilities (numpy-only implementations)
# ----------------------------------------------------------------------------------
def paired_bootstrap_ci(
    a: List[float], b: List[float], n_boot: int = 10000, seed: int = 42
) -> Dict[str, float]:
    """Compute mean difference (b - a) with 95% bootstrap CI and two-sided empirical p-value."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(d)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "p_val": 1.0, "n": 0}

    mean_diff = float(np.mean(d))
    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, n, size=(n_boot, n))
    boot_means = np.mean(d[boot_indices], axis=1)

    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))

    # Empirical p-value under null hypothesis (diff centered at 0)
    # H0: mean(d) == 0 -> shift bootstrap distribution to be zero-centered
    shifted = boot_means - mean_diff
    p_emp = float(np.mean(np.abs(shifted) >= np.abs(mean_diff)))

    return {
        "mean": round(mean_diff, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "p_val": round(max(p_emp, 1.0 / n_boot), 4),
        "n": n,
    }


def compute_cohens_d(a: List[float], b: List[float]) -> Dict[str, float]:
    """Compute paired Cohen's d: mean(diff) / std(diff)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if len(d) < 2:
        return {"d": 0.0, "lo": 0.0, "hi": 0.0}
    std_d = float(np.std(d, ddof=1))
    if std_d < 1e-9:
        d_val = 0.0
    else:
        d_val = float(np.mean(d) / std_d)

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    n = len(d)
    boots = []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        sample = d[idx]
        s = np.std(sample, ddof=1)
        boots.append(float(np.mean(sample) / s) if s > 1e-9 else 0.0)

    return {
        "d": round(d_val, 4),
        "lo": round(float(np.percentile(boots, 2.5)), 4),
        "hi": round(float(np.percentile(boots, 97.5)), 4),
    }


def compute_cliffs_delta(a: List[float], b: List[float]) -> Dict[str, float]:
    """Compute Cliff's delta: non-parametric dominance of b over a."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n_a, n_b = len(arr_a), len(arr_b)
    if n_a == 0 or n_b == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0}

    # Vectorized pairwise comparison
    diff_matrix = arr_b[:, None] - arr_a[None, :]
    concordant = np.sum(diff_matrix > 0)
    discordant = np.sum(diff_matrix < 0)
    delta = float((concordant - discordant) / (n_a * n_b))

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boots = []
    for _ in range(2000):
        idx_a = rng.randint(0, n_a, n_a)
        idx_b = rng.randint(0, n_b, n_b)
        dm = arr_b[idx_b, None] - arr_a[None, idx_a]
        c = np.sum(dm > 0)
        dis = np.sum(dm < 0)
        boots.append(float((c - dis) / (n_a * n_b)))

    return {
        "delta": round(delta, 4),
        "lo": round(float(np.percentile(boots, 2.5)), 4),
        "hi": round(float(np.percentile(boots, 97.5)), 4),
    }


def compute_required_n(cohens_d: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Compute required sample size N for paired t-test using normal approximation."""
    abs_d = abs(cohens_d)
    if abs_d < 1e-4:
        return 100000  # essentially infinite
    # z for two-sided alpha=0.05 is 1.96; z for power=0.80 is 0.8416
    z_alpha = 1.95996
    z_beta = 0.84162
    n_req = math.ceil(((z_alpha + z_beta) / abs_d) ** 2)
    return max(n_req, 8)


def extract_repo_name(instance_id: str) -> str:
    """Extract repository name from SWE-bench instance_id e.g. 'scikit-learn__scikit-learn-12345' -> 'scikit-learn'."""
    if "__" in instance_id:
        return instance_id.split("__")[0]
    return instance_id.split("-")[0]


# ----------------------------------------------------------------------------------
# data loading and aggregation
# ----------------------------------------------------------------------------------
def load_results(results_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("error") is None and "instance_id" in r and "method" in r:
                    rows.append(r)
            except Exception:
                continue
    return rows


def perform_statistical_analysis(
    rows: List[Dict[str, Any]],
    baseline: str = "topk_bm25",
    sweep_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute complete analysis pipeline across all strata."""
    # Group results by (method, budget, instance_id)
    # Average across seeds if multiple seeds exist per instance (e.g. for random)
    data = defaultdict(lambda: defaultdict(list))
    inst_metadata = {}

    budgets_set = set()
    methods_set = set()

    for r in rows:
        inst_id = r["instance_id"]
        method = r["method"]
        budget = int(r["budget"])
        budgets_set.add(budget)
        methods_set.add(method)

        data[(method, budget)][inst_id].append(r)

        if inst_id not in inst_metadata:
            inst_metadata[inst_id] = {
                "gold_files": r.get("gold_files", 1),
                "gold_units": r.get("gold_units", 1),
                "graph_nodes": r.get("graph_nodes", 0),
                "repo": extract_repo_name(inst_id),
            }

    sorted_budgets = sorted(budgets_set)
    sorted_methods = sorted(methods_set)
    if "fl_graph" in sorted_methods:
        sorted_methods.remove("fl_graph")
        sorted_methods.insert(0, "fl_graph")

    # 1. Main comparison table per (method, budget)
    main_table = {}
    effect_sizes = {}
    power_stats = {}

    for budget in sorted_budgets:
        base_inst_map = {
            inst: np.mean([x["func_recall"] for x in data[(baseline, budget)][inst]])
            for inst in data.get((baseline, budget), {})
            if data[(baseline, budget)][inst]
        }
        base_hit_map = {
            inst: np.mean([x["func_hit"] for x in data[(baseline, budget)][inst]])
            for inst in data.get((baseline, budget), {})
            if data[(baseline, budget)][inst]
        }
        base_file_map = {
            inst: np.mean([x["file_recall"] for x in data[(baseline, budget)][inst]])
            for inst in data.get((baseline, budget), {})
            if data[(baseline, budget)][inst]
        }

        for method in sorted_methods:
            method_inst_map = {
                inst: np.mean([x["func_recall"] for x in data[(method, budget)][inst]])
                for inst in data.get((method, budget), {})
                if data[(method, budget)][inst]
            }
            method_hit_map = {
                inst: np.mean([x["func_hit"] for x in data[(method, budget)][inst]])
                for inst in data.get((method, budget), {})
                if data[(method, budget)][inst]
            }
            method_file_map = {
                inst: np.mean([x["file_recall"] for x in data[(method, budget)][inst]])
                for inst in data.get((method, budget), {})
                if data[(method, budget)][inst]
            }

            vals_recall = list(method_inst_map.values())
            vals_hit = list(method_hit_map.values())
            vals_file = list(method_file_map.values())

            m_key = f"{method}@{budget}"
            entry = {
                "method": method,
                "budget": budget,
                "n_instances": len(vals_recall),
                "func_recall_mean": round(float(np.mean(vals_recall)), 4) if vals_recall else 0.0,
                "func_recall_std": round(float(np.std(vals_recall, ddof=1)), 4) if len(vals_recall) > 1 else 0.0,
                "func_hit_mean": round(float(np.mean(vals_hit)), 4) if vals_hit else 0.0,
                "file_recall_mean": round(float(np.mean(vals_file)), 4) if vals_file else 0.0,
            }

            # Paired bootstrap vs baseline
            if method != baseline and base_inst_map and method_inst_map:
                common_instances = sorted(set(base_inst_map.keys()) & set(method_inst_map.keys()))
                a_vals = [base_inst_map[k] for k in common_instances]
                b_vals = [method_inst_map[k] for k in common_instances]

                boot = paired_bootstrap_ci(a_vals, b_vals)
                entry["vs_baseline"] = boot

                # Effect size and power
                cd = compute_cohens_d(a_vals, b_vals)
                c_delta = compute_cliffs_delta(a_vals, b_vals)
                n_req = compute_required_n(cd["d"])

                effect_sizes[m_key] = {"cohens_d": cd, "cliffs_delta": c_delta}
                power_stats[m_key] = {
                    "observed_n": len(common_instances),
                    "required_n_80pct": n_req,
                    "is_powered": len(common_instances) >= n_req,
                }

            main_table[m_key] = entry

    # 2. Stratified analysis: by Repository
    repos = sorted({v["repo"] for v in inst_metadata.values()})
    repo_breakdown = defaultdict(dict)
    for repo in repos:
        repo_instances = [inst for inst, meta in inst_metadata.items() if meta["repo"] == repo]
        for budget in sorted_budgets:
            for method in ("fl_graph", baseline):
                if (method, budget) in data:
                    scores = [
                        np.mean([x["func_recall"] for x in data[(method, budget)][inst]])
                        for inst in repo_instances
                        if inst in data[(method, budget)]
                    ]
                    if scores:
                        repo_breakdown[repo][f"{method}@{budget}"] = {
                            "n": len(scores),
                            "mean": round(float(np.mean(scores)), 4),
                        }

    # 3. Stratified analysis: Single-file vs Multi-file patches
    complexity_breakdown = {"single_file": {}, "multi_file": {}}
    single_insts = [inst for inst, meta in inst_metadata.items() if meta["gold_files"] == 1]
    multi_insts = [inst for inst, meta in inst_metadata.items() if meta["gold_files"] > 1]

    for label, inst_sub in [("single_file", single_insts), ("multi_file", multi_insts)]:
        for budget in sorted_budgets:
            for method in ("fl_graph", baseline):
                if (method, budget) in data:
                    scores = [
                        np.mean([x["func_recall"] for x in data[(method, budget)][inst]])
                        for inst in inst_sub
                        if inst in data[(method, budget)]
                    ]
                    if scores:
                        complexity_breakdown[label][f"{method}@{budget}"] = {
                            "n": len(scores),
                            "mean": round(float(np.mean(scores)), 4),
                        }

    # 4. Frontier Data (curves for plotting)
    frontier_data = {
        "budgets": sorted_budgets,
        "methods": {},
    }
    for method in sorted_methods:
        frontier_data["methods"][method] = {
            "func_recall": [],
            "func_hit": [],
            "file_recall": [],
            "ci_lo": [],
            "ci_hi": [],
        }
        for b in sorted_budgets:
            k = f"{method}@{b}"
            ent = main_table.get(k)
            if ent:
                frontier_data["methods"][method]["func_recall"].append(ent["func_recall_mean"])
                frontier_data["methods"][method]["func_hit"].append(ent["func_hit_mean"])
                frontier_data["methods"][method]["file_recall"].append(ent["file_recall_mean"])
                # CI on mean
                scores = [
                    np.mean([x["func_recall"] for x in data[(method, b)][inst]])
                    for inst in data.get((method, b), {})
                ]
                if len(scores) > 1:
                    rng = np.random.RandomState(42)
                    bm = [float(np.mean(np.array(scores)[rng.randint(0, len(scores), len(scores))])) for _ in range(2000)]
                    frontier_data["methods"][method]["ci_lo"].append(round(float(np.percentile(bm, 2.5)), 4))
                    frontier_data["methods"][method]["ci_hi"].append(round(float(np.percentile(bm, 97.5)), 4))
                else:
                    frontier_data["methods"][method]["ci_lo"].append(ent["func_recall_mean"])
                    frontier_data["methods"][method]["ci_hi"].append(ent["func_recall_mean"])

    # 5. Instance-Level Win / Loss Delta (fl_graph vs topk_bm25 at B=4000 or max budget)
    eval_budget = 4000 if 4000 in sorted_budgets else sorted_budgets[-1]
    instance_deltas = []
    if ("fl_graph", eval_budget) in data and (baseline, eval_budget) in data:
        common = set(data[("fl_graph", eval_budget)].keys()) & set(data[(baseline, eval_budget)].keys())
        for inst in common:
            fl_sc = float(np.mean([x["func_recall"] for x in data[("fl_graph", eval_budget)][inst]]))
            bm_sc = float(np.mean([x["func_recall"] for x in data[(baseline, eval_budget)][inst]]))
            meta = inst_metadata.get(inst, {})
            instance_deltas.append({
                "instance_id": inst,
                "fl_score": round(fl_sc, 4),
                "bm25_score": round(bm_sc, 4),
                "delta": round(fl_sc - bm_sc, 4),
                "gold_files": meta.get("gold_files", 1),
                "gold_units": meta.get("gold_units", 1),
                "graph_nodes": meta.get("graph_nodes", 0),
            })
    instance_deltas.sort(key=lambda x: x["delta"], reverse=True)
    top_wins = instance_deltas[:10]
    top_losses = sorted(instance_deltas, key=lambda x: x["delta"])[:10]

    # 6. Sensitivity Sweep Summary
    sweep_summary = {}
    if sweep_rows:
        sw_groups = defaultdict(list)
        for sr in sweep_rows:
            if sr.get("error") is None:
                param = sr.get("sweep_param", "unknown")
                bud = sr.get("budget", 2000)
                sw_groups[(param, bud)].append(sr["func_recall"])
        for (param, bud), vals in sorted(sw_groups.items()):
            sweep_summary[f"{param}@{bud}"] = {
                "n": len(vals),
                "mean_recall": round(float(np.mean(vals)), 4),
                "std_recall": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
            }

    return {
        "main_table": main_table,
        "effect_sizes": effect_sizes,
        "power_stats": power_stats,
        "repo_breakdown": dict(repo_breakdown),
        "complexity_breakdown": complexity_breakdown,
        "frontier_data": frontier_data,
        "top_wins": top_wins,
        "top_losses": top_losses,
        "sweep_summary": sweep_summary,
        "eval_budget": eval_budget,
        "baseline": baseline,
    }


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Format analysis into publication-quality GitHub-flavored Markdown."""
    lines = []
    lines.append("# Empirical Evaluation & Statistical Analysis")
    lines.append("")
    lines.append("> Grounded benchmark of budgeted code context selection on SWE-bench instances.")
    lines.append("> Testing the novelty thesis: Does facility-location graph coverage beat lexical retrieval under constrained token budgets?")
    lines.append("")

    # Section 1: Main Comparison Table
    lines.append("## 1. Main Method Comparison & Paired Bootstrap")
    lines.append("")
    lines.append("| Method | Budget | N | Func Recall (Mean ± Std) | Func Hit Rate | File Recall | vs. BM25 Baseline Δ [95% CI] | p-value |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    baseline = report["baseline"]
    for k, v in report["main_table"].items():
        meth = v["method"]
        bud = v["budget"]
        n_inst = v["n_instances"]
        f_rec = f"{v['func_recall_mean']:.4f} ± {v['func_recall_std']:.4f}"
        f_hit = f"{v['func_hit_mean']:.4f}"
        f_file = f"{v['file_recall_mean']:.4f}"

        if "vs_baseline" in v:
            boot = v["vs_baseline"]
            diff_str = f"**{boot['mean']:+.4f}** [{boot['lo']:+.4f}, {boot['hi']:+.4f}]"
            pval_str = f"{boot['p_val']:.4f}"
        elif meth == baseline:
            diff_str = "*(baseline)*"
            pval_str = "-"
        else:
            diff_str = "-"
            pval_str = "-"

        lines.append(f"| `{meth}` | {bud} | {n_inst} | {f_rec} | {f_hit} | {f_file} | {diff_str} | {pval_str} |")
    lines.append("")

    # Section 2: Effect Size & Power Analysis
    lines.append("## 2. Effect Sizes & Statistical Power")
    lines.append("")
    lines.append("| Comparison | Budget | Cohen's d [95% CI] | Cliff's δ [95% CI] | Observed N | Req N (80% Power) | Statistically Powered? |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for k, eff in report["effect_sizes"].items():
        cd = eff["cohens_d"]
        c_delta = eff["cliffs_delta"]
        pw = report["power_stats"].get(k, {})
        cd_str = f"{cd['d']:.4f} [{cd['lo']:.4f}, {cd['hi']:.4f}]"
        cdelta_str = f"{c_delta['delta']:.4f} [{c_delta['lo']:.4f}, {c_delta['hi']:.4f}]"
        obs_n = pw.get("observed_n", "-")
        req_n = pw.get("required_n_80pct", "-")
        is_pow = "✅ Yes" if pw.get("is_powered") else "⚠️ Borderline"

        meth_bud = k.split("@")
        lines.append(f"| `{meth_bud[0]}` | {meth_bud[1]} | {cd_str} | {cdelta_str} | {obs_n} | {req_n} | {is_pow} |")
    lines.append("")

    # Section 3: Stratified Analysis (Repository & Complexity)
    lines.append("## 3. Stratified Generalizability Analysis")
    lines.append("")
    lines.append("### 3.1 Breakdown by Repository")
    lines.append("")
    lines.append("| Repository | Budget | `fl_graph` Recall | `topk_bm25` Recall | Advantage |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for repo, data_dict in report["repo_breakdown"].items():
        # inspect budgets
        budgets = sorted({int(k.split("@")[1]) for k in data_dict})
        for b in budgets:
            fl_key = f"fl_graph@{b}"
            bm_key = f"{baseline}@{b}"
            fl_sc = data_dict.get(fl_key, {}).get("mean", 0.0)
            bm_sc = data_dict.get(bm_key, {}).get("mean", 0.0)
            adv = fl_sc - bm_sc
            adv_str = f"**{adv:+.4f}**" if adv > 0 else f"{adv:+.4f}"
            lines.append(f"| `{repo}` | {b} | {fl_sc:.4f} | {bm_sc:.4f} | {adv_str} |")
    lines.append("")

    lines.append("### 3.2 Breakdown by Patch Complexity (Single vs Multi-File)")
    lines.append("")
    lines.append("| Complexity | Budget | `fl_graph` Recall | `topk_bm25` Recall | Advantage |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for comp_label, comp_data in report["complexity_breakdown"].items():
        name = "Single-File Patch (gold_files=1)" if comp_label == "single_file" else "Multi-File Patch (gold_files>1)"
        budgets = sorted({int(k.split("@")[1]) for k in comp_data})
        for b in budgets:
            fl_sc = comp_data.get(f"fl_graph@{b}", {}).get("mean", 0.0)
            bm_sc = comp_data.get(f"{baseline}@{b}", {}).get("mean", 0.0)
            adv = fl_sc - bm_sc
            adv_str = f"**{adv:+.4f}**" if adv > 0 else f"{adv:+.4f}"
            lines.append(f"| {name} | {b} | {fl_sc:.4f} | {bm_sc:.4f} | {adv_str} |")
    lines.append("")

    # Section 4: Budget Frontier
    lines.append("## 4. The Accuracy-versus-Budget Frontier")
    lines.append("")
    lines.append("Under hard token budgets, does graph coverage dominate at low budgets where selection must prioritize coverage over raw word overlap?")
    lines.append("")
    lines.append("| Budget (Tokens) | `fl_graph` | `topk_prior` | `topk_bm25` | `random` | Frontier Gap (fl - bm25) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for b in report["frontier_data"]["budgets"]:
        fl_val = report["main_table"].get(f"fl_graph@{b}", {}).get("func_recall_mean", 0.0)
        pr_val = report["main_table"].get(f"topk_prior@{b}", {}).get("func_recall_mean", 0.0)
        bm_val = report["main_table"].get(f"{baseline}@{b}", {}).get("func_recall_mean", 0.0)
        rd_val = report["main_table"].get(f"random@{b}", {}).get("func_recall_mean", 0.0)
        gap = fl_val - bm_val
        lines.append(f"| {b} | {fl_val:.4f} | {pr_val:.4f} | {bm_val:.4f} | {rd_val:.4f} | **{gap:+.4f}** |")
    lines.append("")

    # Section 5: Top Win / Loss Case Studies
    lines.append(f"## 5. Instance-Level Case Studies (Budget = {report['eval_budget']} tokens)")
    lines.append("")
    lines.append("### 5.1 Top 10 Wins (`fl_graph` > `topk_bm25`)")
    lines.append("")
    lines.append("| Instance ID | Gold Files | Gold Units | Nodes | `fl_graph` | `topk_bm25` | Δ Gain |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for w in report["top_wins"]:
        lines.append(f"| `{w['instance_id']}` | {w['gold_files']} | {w['gold_units']} | {w['graph_nodes']} | {w['fl_score']:.2f} | {w['bm25_score']:.2f} | **{w['delta']:+.2f}** |")
    lines.append("")

    lines.append("### 5.2 Top 10 Losses (`fl_graph` < `topk_bm25`)")
    lines.append("")
    lines.append("| Instance ID | Gold Files | Gold Units | Nodes | `fl_graph` | `topk_bm25` | Δ Loss |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for l in report["top_losses"]:
        lines.append(f"| `{l['instance_id']}` | {l['gold_files']} | {l['gold_units']} | {l['graph_nodes']} | {l['fl_score']:.2f} | {l['bm25_score']:.2f} | **{l['delta']:+.2f}** |")
    lines.append("")

    # Section 6: Sensitivity Sweep if available
    if report["sweep_summary"]:
        lines.append("## 6. Sensitivity Sweep Analysis")
        lines.append("")
        lines.append("| Parameter Setting | Budget | N | Mean Func Recall ± Std |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for k, v in report["sweep_summary"].items():
            lines.append(f"| `{k.split('@')[0]}` | {k.split('@')[1]} | {v['n']} | {v['mean_recall']:.4f} ± {v['std_recall']:.4f} |")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------------------
# self-test on synthetic benchmark data
# ----------------------------------------------------------------------------------
def selftest():
    """Selftest validating statistical estimators, bootstrap CIs, power formulas, and report generation."""
    rng = np.random.RandomState(42)
    n_instances = 50
    budgets = [1000, 2000, 4000, 8000]
    methods = ["fl_graph", "topk_prior", "topk_bm25", "random"]

    rows = []
    for inst_idx in range(n_instances):
        inst_id = f"sklearn__scikit-learn-{inst_idx:04d}" if inst_idx < 30 else f"django__django-{inst_idx:04d}"
        gold_files = 1 if inst_idx % 2 == 0 else 2
        gold_units = rng.randint(1, 4)
        graph_nodes = rng.randint(500, 2500)

        for b in budgets:
            # Simulate frontier: fl_graph has larger advantage at low budget, narrowing at high budget
            base_bm25 = min(0.15 + 0.12 * math.log2(b / 1000 + 1) + rng.normal(0, 0.05), 1.0)
            base_bm25 = max(base_bm25, 0.0)

            # fl_graph adds advantage especially at lower budgets
            fl_bonus = max(0.18 - 0.03 * (b / 1000), 0.02)
            fl_score = min(max(base_bm25 + fl_bonus + rng.normal(0, 0.04), 0.0), 1.0)
            prior_score = min(max(base_bm25 + 0.03 + rng.normal(0, 0.04), 0.0), 1.0)
            rand_score = min(max(0.05 + 0.03 * (b / 1000) + rng.normal(0, 0.02), 0.0), 1.0)

            scores_map = {
                "fl_graph": fl_score,
                "topk_prior": prior_score,
                "topk_bm25": base_bm25,
                "random": rand_score,
            }

            for m in methods:
                rows.append({
                    "instance_id": inst_id,
                    "method": m,
                    "budget": b,
                    "seed": 0,
                    "tokens": int(b * 0.95),
                    "n_units": 20,
                    "file_recall": min(scores_map[m] + 0.2, 1.0),
                    "func_recall": round(scores_map[m], 4),
                    "func_hit": 1.0 if scores_map[m] > 0.3 else 0.0,
                    "gold_files": gold_files,
                    "gold_units": gold_units,
                    "graph_nodes": graph_nodes,
                    "graph_edges": {"call": 100},
                    "utility": 0.5 if m == "fl_graph" else None,
                    "error": None,
                })

    report = perform_statistical_analysis(rows, baseline="topk_bm25")

    # Validations
    assert "fl_graph@1000" in report["main_table"]
    fl_1000 = report["main_table"]["fl_graph@1000"]
    assert fl_1000["func_recall_mean"] > 0
    assert "vs_baseline" in fl_1000

    boot = fl_1000["vs_baseline"]
    assert boot["lo"] <= boot["mean"] <= boot["hi"], (boot["lo"], boot["mean"], boot["hi"])

    eff = report["effect_sizes"]["fl_graph@1000"]
    assert -1.0 <= eff["cliffs_delta"]["delta"] <= 1.0
    assert report["power_stats"]["fl_graph@1000"]["required_n_80pct"] > 0

    assert len(report["frontier_data"]["budgets"]) == 4
    assert len(report["top_wins"]) > 0
    assert len(report["top_losses"]) > 0

    md = generate_markdown_report(report)
    assert "# Empirical Evaluation & Statistical Analysis" in md
    assert "Main Method Comparison" in md

    print("stats_analysis selftest passed: bootstrap CIs, effect sizes, power formulas, markdown formatting")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for code context selection results")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest")

    an = sub.add_parser("analyze")
    an.add_argument("--results", required=True, help="Path to results.jsonl")
    an.add_argument("--sweep", help="Optional path to sweep_results.jsonl")
    an.add_argument("--baseline", default="topk_bm25", help="Baseline method name")
    an.add_argument("--out-md", default="analysis.md", help="Output Markdown report path")
    an.add_argument("--out-json", help="Optional machine-readable JSON output path")
    an.add_argument("--out-frontier", default="frontier_data.json", help="Output frontier plot data JSON")

    args = parser.parse_args()

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "analyze":
        rows = load_results(args.results)
        sweep_rows = load_results(args.sweep) if args.sweep else None

        sys.stderr.write(f"Loaded {len(rows)} results from {args.results}\n")
        report = perform_statistical_analysis(rows, baseline=args.baseline, sweep_rows=sweep_rows)

        # Write markdown
        md_text = generate_markdown_report(report)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md_text)
        sys.stderr.write(f"Wrote analysis report to {args.out_md}\n")

        # Write frontier data
        with open(args.out_frontier, "w", encoding="utf-8") as f:
            json.dump(report["frontier_data"], f, indent=2)
        sys.stderr.write(f"Wrote frontier data to {args.out_frontier}\n")

        # Write JSON if requested
        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            sys.stderr.write(f"Wrote full report JSON to {args.out_json}\n")

        # Print executive summary to stdout
        print(md_text)


if __name__ == "__main__":
    main()
