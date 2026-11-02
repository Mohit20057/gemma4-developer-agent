"""
eval_harness.py: Orchestrated evaluation runner for code context selection (Step 2B).

Executes context selection methods against benchmark instances and tracks publication metrics:
  * Methods: fl_graph (facility-location graph coverage), topk_prior, topk_bm25, random
  * Budgets: 1000, 2000, 4000, 8000 tokens
  * Metrics: tokens, n_units, file_recall, func_recall, func_hit, utility, wall_s
  * Fault-tolerance: records per-instance errors gracefully without crashing the suite
  * Resume-awareness: automatically skips completed (instance_id, method, budget, seed) pairs
  * Sensitivity sweep: sweeps hyper-parameters (gamma, tau, dmax) to evaluate stability

Conventions (aligning with repo_budget.py):
  * Input: instances.jsonl (records with instance_id, repo_dir, issue_text, gold_patch)
  * Output: results.jsonl (one JSON line per evaluation run)
"""

import argparse
import ast
import difflib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Import core algorithms directly from repo_budget
from repo_budget import (
    METHODS,
    CodeGraph,
    Prepared,
    RBConfig,
    build_graph,
    evaluate,
    gold_units,
    prepare,
    select,
    utility,
)


def _git_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def get_provenance() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git": _git_hash() or "unknown",
    }


def load_completed_keys(results_path: Path) -> Set[Tuple[str, str, int, int]]:
    """Return set of (instance_id, method, budget, seed) already recorded in results.jsonl."""
    completed = set()
    if not results_path.exists():
        return completed
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("error") is None and "instance_id" in rec and "method" in rec:
                    completed.add((rec["instance_id"], rec["method"], int(rec["budget"]), int(rec["seed"])))
            except Exception:
                continue
    return completed


def run_evaluation(
    instances_path: str,
    out_path: str,
    budgets: List[int],
    methods: List[str],
    seeds: List[int],
    cfg: RBConfig,
    resume: bool = True,
) -> int:
    """Run full evaluation suite over all instances and append results."""
    inst_p = Path(instances_path).resolve()
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_keys(out_p) if resume else set()
    prov = get_provenance()

    if not inst_p.exists():
        raise FileNotFoundError(f"Instances file not found: {inst_p}")

    total_instances = 0
    with open(inst_p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_instances += 1

    sys.stderr.write(
        f"Starting evaluation: {total_instances} instances, budgets={budgets}, methods={methods}\n"
    )
    if completed:
        sys.stderr.write(f"Resuming: found {len(completed)} existing evaluation records.\n")

    records_written = 0
    instance_idx = 0

    with open(inst_p, "r", encoding="utf-8") as fin, open(out_p, "a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            instance_idx += 1
            inst = json.loads(line)
            inst_id = inst["instance_id"]
            repo_dir = inst["repo_dir"]
            issue_text = inst["issue_text"]
            gold_patch = inst["gold_patch"]

            t0 = time.perf_counter()

            # Verify repo directory
            if not os.path.isdir(repo_dir):
                err_msg = f"Repo directory does not exist: {repo_dir}"
                sys.stderr.write(f"[{instance_idx}/{total_instances}] ERROR {inst_id}: {err_msg}\n")
                err_rec = {
                    "instance_id": inst_id,
                    "error": err_msg,
                    "method": None,
                    "budget": None,
                    "seed": None,
                    "provenance": prov,
                }
                fout.write(json.dumps(err_rec) + "\n")
                fout.flush()
                continue

            try:
                graph = build_graph(repo_dir, cfg)
                prep = prepare(graph, issue_text, cfg)
                gfiles, gunits = gold_units(graph, gold_patch)
                prep_time = time.perf_counter() - t0
            except Exception as e:
                err_msg = f"Graph/Prep error: {e}"
                sys.stderr.write(f"[{instance_idx}/{total_instances}] ERROR {inst_id}: {err_msg}\n")
                err_rec = {
                    "instance_id": inst_id,
                    "error": err_msg,
                    "method": None,
                    "budget": None,
                    "seed": None,
                    "provenance": prov,
                }
                fout.write(json.dumps(err_rec) + "\n")
                fout.flush()
                continue

            n_nodes = len(graph.nodes)
            edge_summary = dict(graph.edge_counts)

            for method in methods:
                method_seeds = seeds if method == "random" else [0]
                for budget in budgets:
                    for seed in method_seeds:
                        key = (inst_id, method, budget, seed)
                        if key in completed:
                            continue

                        t_sel = time.perf_counter()
                        try:
                            chosen_indices = select(prep, budget, method, seed)
                            chosen_ids = [prep.ids[v] for v in chosen_indices]
                            metrics = evaluate(graph, chosen_ids, gfiles, gunits)
                            u_val = (
                                utility(prep, chosen_indices)
                                if method == "fl_graph"
                                else None
                            )
                            wall_s = (time.perf_counter() - t_sel) + (prep_time / max(len(methods) * len(budgets), 1))

                            rec = {
                                "instance_id": inst_id,
                                "method": method,
                                "budget": budget,
                                "seed": seed,
                                "tokens": metrics["tokens"],
                                "n_units": metrics["n_units"],
                                "file_recall": metrics["file_recall"],
                                "func_recall": metrics["func_recall"],
                                "func_hit": metrics["func_hit"],
                                "gold_files": len(gfiles),
                                "gold_units": len(gunits),
                                "graph_nodes": n_nodes,
                                "graph_edges": edge_summary,
                                "utility": u_val,
                                "config": asdict(cfg),
                                "provenance": prov,
                                "wall_s": round(wall_s, 4),
                                "error": None,
                            }
                            fout.write(json.dumps(rec) + "\n")
                            fout.flush()
                            records_written += 1
                        except Exception as e:
                            err_rec = {
                                "instance_id": inst_id,
                                "method": method,
                                "budget": budget,
                                "seed": seed,
                                "error": f"Selection error: {e}",
                                "provenance": prov,
                            }
                            fout.write(json.dumps(err_rec) + "\n")
                            fout.flush()

            sys.stderr.write(
                f"[{instance_idx}/{total_instances}] {inst_id} | nodes={n_nodes} edges={sum(edge_summary.values())} | "
                f"gold_units={len(gunits)} | time={time.perf_counter() - t0:.2f}s\n"
            )

    sys.stderr.write(f"Completed evaluation run. Wrote {records_written} new records to {out_p}\n")
    return records_written


def run_sensitivity_sweep(
    instances_path: str,
    out_path: str,
    budgets: List[int] = [2000, 4000],
    gamma_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    tau_values: List[float] = [0.5, 1.0, 2.0],
    dmax_values: List[float] = [2.0, 3.0, 5.0],
) -> int:
    """Run hyperparameter sensitivity grid for fl_graph against topk_bm25."""
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    prov = get_provenance()

    inst_records = []
    with open(instances_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inst_records.append(json.loads(line))

    sys.stderr.write(
        f"Starting sensitivity sweep over {len(inst_records)} instances...\n"
        f"  gamma: {gamma_values}\n"
        f"  tau: {tau_values}\n"
        f"  dmax: {dmax_values}\n"
        f"  budgets: {budgets}\n"
    )

    count = 0
    with open(out_p, "a", encoding="utf-8") as fout:
        for inst in inst_records:
            inst_id = inst["instance_id"]
            repo_dir = inst["repo_dir"]
            issue_text = inst["issue_text"]
            gold_patch = inst["gold_patch"]

            if not os.path.isdir(repo_dir):
                continue

            try:
                # Baseline graph and bm25
                base_cfg = RBConfig()
                graph = build_graph(repo_dir, base_cfg)
                gfiles, gunits = gold_units(graph, gold_patch)
                base_prep = prepare(graph, issue_text, base_cfg)

                # Record baseline topk_bm25 once
                for b in budgets:
                    bm25_chosen = [base_prep.ids[v] for v in select(base_prep, b, "topk_bm25", 0)]
                    bm25_met = evaluate(graph, bm25_chosen, gfiles, gunits)
                    rec = {
                        "instance_id": inst_id,
                        "method": "topk_bm25",
                        "budget": b,
                        "sweep_param": "baseline",
                        "config": asdict(base_cfg),
                        "func_recall": bm25_met["func_recall"],
                        "func_hit": bm25_met["func_hit"],
                        "file_recall": bm25_met["file_recall"],
                        "provenance": prov,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    count += 1

                # 1. Sweep gamma (holding tau=1.0, dmax=3.0)
                for g_val in gamma_values:
                    cfg_g = RBConfig(gamma=g_val, tau=1.0, dmax=3.0)
                    prep_g = prepare(graph, issue_text, cfg_g)
                    for b in budgets:
                        chosen = [prep_g.ids[v] for v in select(prep_g, b, "fl_graph", 0)]
                        met = evaluate(graph, chosen, gfiles, gunits)
                        rec = {
                            "instance_id": inst_id,
                            "method": "fl_graph",
                            "budget": b,
                            "sweep_param": f"gamma={g_val}",
                            "config": asdict(cfg_g),
                            "func_recall": met["func_recall"],
                            "func_hit": met["func_hit"],
                            "file_recall": met["file_recall"],
                            "provenance": prov,
                        }
                        fout.write(json.dumps(rec) + "\n")
                        count += 1

                # 2. Sweep tau (holding gamma=0.5, dmax=3.0)
                for tau_val in tau_values:
                    if tau_val == 1.0:
                        continue  # already covered in gamma=0.5
                    cfg_tau = RBConfig(gamma=0.5, tau=tau_val, dmax=3.0)
                    prep_tau = prepare(graph, issue_text, cfg_tau)
                    for b in budgets:
                        chosen = [prep_tau.ids[v] for v in select(prep_tau, b, "fl_graph", 0)]
                        met = evaluate(graph, chosen, gfiles, gunits)
                        rec = {
                            "instance_id": inst_id,
                            "method": "fl_graph",
                            "budget": b,
                            "sweep_param": f"tau={tau_val}",
                            "config": asdict(cfg_tau),
                            "func_recall": met["func_recall"],
                            "func_hit": met["func_hit"],
                            "file_recall": met["file_recall"],
                            "provenance": prov,
                        }
                        fout.write(json.dumps(rec) + "\n")
                        count += 1

                # 3. Sweep dmax (holding gamma=0.5, tau=1.0)
                for dmax_val in dmax_values:
                    if dmax_val == 3.0:
                        continue  # already covered
                    cfg_dmax = RBConfig(gamma=0.5, tau=1.0, dmax=dmax_val)
                    prep_dmax = prepare(graph, issue_text, cfg_dmax)
                    for b in budgets:
                        chosen = [prep_dmax.ids[v] for v in select(prep_dmax, b, "fl_graph", 0)]
                        met = evaluate(graph, chosen, gfiles, gunits)
                        rec = {
                            "instance_id": inst_id,
                            "method": "fl_graph",
                            "budget": b,
                            "sweep_param": f"dmax={dmax_val}",
                            "config": asdict(cfg_dmax),
                            "func_recall": met["func_recall"],
                            "func_hit": met["func_hit"],
                            "file_recall": met["file_recall"],
                            "provenance": prov,
                        }
                        fout.write(json.dumps(rec) + "\n")
                        count += 1

                fout.flush()
                sys.stderr.write(f"Swept instance {inst_id}\n")
            except Exception as e:
                sys.stderr.write(f"Sweep error on {inst_id}: {e}\n")

    sys.stderr.write(f"Sensitivity sweep finished. Wrote {count} records to {out_p}\n")
    return count


def selftest():
    """Selftest validating harness orchestration, results format, and resume handling on synthetic data."""
    files = {
        "app/__init__.py": "",
        "app/config.py": (
            "from .parser import parse_header\n\n"
            "def load_config(path):\n"
            "    with open(path) as fh:\n"
            "        first = fh.readline()\n"
            "    return parse_header(first)\n"
        ),
        "app/parser.py": (
            "def parse_header(line):\n"
            "    key, value = line.split(':')\n"
            "    return {key.strip(): value.strip()}\n"
        ),
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo_p = tmp_p / "mini_repo"
        for rel, content in files.items():
            f = repo_p / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")

        # Create diff
        old = files["app/parser.py"]
        new = old.replace("line.split(':')", "line.split(':', 1)")
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(True),
                new.splitlines(True),
                "a/app/parser.py",
                "b/app/parser.py",
            )
        )

        inst_file = tmp_p / "instances.jsonl"
        fake_inst = {
            "instance_id": "test__mini_repo-1",
            "repo_dir": str(repo_p),
            "issue_text": "load_config crashes when colon in header value",
            "gold_patch": diff,
        }
        inst_file.write_text(json.dumps(fake_inst) + "\n", encoding="utf-8")

        out_results = tmp_p / "results.jsonl"
        cfg = RBConfig()

        # Run evaluation
        n_rec = run_evaluation(
            instances_path=str(inst_file),
            out_path=str(out_results),
            budgets=[50, 200],
            methods=["fl_graph", "topk_bm25", "random"],
            seeds=[0, 1],
            cfg=cfg,
            resume=False,
        )
        assert n_rec > 0, "No records written"
        assert out_results.exists()

        rows = [json.loads(l) for l in out_results.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            assert r["instance_id"] == "test__mini_repo-1"
            assert r["method"] in ["fl_graph", "topk_bm25", "random"]
            assert r["tokens"] <= r["budget"] + 1e-9
            assert r["gold_units"] == 1
            assert r["error"] is None
            assert "python" in r["provenance"]

        # Test resume functionality: running again with resume=True should write 0 new records
        n_resume = run_evaluation(
            instances_path=str(inst_file),
            out_path=str(out_results),
            budgets=[50, 200],
            methods=["fl_graph", "topk_bm25", "random"],
            seeds=[0, 1],
            cfg=cfg,
            resume=True,
        )
        assert n_resume == 0, f"Expected 0 new records on resume, got {n_resume}"

        # Test sweep
        sweep_out = tmp_p / "sweep.jsonl"
        n_sweep = run_sensitivity_sweep(
            instances_path=str(inst_file),
            out_path=str(sweep_out),
            budgets=[50],
            gamma_values=[0.0, 0.5],
            tau_values=[1.0],
            dmax_values=[3.0],
        )
        assert n_sweep > 0
        assert sweep_out.exists()

    print("eval_harness selftest passed: evaluation run, resume mechanics, metrics recording, parameter sweep")


def main():
    parser = argparse.ArgumentParser(description="Evaluation harness for budgeted code context selection")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest")

    r = sub.add_parser("run")
    r.add_argument("--instances", required=True, help="Path to instances.jsonl")
    r.add_argument("--out", default="results.jsonl", help="Output results.jsonl path")
    r.add_argument("--budgets", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    r.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--tau", type=float, default=1.0)
    r.add_argument("--dmax", type=float, default=3.0)
    r.add_argument("--beta", type=float, default=0.5)
    r.add_argument("--gamma", type=float, default=0.5)
    r.add_argument("--no-resume", action="store_true", help="Do not skip previously computed combinations")

    sw = sub.add_parser("sweep")
    sw.add_argument("--instances", required=True, help="Path to instances.jsonl")
    sw.add_argument("--out", default="sweep_results.jsonl", help="Output sweep_results.jsonl path")
    sw.add_argument("--budgets", type=int, nargs="+", default=[2000, 4000])
    sw.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    sw.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    sw.add_argument("--dmaxs", type=float, nargs="+", default=[2.0, 3.0, 5.0])

    args = parser.parse_args()

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "run":
        cfg = RBConfig(tau=args.tau, dmax=args.dmax, beta=args.beta, gamma=args.gamma)
        run_evaluation(
            instances_path=args.instances,
            out_path=args.out,
            budgets=args.budgets,
            methods=args.methods,
            seeds=args.seeds,
            cfg=cfg,
            resume=not args.no_resume,
        )
    elif args.cmd == "sweep":
        run_sensitivity_sweep(
            instances_path=args.instances,
            out_path=args.out,
            budgets=args.budgets,
            gamma_values=args.gammas,
            tau_values=args.taus,
            dmax_values=args.dmaxs,
        )


if __name__ == "__main__":
    main()
