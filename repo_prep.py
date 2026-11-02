"""
repo_prep.py: SWE-bench Lite dataset preparation and repository checkouts (Step 2A).

Prepares benchmark instances conforming to the conventions established in Step 1 (repo_budget.py):
  * instances.jsonl: one JSON per line with:
      - instance_id (e.g., 'scikit-learn__scikit-learn-12345')
      - repo_dir (path to repository checked out at the base_commit)
      - issue_text (problem_statement from SWE-bench)
      - gold_patch (unified diff patch from SWE-bench)

Features:
  * Reads from local parquet, local jsonl, or downloads directly from Hugging Face.
  * Git caching: clones bare repos once into a cache directory and creates detached worktrees
    per instance at base_commit (fast, minimal disk footprint).
  * Robust filtering by repo name, instance limit, and offline `--skip-clone` mode.
  * Standalone selftest on a synthetic git repository without network calls.

Usage:
  python repo_prep.py prepare --source ./SWE-bench_Lite.parquet --out instances.jsonl --worktree-dir ./swebench_repos
  python repo_prep.py prepare --download --repos scikit-learn/scikit-learn --max-instances 20 --out instances.jsonl
  python repo_prep.py selftest
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set

HF_SWEBENCH_LITE_PARQUET = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/data/test-00000-of-00001.parquet"
)
HF_SWEBENCH_LITE_JSONL = (
    "https://raw.githubusercontent.com/swe-bench/SWE-bench/main/swebench/harness/test.jsonl"
)


def load_dataset_records(source_path_or_url: str) -> List[Dict]:
    """Load SWE-bench records from a local file path or remote URL (supports .parquet, .jsonl, .json)."""
    p = str(source_path_or_url)
    is_url = p.startswith("http://") or p.startswith("https://")

    if is_url:
        sys.stderr.write(f"Downloading dataset from {p} ...\n")
        suffix = ".parquet" if ".parquet" in p else ".jsonl"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        urllib.request.urlretrieve(p, tmp_path)
        file_to_read = tmp_path
        cleanup = True
    else:
        file_to_read = p
        cleanup = False

    records = []
    try:
        if file_to_read.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(file_to_read)
                records = table.to_pylist()
            except ImportError:
                import pandas as pd
                df = pd.read_parquet(file_to_read)
                records = df.to_dict(orient="records")
        elif file_to_read.endswith(".jsonl"):
            with open(file_to_read, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif file_to_read.endswith(".json"):
            with open(file_to_read, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
        else:
            # Fallback attempt as JSONL
            with open(file_to_read, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    finally:
        if cleanup and os.path.exists(file_to_read):
            try:
                os.remove(file_to_read)
            except OSError:
                pass

    return records


def ensure_bare_repo(repo_name: str, cache_dir: Path) -> Path:
    """Ensure a bare clone of https://github.com/{repo_name}.git exists in cache_dir."""
    safe_name = repo_name.replace("/", "__")
    bare_path = cache_dir / f"{safe_name}.git"

    if not bare_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        clone_url = f"https://github.com/{repo_name}.git"
        sys.stderr.write(f"Cloning bare repository: {clone_url} -> {bare_path} ...\n")
        cmd = ["git", "clone", "--bare", clone_url, str(bare_path)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to clone {clone_url}: {res.stderr}")
    else:
        sys.stderr.write(f"Using cached bare repo for {repo_name} at {bare_path}\n")

    return bare_path


def checkout_worktree(bare_path: Path, worktree_dir: Path, instance_id: str, base_commit: str) -> Path:
    """Create a detached git worktree for a specific instance at base_commit."""
    inst_dir = worktree_dir / instance_id
    if inst_dir.exists():
        return inst_dir

    worktree_dir.mkdir(parents=True, exist_ok=True)
    # git worktree add --detach <path> <commit>
    cmd = [
        "git",
        f"--git-dir={bare_path}",
        "worktree",
        "add",
        "--detach",
        "--force",
        str(inst_dir),
        base_commit,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        # Fallback to local clone if worktree fails (e.g. across drives or certain Windows fs setups)
        sys.stderr.write(f"Worktree failed ({res.stderr.strip()}), trying local clone fallback...\n")
        cmd_fallback = ["git", "clone", str(bare_path), str(inst_dir)]
        res_fb = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res_fb.returncode != 0:
            raise RuntimeError(f"Checkout failed for {instance_id}: {res_fb.stderr}")
        subprocess.run(["git", "checkout", base_commit], cwd=str(inst_dir), check=True, stdout=subprocess.DEVNULL)

    return inst_dir


def prepare_instances(
    source: str,
    out_path: str,
    cache_dir: str = "./cache",
    worktree_dir: str = "./swebench_repos",
    repos: Optional[List[str]] = None,
    max_instances: Optional[int] = None,
    skip_clone: bool = False,
) -> int:
    """Prepare SWE-bench instances and write instances.jsonl."""
    raw_records = load_dataset_records(source)
    sys.stderr.write(f"Loaded {len(raw_records)} raw records.\n")

    selected_repos: Optional[Set[str]] = set(repos) if repos else None
    cache_p = Path(cache_dir).resolve()
    worktree_p = Path(worktree_dir).resolve()
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_p, "w", encoding="utf-8") as out_f:
        for rec in raw_records:
            if max_instances and count >= max_instances:
                break

            repo_name = rec.get("repo", "")
            if selected_repos and repo_name not in selected_repos:
                continue

            instance_id = rec.get("instance_id")
            base_commit = rec.get("base_commit")
            problem_statement = rec.get("problem_statement", "")
            gold_patch = rec.get("patch", "")

            if not instance_id or not base_commit:
                continue

            if skip_clone:
                inst_dir = worktree_p / instance_id
                target_repo_dir = str(inst_dir)
            else:
                try:
                    bare_path = ensure_bare_repo(repo_name, cache_p)
                    inst_dir = checkout_worktree(bare_path, worktree_p, instance_id, base_commit)
                    target_repo_dir = str(inst_dir)
                except Exception as e:
                    sys.stderr.write(f"Error preparing {instance_id} ({repo_name}): {e}\n")
                    continue

            instance_record = {
                "instance_id": instance_id,
                "repo_dir": target_repo_dir,
                "issue_text": problem_statement,
                "gold_patch": gold_patch,
            }
            out_f.write(json.dumps(instance_record) + "\n")
            count += 1
            if count % 10 == 0:
                sys.stderr.write(f"Prepared {count} instances...\n")

    sys.stderr.write(f"Done. Wrote {count} instances to {out_p}\n")
    return count


def selftest():
    """Selftest validating repo prep and formatting on a synthetic git repo without network."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo_dir = tmp_p / "fake_repo"
        repo_dir.mkdir()

        # Initialize git repo and make commits
        subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo_dir), check=True)
        subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=str(repo_dir), check=True)

        py_file = repo_dir / "mod.py"
        py_file.write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
        subprocess.run(["git", "commit", "-m", "Commit 1"], cwd=str(repo_dir), check=True, stdout=subprocess.DEVNULL)

        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()

        # Commit 2 (target fix)
        py_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        patch_text = subprocess.check_output(["git", "diff", base_commit], cwd=str(repo_dir), text=True)

        # Create dummy jsonl dataset
        raw_jsonl = tmp_p / "swe_lite_test.jsonl"
        fake_rec = {
            "instance_id": "test__fake_repo-101",
            "repo": "test/fake_repo",
            "base_commit": base_commit,
            "problem_statement": "hello should return world",
            "patch": patch_text,
        }
        raw_jsonl.write_text(json.dumps(fake_rec) + "\n", encoding="utf-8")

        # Create bare clone in cache to simulate download
        cache_dir = tmp_p / "cache"
        cache_dir.mkdir()
        bare_repo = cache_dir / "test__fake_repo.git"
        subprocess.run(["git", "clone", "--bare", str(repo_dir), str(bare_repo)], check=True, stdout=subprocess.DEVNULL)

        out_jsonl = tmp_p / "instances.jsonl"
        worktrees = tmp_p / "worktrees"

        # Run prepare
        n = prepare_instances(
            source=str(raw_jsonl),
            out_path=str(out_jsonl),
            cache_dir=str(cache_dir),
            worktree_dir=str(worktrees),
            repos=["test/fake_repo"],
            max_instances=1,
            skip_clone=False,
        )
        assert n == 1, f"Expected 1 instance prepared, got {n}"
        assert out_jsonl.exists()

        lines = [json.loads(l) for l in out_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        rec = lines[0]
        assert rec["instance_id"] == "test__fake_repo-101"
        assert rec["issue_text"] == "hello should return world"
        assert "diff --git" in rec["gold_patch"]
        assert Path(rec["repo_dir"]).exists()
        assert (Path(rec["repo_dir"]) / "mod.py").read_text(encoding="utf-8") == "def hello():\n    return 'hello'\n"

    print("repo_prep selftest passed: dataset parsing, git worktree checkout, instance schema conformance")


def main():
    parser = argparse.ArgumentParser(description="Prepare SWE-bench Lite instances for repo_budget evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest")

    p = sub.add_parser("prepare")
    p.add_argument("--source", default=HF_SWEBENCH_LITE_PARQUET, help="Path or URL to dataset parquet/jsonl")
    p.add_argument("--download", action="store_true", help="Download official SWE-bench Lite parquet from HF")
    p.add_argument("--out", default="instances.jsonl", help="Output instances.jsonl path")
    p.add_argument("--cache-dir", default="./cache", help="Directory to cache bare repository clones")
    p.add_argument("--worktree-dir", default="./swebench_repos", help="Directory for checked-out instances")
    p.add_argument("--repos", nargs="+", help="Whitelist of repositories (e.g. scikit-learn/scikit-learn)")
    p.add_argument("--max-instances", type=int, help="Maximum instances to extract")
    p.add_argument("--skip-clone", action="store_true", help="Skip git clone/checkout operations")

    args = parser.parse_args()

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "prepare":
        src = HF_SWEBENCH_LITE_PARQUET if args.download else args.source
        prepare_instances(
            source=src,
            out_path=args.out,
            cache_dir=args.cache_dir,
            worktree_dir=args.worktree_dir,
            repos=args.repos,
            max_instances=args.max_instances,
            skip_clone=args.skip_clone,
        )


if __name__ == "__main__":
    main()
