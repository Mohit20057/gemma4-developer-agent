"""
repo_budget.py: budgeted, topology-aware context selection for local coding agents (Step 1).

Research question (the novelty claim must be earned by experiments, not asserted):
  Under a hard context budget B (tokens), does choosing code units by their GRAPH COVERAGE of the
  likely edit locations beat choosing them by lexical/semantic relevance alone, and how does the
  accuracy-vs-budget frontier of small local models look?

Formal objective (facility location on the code graph):
  G = (V, E) code graph. pi(u) = estimated probability that unit u is an edit target
  (BM25 prior diffused by personalised PageRank). s(u, v) = exp(-d(u, v) / tau), d = truncated
  shortest-path distance with edge-type lengths (calls, shared self-attributes, overrides, same
  class / file hubs); for v != u it is divided by 1 + gamma*log2(1 + degree(v)) so that generic
  hub functions, which are close to everything, do not look informative about any one target.
  For a selection X with token cost c(X) <= B:

        U(X) = sum_u pi(u) * max_{v in X} s(u, v)

  U is monotone submodular (a facility-location function), so the better of the unit-cost greedy and
  the cost-benefit greedy has a constant-factor guarantee under the knapsack constraint
  (about 0.316 of optimal; Leskovec et al., CELF, KDD 2007; verify the citation before using it).
  The guarantee covers U only; whether high U means better localization is an empirical question.
  Top-k-by-relevance is the modular special case with s(u, v) = 1[u == v].

Experiment conventions used by every later step:
  * instances.jsonl: one JSON per line with instance_id, repo_dir (checked out at the base commit),
    issue_text, gold_patch. From Hugging Face SWE-bench: problem_statement -> issue_text,
    patch -> gold_patch.
  * results.jsonl: one record per (instance_id, method, budget, seed) with metrics, config, provenance.
  * Comparisons use a paired bootstrap over instances.

Commands:
  python repo_budget.py selftest
  python repo_budget.py run --instances instances.jsonl --out results.jsonl --budgets 1000 2000 4000 8000
  python repo_budget.py summarize --results results.jsonl --baseline topk_bm25

Dependencies: Python 3.8+, numpy. Nothing else; no network.
"""
import argparse
import ast
import difflib
import heapq
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SKIP_DIRS = {".git", ".hg", ".svn", "venv", ".venv", "env", "node_modules", "build", "dist",
             "__pycache__", ".tox", "site-packages", "docs", "doc"}
STOP = set("a an and are as at be but by for from has have if in into is it its not of on or that "
           "the this to was were will with when which while you your we our can do does did should "
           "would could so then than there these those get got use used using".split())


# ----------------------------------------------------------------------------------
# configuration and records
# ----------------------------------------------------------------------------------
@dataclass
class RBConfig:
    tau: float = 1.0           # proximity decay: s = exp(-d / tau)
    dmax: float = 3.0          # truncate shortest paths beyond this length
    beta: float = 0.5          # weight of the PageRank-diffused prior (0 = BM25 only)
    restart: float = 0.3       # PageRank restart probability
    ppr_iters: int = 20
    kappa: float = 8.0         # sharpness of the BM25 -> probability map
    top_targets: int = 200     # targets kept when building the coverage function
    gamma: float = 0.5         # specificity penalty: s(u, v) is divided by 1 + gamma*log2(1 + degree(v)) for v != u
    include_tests: bool = False
    chars_per_token: float = 3.5


@dataclass
class Node:
    id: str
    path: str
    qualname: str
    start: int
    end: int
    cost: int
    tokens: list = field(default_factory=list, repr=False)


class CodeGraph:
    def __init__(self):
        self.nodes = {}                       # candidate units: functions and methods
        self.adj = defaultdict(dict)          # includes hub nodes; undirected; value = edge length
        self.edge_counts = Counter()
        self.parse_errors = 0

    def add_edge(self, u, v, length, kind):
        if u == v:
            return
        if v not in self.adj[u] or length < self.adj[u][v]:
            self.adj[u][v] = length
            self.adj[v][u] = length
        self.edge_counts[kind] += 1

    def has_edge(self, u, v):
        return v in self.adj.get(u, {})


# ----------------------------------------------------------------------------------
# tokenisation and BM25
# ----------------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokenize(text):
    out = []
    for w in _WORD.findall(text):
        parts = [p.lower() for p in _CAMEL.findall(w.replace("_", " "))]
        parts = [p for p in parts if len(p) > 1 and p not in STOP]
        out.extend(parts)
        if len(parts) > 1:
            out.append(w.lower())
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.n = len(docs)
        self.k1, self.b = k1, b
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = max(float(self.len.mean()), 1.0) if self.n else 1.0
        self.post = defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.post[t].append((i, f))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.post.items()}

    def scores(self, query_tokens):
        out = np.zeros(self.n)
        for t, qc in Counter(query_tokens).items():
            if t not in self.post:
                continue
            w = self.idf[t] * (1.0 + math.log(qc))
            for i, f in self.post[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                out[i] += w * f * (self.k1 + 1) / denom
        return out


# ----------------------------------------------------------------------------------
# graph construction (Python sources, standard library only)
# ----------------------------------------------------------------------------------
def iter_py_files(root, include_tests=False):
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        rel = Path(dirpath).relative_to(root)
        if not include_tests and any(p in ("test", "tests", "testing") for p in rel.parts):
            continue
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            if not include_tests and (f.startswith("test_") or f.endswith("_test.py")
                                      or f == "conftest.py"):
                continue
            yield Path(dirpath) / f


def _module_parts(rel):
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _call_targets(fn_node):
    """Yield ('name', n), ('self', n), ('attr', n) or ('modattr', base, n) for calls inside fn_node."""
    for n in ast.walk(fn_node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name):
            yield ("name", f.id)
        elif isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name):
                if f.value.id in ("self", "cls"):
                    yield ("self", f.attr)
                else:
                    yield ("modattr", f.value.id, f.attr)
            else:
                yield ("attr", f.attr)


def _attr_usage(fn_node):
    writes, reads = set(), set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            if isinstance(n.ctx, ast.Store):
                writes.add(n.attr)
            elif isinstance(n.ctx, ast.Load):
                reads.add(n.attr)
    return writes, reads


def build_graph(root, cfg=None):
    cfg = cfg or RBConfig()
    root = Path(root)
    g = CodeGraph()
    files = {}
    for fp in iter_py_files(root, cfg.include_tests):
        rel = fp.relative_to(root).as_posix()
        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
            files[rel] = (ast.parse(src), src.splitlines())
        except (SyntaxError, ValueError, OSError):
            g.parse_errors += 1

    # module index with every dotted suffix so src/ layouts still resolve
    mod_index = defaultdict(set)
    for rel in files:
        parts = _module_parts(rel)
        for i in range(len(parts)):
            mod_index[".".join(parts[i:])].add(rel)

    per_fn = {}                       # node id -> (calls, writes, reads, class_key, name, rel)
    by_name = defaultdict(list)
    module_funcs = {}
    class_methods = defaultdict(dict)  # (rel, cls_qual) -> {name: id}
    class_by_name = defaultdict(list)
    class_bases = {}
    imports = {}

    def collect(rel, lines, body, cls_stack):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = ".".join(cls_stack + [n.name])
                nid = f"{rel}::{qual}"
                if nid in g.nodes:
                    nid = f"{nid}#{n.lineno}"
                start = min([n.lineno] + [d.lineno for d in n.decorator_list])
                end = getattr(n, "end_lineno", None) or n.lineno
                text = "\n".join(lines[start - 1:end])
                cost = int(math.ceil(len(text) / cfg.chars_per_token)) + 8
                doc = ast.get_docstring(n) or ""
                toks = (tokenize(n.name) * 3 + tokenize(qual) + tokenize(rel) + tokenize(doc)
                        + tokenize(text))
                g.nodes[nid] = Node(nid, rel, qual, start, end, cost, toks)
                w, r = _attr_usage(n)
                cls_key = (rel, ".".join(cls_stack)) if cls_stack else None
                per_fn[nid] = (list(_call_targets(n)), w, r, cls_key, n.name, rel)
                by_name[n.name].append(nid)
                if cls_key:
                    class_methods[cls_key][n.name] = nid
                else:
                    module_funcs[(rel, n.name)] = nid
            elif isinstance(n, ast.ClassDef):
                key = (rel, ".".join(cls_stack + [n.name]))
                class_by_name[n.name].append(key)
                class_bases[key] = [b.id if isinstance(b, ast.Name)
                                    else b.attr if isinstance(b, ast.Attribute) else ""
                                    for b in n.bases]
                collect(rel, lines, n.body, cls_stack + [n.name])
            elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for attr in ("body", "orelse", "finalbody"):
                    collect(rel, lines, getattr(n, attr, []) or [], cls_stack)
                for h in getattr(n, "handlers", []) or []:
                    collect(rel, lines, h.body, cls_stack)

    for rel, (tree, lines) in files.items():
        collect(rel, lines, tree.body, [])
        parts = _module_parts(rel)
        pkg = parts if rel.endswith("__init__.py") else parts[:-1]
        imp = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    imp[a.asname or a.name.split(".")[0]] = (a.name if a.asname else a.name.split(".")[0], None)
            elif isinstance(n, ast.ImportFrom):
                base = pkg[:max(len(pkg) - (n.level - 1), 0)] if n.level else []
                mod = ".".join(base + (n.module.split(".") if n.module else []))
                for a in n.names:
                    imp[a.asname or a.name] = (mod, a.name)
        imports[rel] = imp

    # hubs: functions -> class hub -> file hub keep siblings close but not adjacent
    for nid, node in g.nodes.items():
        cls_key = per_fn[nid][3]
        file_hub = f"FILE::{node.path}"
        if cls_key:
            hub = f"CLASS::{cls_key[0]}::{cls_key[1]}"
            g.add_edge(nid, hub, 0.75, "class_hub")
            g.add_edge(hub, file_hub, 0.5, "file_hub")
        else:
            g.add_edge(nid, file_hub, 1.25, "file_hub")

    def methods_of(cls_key, name, seen=None):
        seen = seen or set()
        if cls_key in seen:
            return []
        seen.add(cls_key)
        if name in class_methods.get(cls_key, {}):
            return [class_methods[cls_key][name]]
        out = []
        for b in class_bases.get(cls_key, []):
            for ck in class_by_name.get(b, []):
                out += methods_of(ck, name, seen)
        return out

    def resolve(call, rel, cls_key):
        kind = call[0]
        if kind == "self":
            hits = methods_of(cls_key, call[1]) if cls_key else []
            if hits:
                return hits
            return [i for i in by_name.get(call[1], []) if per_fn[i][3]][:5]
        if kind == "name":
            name = call[1]
            if (rel, name) in module_funcs:
                return [module_funcs[(rel, name)]]
            if name in imports.get(rel, {}):
                mod, orig = imports[rel][name]
                for target_rel in mod_index.get(mod, ()):
                    if orig and (target_rel, orig) in module_funcs:
                        return [module_funcs[(target_rel, orig)]]
                    for ck in class_by_name.get(orig or name, []):
                        if ck[0] == target_rel and "__init__" in class_methods.get(ck, {}):
                            return [class_methods[ck]["__init__"]]
            for ck in class_by_name.get(name, []):
                if "__init__" in class_methods.get(ck, {}):
                    return [class_methods[ck]["__init__"]]
            cands = [i for i in by_name.get(name, []) if not per_fn[i][3]]
            return cands if 0 < len(cands) <= 3 else []
        if kind == "modattr":
            base, name = call[1], call[2]
            if base in imports.get(rel, {}):
                mod, orig = imports[rel][base]
                full = mod if orig is None else f"{mod}.{orig}" if mod else orig
                for target_rel in set(mod_index.get(full, ())) | set(mod_index.get(mod, ())):
                    if (target_rel, name) in module_funcs:
                        return [module_funcs[(target_rel, name)]]
            cands = [i for i in by_name.get(name, []) if per_fn[i][3]]
            return cands if 0 < len(cands) <= 5 else []
        cands = [i for i in by_name.get(call[1], []) if per_fn[i][3]]
        return cands if 0 < len(cands) <= 5 else []

    for nid, (calls, writes, reads, cls_key, name, rel) in per_fn.items():
        for call in set(calls):
            hits = resolve(call, rel, cls_key)
            for h in hits:
                g.add_edge(nid, h, 1.0 + 0.5 * math.log2(len(hits)), "call")

    # def-use through instance attributes: writers <-> readers inside one class
    by_class = defaultdict(list)
    for nid, (_, writes, reads, cls_key, name, _) in per_fn.items():
        if cls_key:
            by_class[cls_key].append((nid, writes, reads - set(class_methods[cls_key])))
    for cls_key, members in by_class.items():
        attrs = defaultdict(lambda: ([], []))
        for nid, writes, reads in members:
            for a in writes:
                attrs[a][0].append(nid)
            for a in reads:
                attrs[a][1].append(nid)
        for a, (ws, rs) in attrs.items():
            if len(ws) * len(rs) > 200:
                continue
            for w in ws:
                for r in rs:
                    g.add_edge(w, r, 1.0, "shares_attr")

    # overrides: same method name in a class and one of its (name-resolved) bases
    for cls_key, bases in class_bases.items():
        for name, nid in class_methods.get(cls_key, {}).items():
            for b in bases:
                for ck in class_by_name.get(b, []):
                    if name in class_methods.get(ck, {}):
                        g.add_edge(nid, class_methods[ck][name], 0.5, "override")
    return g


# ----------------------------------------------------------------------------------
# prior, proximity, selection
# ----------------------------------------------------------------------------------
def issue_tokens(issue_text):
    toks = tokenize(issue_text)
    for span in re.findall(r"`([^`]+)`", issue_text):
        toks += tokenize(span) * 2
    for path in re.findall(r"[\w./-]+\.py", issue_text):
        toks += tokenize(path) * 2
    return toks


def diffuse(graph, p0, alpha, iters):
    x = dict(p0)
    for _ in range(iters):
        nx = defaultdict(float)
        for u, mass in x.items():
            nbrs = graph.adj.get(u)
            if not nbrs:
                nx[u] += mass
                continue
            tot = sum(1.0 / l for l in nbrs.values())
            for v, l in nbrs.items():
                nx[v] += (1 - alpha) * mass * (1.0 / l) / tot
        for u, m in p0.items():
            nx[u] += alpha * m
        x = nx
    return x


def proximity_from(graph, src, dmax, tau, candidates):
    dist = {src: 0.0}
    heap = [(0.0, src)]
    out = {}
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if u in candidates:
            out[u] = math.exp(-d / tau)
        for v, l in graph.adj.get(u, {}).items():
            nd = d + l
            if nd <= dmax and nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return out


@dataclass
class Prepared:
    ids: list
    costs: np.ndarray
    p_bm25: np.ndarray
    pi: np.ndarray
    targets: list            # indices into ids
    pi_t: np.ndarray         # normalised prior over targets
    cover: dict              # candidate index -> [(target position, proximity)]


def prepare(graph, issue_text, cfg=None):
    cfg = cfg or RBConfig()
    ids = list(graph.nodes)
    n = len(ids)
    if n == 0:
        return Prepared([], np.zeros(0), np.zeros(0), np.zeros(0), [], np.zeros(0), {})
    bm = BM25([graph.nodes[i].tokens for i in ids])
    s = bm.scores(issue_tokens(issue_text))
    smax = float(s.max())
    w = np.exp(cfg.kappa * s / smax) if smax > 0 else np.ones(n)
    p0 = w / w.sum()
    if cfg.beta > 0:
        spread = diffuse(graph, {ids[i]: float(p0[i]) for i in range(n) if p0[i] > 1e-6},
                         cfg.restart, cfg.ppr_iters)
        q = np.array([spread.get(i, 0.0) for i in ids])
        q = q / q.sum() if q.sum() > 0 else p0
        pi = (1 - cfg.beta) * p0 + cfg.beta * q
    else:
        pi = p0
    order = np.argsort(-pi)[:cfg.top_targets]
    pi_t = pi[order] / pi[order].sum()
    index = {nid: i for i, nid in enumerate(ids)}
    cand = set(ids)
    deg = {i: sum(1 for nb in graph.adj.get(i, {}) if not nb.startswith(("FILE::", "CLASS::")))
           for i in ids}
    cover = defaultdict(list)
    for pos, ti in enumerate(order):
        for v, sv in proximity_from(graph, ids[ti], cfg.dmax, cfg.tau, cand).items():
            if v != ids[ti]:
                sv = sv / (1.0 + cfg.gamma * math.log2(1 + deg[v]))
            cover[index[v]].append((pos, sv))
    costs = np.array([graph.nodes[i].cost for i in ids], dtype=float)
    return Prepared(ids, costs, p0, pi, [int(t) for t in order], pi_t, dict(cover))


def utility(prep, selected):
    best = np.zeros(len(prep.pi_t))
    for v in selected:
        for pos, sv in prep.cover.get(v, ()):
            if sv > best[pos]:
                best[pos] = sv
    return float((prep.pi_t * best).sum())


def greedy(prep, budget, ratio=True):
    """Lazy greedy (CELF) on the facility-location utility; stops when no feasible gain remains."""
    cur = np.zeros(len(prep.pi_t))
    chosen, spent = [], 0.0

    def gain(v):
        return sum(prep.pi_t[pos] * max(0.0, sv - cur[pos]) for pos, sv in prep.cover[v])

    heap = []
    for v in prep.cover:
        if prep.costs[v] <= budget:
            g0 = gain(v)
            if g0 > 0:
                heapq.heappush(heap, (-(g0 / prep.costs[v] if ratio else g0), v, 0))
    while heap:
        negp, v, stamp = heapq.heappop(heap)
        if prep.costs[v] > budget - spent:
            continue
        if stamp == len(chosen):
            chosen.append(v)
            spent += prep.costs[v]
            for pos, sv in prep.cover[v]:
                if sv > cur[pos]:
                    cur[pos] = sv
        else:
            g1 = gain(v)
            if g1 > 1e-15:
                heapq.heappush(heap, (-(g1 / prep.costs[v] if ratio else g1), v, len(chosen)))
    return chosen


def select_fl(prep, budget):
    a, b = greedy(prep, budget, True), greedy(prep, budget, False)
    return a if utility(prep, a) >= utility(prep, b) else b


def select_topk(prep, budget, key="pi"):
    score = prep.pi if key == "pi" else prep.p_bm25
    chosen, spent = [], 0.0
    for v in np.argsort(-score):
        if spent + prep.costs[v] <= budget:
            chosen.append(int(v))
            spent += prep.costs[v]
    return chosen


def select_random(prep, budget, seed=0):
    rng = random.Random(seed)
    order = list(range(len(prep.ids)))
    rng.shuffle(order)
    chosen, spent = [], 0.0
    for v in order:
        if spent + prep.costs[v] <= budget:
            chosen.append(v)
            spent += prep.costs[v]
    return chosen


METHODS = ("fl_graph", "topk_prior", "topk_bm25", "random")


def select(prep, budget, method, seed=0):
    if method == "fl_graph":
        return select_fl(prep, budget)
    if method == "topk_prior":
        return select_topk(prep, budget, "pi")
    if method == "topk_bm25":
        return select_topk(prep, budget, "bm25")
    if method == "random":
        return select_random(prep, budget, seed)
    raise ValueError(f"unknown method {method}")


# ----------------------------------------------------------------------------------
# gold patch -> target units, and metrics
# ----------------------------------------------------------------------------------
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def patch_touched_lines(diff_text):
    """{path: [old-file line numbers touched]} from a unified diff (insertions map to the nearest old line)."""
    touched = defaultdict(list)
    path, old = None, 0
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            p = line[4:].split("\t")[0].strip()
            path = None if p == "/dev/null" else (p[2:] if p.startswith("a/") else p)
        elif line.startswith("+++ "):
            if path is None:
                p = line[4:].split("\t")[0].strip()
                path = p[2:] if p.startswith("b/") else p
        elif line.startswith("@@"):
            m = _HUNK.match(line)
            if m:
                old = int(m.group(1))
        elif path and line.startswith("-") and not line.startswith("---"):
            touched[path].append(old)
            old += 1
        elif path and line.startswith("+") and not line.startswith("+++"):
            touched[path].append(max(old - 1, 1))
            touched[path].append(old)
        elif path and line.startswith(" "):
            old += 1
    return dict(touched)


def gold_units(graph, diff_text):
    """(gold files, gold unit ids). Lines outside any function count toward files only."""
    touched = patch_touched_lines(diff_text)
    by_file = defaultdict(list)
    for node in graph.nodes.values():
        by_file[node.path].append(node)
    files, units = set(touched), set()
    for path, lines in touched.items():
        for ln in lines:
            hits = [n for n in by_file.get(path, ()) if n.start <= ln <= n.end]
            if hits:
                units.add(min(hits, key=lambda n: n.end - n.start).id)
    return files, units


def evaluate(graph, selected_ids, gold_files, gold_unit_ids):
    sel = set(selected_ids)
    sel_files = {graph.nodes[i].path for i in sel}
    return {
        "tokens": int(sum(graph.nodes[i].cost for i in sel)),
        "n_units": len(sel),
        "file_recall": (len(sel_files & gold_files) / len(gold_files)) if gold_files else None,
        "func_recall": (len(sel & gold_unit_ids) / len(gold_unit_ids)) if gold_unit_ids else None,
        "func_hit": (1.0 if sel & gold_unit_ids else 0.0) if gold_unit_ids else None,
    }


# ----------------------------------------------------------------------------------
# paired bootstrap and runners
# ----------------------------------------------------------------------------------
def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """Mean difference (b - a) over instances with a 95% bootstrap interval."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.RandomState(seed)
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"mean": round(float(d.mean()), 4), "lo": round(float(np.percentile(boots, 2.5)), 4),
            "hi": round(float(np.percentile(boots, 97.5)), 4), "n": int(len(d))}


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return None


def run_instances(instances_path, out_path, budgets, methods, seeds, cfg):
    prov = {"python": platform.python_version(), "numpy": np.__version__, "git": _git_hash()}
    n = 0
    with open(instances_path, encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            inst = json.loads(line)
            graph = build_graph(inst["repo_dir"], cfg)
            prep = prepare(graph, inst["issue_text"], cfg)
            gfiles, gunits = gold_units(graph, inst["gold_patch"])
            for method in methods:
                for budget in budgets:
                    for seed in (seeds if method == "random" else [0]):
                        chosen = [prep.ids[v] for v in select(prep, budget, method, seed)]
                        rec = {"instance_id": inst["instance_id"], "method": method,
                               "budget": budget, "seed": seed, "gold_files": len(gfiles),
                               "gold_units": len(gunits), "graph_nodes": len(graph.nodes),
                               "graph_edges": dict(graph.edge_counts),
                               "utility": utility(prep, [prep.ids.index(c) for c in chosen])
                               if method == "fl_graph" else None,
                               "config": asdict(cfg), "provenance": prov}
                        rec.update(evaluate(graph, chosen, gfiles, gunits))
                        fout.write(json.dumps(rec) + "\n")
                        n += 1
    return n


def summarize(results_path, baseline, metric="func_recall"):
    rows = [json.loads(l) for l in open(results_path, encoding="utf-8") if l.strip()]
    table = defaultdict(dict)
    for r in rows:
        if r.get(metric) is not None:
            table[(r["method"], r["budget"])].setdefault(r["instance_id"], []).append(r[metric])
    out = {}
    for (method, budget), per in sorted(table.items()):
        mean_by_inst = {k: float(np.mean(v)) for k, v in per.items()}
        entry = {"n": len(per), "mean": round(float(np.mean(list(mean_by_inst.values()))), 4)}
        base = table.get((baseline, budget))
        if base and method != baseline:
            common = sorted(set(mean_by_inst) & set(base))
            entry["vs_" + baseline] = paired_bootstrap([np.mean(base[k]) for k in common],
                                                       [mean_by_inst[k] for k in common])
        out[f"{method}@{budget}"] = entry
    return {"metric": metric, "table": out}


# ----------------------------------------------------------------------------------
# self-test on a synthetic repository
# ----------------------------------------------------------------------------------
def _naive_greedy(prep, budget, ratio):
    cur = np.zeros(len(prep.pi_t))
    chosen, spent, left = [], 0.0, set(prep.cover)
    while True:
        best, bv = 0.0, None
        for v in sorted(left):
            if prep.costs[v] > budget - spent:
                continue
            g = sum(prep.pi_t[p] * max(0.0, s - cur[p]) for p, s in prep.cover[v])
            score = g / prep.costs[v] if ratio else g
            if score > best + 1e-15:
                best, bv = score, v
        if bv is None:
            return chosen
        chosen.append(bv)
        spent += prep.costs[bv]
        left.discard(bv)
        for p, s in prep.cover[bv]:
            cur[p] = max(cur[p], s)


def selftest():
    files = {
        "app/__init__.py": "",
        "app/config.py": (
            "from .parser import parse_header\n\n"
            "def load_config(path):\n"
            "    \"\"\"Load a config file and parse its header.\"\"\"\n"
            "    with open(path) as fh:\n"
            "        first = fh.readline()\n"
            "    return parse_header(first)\n\n"
            "def dump_config(cfg):\n"
            "    return str(cfg)\n"),
        "app/parser.py": (
            "def parse_header(line):\n"
            "    key, value = line.split(':')\n"
            "    return {key.strip(): value.strip()}\n\n"
            "def parse_body(lines):\n"
            "    return [l.strip() for l in lines]\n"),
        "app/store.py": (
            "class Store:\n"
            "    def set_value(self, k, v):\n"
            "        self._data = {k: v}\n\n"
            "    def get_value(self, k):\n"
            "        return self._data.get(k)\n\n"
            "class Cache(Store):\n"
            "    def get_value(self, k):\n"
            "        return super().get_value(k)\n"),
        "tests/test_app.py": "def test_x():\n    assert True\n",
    }
    with tempfile.TemporaryDirectory() as tmp:
        for rel, text in files.items():
            p = Path(tmp) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        cfg = RBConfig()
        g = build_graph(tmp, cfg)
        ids = set(g.nodes)
        assert "tests/test_app.py::test_x" not in ids, "tests must be excluded by default"
        assert "app/config.py::load_config" in ids and "app/parser.py::parse_header" in ids
        assert g.has_edge("app/config.py::load_config", "app/parser.py::parse_header"), "call edge"
        assert g.has_edge("app/store.py::Store.set_value", "app/store.py::Store.get_value"), "shared attr edge"
        assert g.has_edge("app/store.py::Cache.get_value", "app/store.py::Store.get_value"), "override edge"

        old = files["app/parser.py"]
        new = old.replace("line.split(':')", "line.split(':', 1)")
        diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                            "a/app/parser.py", "b/app/parser.py"))
        gfiles, gunits = gold_units(g, diff)
        assert gfiles == {"app/parser.py"} and gunits == {"app/parser.py::parse_header"}, (gfiles, gunits)

        issue = "load_config crashes when the config header value contains a colon"
        prep = prepare(g, issue, cfg)
        for method in METHODS:
            for budget in (30, 60, 200, 2000):
                chosen = select(prep, budget, method, 0)
                assert sum(prep.costs[v] for v in chosen) <= budget + 1e-9, (method, budget)
        big = [prep.ids[v] for v in select(prep, 2000, "fl_graph")]
        assert "app/parser.py::parse_header" in big, "graph coverage should reach the callee"
        utils = [utility(prep, select(prep, b, "fl_graph")) for b in (30, 60, 120, 400, 2000)]
        assert all(utils[i] <= utils[i + 1] + 1e-9 for i in range(len(utils) - 1)), utils

        rng = np.random.RandomState(1)
        fake = Prepared(list(range(40)), rng.randint(5, 30, 40).astype(float), np.zeros(40),
                        np.zeros(40), list(range(15)), rng.dirichlet(np.ones(15)),
                        {v: [(int(p), float(rng.rand())) for p in rng.choice(15, 4, replace=False)]
                         for v in range(40)})
        for ratio in (True, False):
            for b in (30, 90, 300):
                lazy = greedy(fake, b, ratio)
                naive = _naive_greedy(fake, b, ratio)
                assert abs(utility(fake, lazy) - utility(fake, naive)) < 1e-9, (ratio, b)
        assert paired_bootstrap([0, 0, 0, 0], [1, 1, 1, 1])["lo"] == 1.0
    print("selftest passed: graph edges, patch mapping, budget feasibility, monotone utility, lazy == naive greedy")


def main():
    ap = argparse.ArgumentParser(description="Budgeted, topology-aware context selection")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--instances", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--budgets", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    r.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--tau", type=float, default=1.0)
    r.add_argument("--dmax", type=float, default=3.0)
    r.add_argument("--beta", type=float, default=0.5)
    r.add_argument("--gamma", type=float, default=0.5)
    s = sub.add_parser("summarize")
    s.add_argument("--results", required=True)
    s.add_argument("--baseline", default="topk_bm25")
    s.add_argument("--metric", default="func_recall",
                   choices=["func_recall", "func_hit", "file_recall"])
    args = ap.parse_args()

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "run":
        cfg = RBConfig(tau=args.tau, dmax=args.dmax, beta=args.beta, gamma=args.gamma)
        n = run_instances(args.instances, args.out, args.budgets, args.methods, args.seeds, cfg)
        print(f"wrote {n} records to {args.out}")
    else:
        print(json.dumps(summarize(args.results, args.baseline, args.metric), indent=2))


if __name__ == "__main__":
    main()
