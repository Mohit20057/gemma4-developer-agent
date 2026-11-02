# Budgeted Context Selection and Tool Conformance for Local Gemma 4 Agents

[![Kaggle Track](https://img.shields.io/badge/Kaggle-Paper%20Track-blue.svg)](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
[![Target Awards](https://img.shields.io/badge/Target%20Awards-Best%20New%20Resource%20%7C%20Overall%20Best%20Paper-gold.svg)](#)
[![Hardware](https://img.shields.io/badge/Hardware-6GB%20Consumer%20GPU-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official Open-Source Repository and Replication Suite** for the Kaggle Research Paper Track: *Google - The Gemma 4 Developer Agent Paper Track*.

---

## Abstract

Coding agent benchmarks typically assume access to cloud APIs with 100k+ token windows. Running open-weights models like Gemma 4 locally on consumer laptops (6–8 GB VRAM) exposes two hurdles: the KV-cache rapidly exhausts GPU memory, triggering out-of-memory crashes, and tool-calling parsers desynchronize across multi-turn loops from minor syntax quirks.

We demonstrate that principled combinatorial context selection allows local models to localize defect sites within tight token budgets:
1. **Facility-Location Context Selection (`repo_budget`)**: Formulating code context selection under budget $B$ as a monotone submodular facility location problem on the code multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, w, c)$, our CELF greedy solver maximizes topological coverage of latent edit targets, outperforming BM25 retrieval by $+14.2\%$ at tight budgets ($B \le 2,000$ tokens, $p < 0.0001$, Cliff's $\delta = +0.58$).
2. **Gemma 4 Conformance Battery & Normalizer (Resource Track)**: An open 10-test diagnostic suite and typed Handshake AI protocol envelope eliminating formatting crashes and restoring 100% execution validity.

Paired with NVIDIA TensorRT-LLM profiling, our framework proves 4-bit Gemma 4 models (E2B/E4B) localize defects safely on a 6 GB GPU within a 4,000-token envelope, leaving over 3 GB of VRAM headroom.

---

## Key Empirical Findings

### 1. Accuracy-vs-Budget Frontier on SWE-bench (10,000 Paired Bootstrap Resamples)

| Method | Budget $B$ | Func Recall | Func Hit | File Recall | vs. BM25 Δ [95% CI] | $p$-value | Cohen's $d$ | Cliff's $\delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `topk_bm25` | 1000 | 0.255 | 0.380 | 0.441 | baseline | — | — | — |
| **`fl_graph`** | **1000** | **0.397** | **0.580** | **0.612** | **+0.142 [+0.104, +0.181]** | **< 0.0001** | **+1.28** | **+0.58** |
| `topk_bm25` | 2000 | 0.372 | 0.520 | 0.580 | baseline | — | — | — |
| **`fl_graph`** | **2000** | **0.498** | **0.680** | **0.721** | **+0.126 [+0.091, +0.162]** | **< 0.0001** | **+1.12** | **+0.51** |
| `topk_bm25` | 4000 | 0.489 | 0.660 | 0.710 | baseline | — | — | — |
| **`fl_graph`** | **4000** | **0.574** | **0.760** | **0.801** | **+0.085 [+0.052, +0.119]** | **< 0.0001** | **+0.79** | **+0.39** |
| `topk_bm25` | 8000 | 0.591 | 0.780 | 0.814 | baseline | — | — | — |
| **`fl_graph`** | **8000** | **0.639** | **0.840** | **0.859** | **+0.047 [+0.018, +0.077]** | **0.0018** | **+0.46** | **+0.24** |

### 2. Gemma 4 Tool Conformance Suite (10 Diagnostic Classes — 100% Pass)

| ID | Failure Mode | Real Symptom | Normalizer Repair Strategy | Pass |
| :--- | :--- | :--- | :--- | :--- |
| `TC01` | Clean Call | Baseline JSON invocation | Native JSON deserialization | 100% |
| `TC02` | Thought Interleaving | `<thought>` precedes call | Reasoning trace extraction | 100% |
| `TC03` | Codeblock Wrapper | Call wrapped in ` ```json ` | Code fence stripping | 100% |
| `TC04` | Unescaped Newlines | Raw diff lines break JSON | Multiline newline escaping | 100% |
| `TC05` | Trailing Commas | Terminal comma in array/obj | Lookahead comma removal | 100% |
| `TC06` | Empty Tool Call | Runtime emits `call:{}` | Prompt re-trigger detection | 100% |
| `TC07` | Text Action Format | Model outputs `ACTION: tool` | Text-action grammar fallback | 100% |
| `TC08` | Token Truncation | Cut off at max tokens | Unclosed quote/brace balance | 100% |
| `TC09` | Double-Encoded JSON | Nested stringified JSON | Recursive string unwrapping | 100% |
| `TC10` | Repeated Call Tags | Repetitive `call:call:tool` | Idempotent token deduping | 100% |

### 3. VRAM Footprint & Feasibility Boundaries on 6.0 GB Laptop GPU

| Model | Precision | Weight VRAM | Headroom | Max Context | Safe Budget | Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma-4-E2B** | **4-bit AWQ** | **1.23 GB** | **4.77 GB** | **32,768** | **4,000** | **Native GPU (Fast)** |
| **Gemma-4-E4B** | **4-bit AWQ** | **2.41 GB** | **3.59 GB** | **32,768** | **4,000** | **Native GPU (Fast)** |
| Gemma-4-E4B | FP16 | 9.63 GB | - | 0 | 0 | CUDA OOM |

---

## Repository Structure

```
├── RESEARCH_PAPER_WRITEUP.md      # Official 2,995-word competition writeup
├── gemma4_paper_submission.ipynb  # 21-cell interactive replication notebook (139 KB)
├── repo_budget.py                 # Monotone submodular facility-location engine & CELF solver
├── repo_prep.py                   # SWE-bench dataset ingestion & detached git worktrees
├── eval_harness.py                # Multi-budget evaluation runner & parameter sweeps
├── stats_analysis.py              # 10,000-resample paired bootstrap statistical engine
├── prompt_skeleton.py             # Gemma 4 tag masking & prompt builder
├── gemma4_conformance.py          # 10-test tool conformance battery & Handshake normalizer
├── hardware_profile.py            # GQA KV-cache memory profiler & TRT-LLM feasibility matrix
├── thumbnail_560x280.png          # Competition writeup card & banner image
└── gemma4_developer_agent_suite.zip # All-in-one resource package (80.2 KB)
```

---

## Quickstart & Replication

All modules require **pure Python 3.8+ and NumPy only** (zero exotic dependencies).

```bash
# Run all independent module selftests
python repo_budget.py selftest
python repo_prep.py selftest
python eval_harness.py selftest
python stats_analysis.py selftest
python prompt_skeleton.py selftest
python gemma4_conformance.py selftest
python hardware_profile.py selftest
```

---

## Citation & DOI
If you find this work or resource suite helpful, cite this work:
```bibtex
@misc{gemma4_budgeted_selection_2026,
  title={Budgeted Context Selection and Tool Conformance for Local Gemma 4 Agents},
  author={Nehete, Mohit},
  year={2026},
  howpublished={Kaggle Research Paper Track: Google - The Gemma 4 Developer Agent Paper Track}
}
```
