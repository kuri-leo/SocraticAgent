# SocraticAgent: An Autonomous Agent for Unlocking Latent Knowledge in LLMs (AAMAS 2026)

Official repository for the AAMAS 2026 paper: **"SocraticAgent: An Autonomous Agent for Unlocking Latent Knowledge in LLMs"**.

This repository contains the demo code for SocraticAgent, a zero-shot autonomous agent that improves LLM reasoning by guiding models to recall and apply their own latent knowledge.

**Authors:** Yang Yan, Yu Lu, Renjun Xu, Zhenzhong Lan

## 📜 Abstract

Reasoning failures in Large Language Models (LLMs) used by autonomous agents are often attributed to knowledge deficits, leading to a reliance on solutions like Retrieval-Augmented Generation (RAG) or parametric fine-tuning. This paper empirically demonstrates that this assumption is often flawed. We identify a quantifiable "knowledge recall gap": while modern LLMs possess 90–97% of the necessary facts for a task, they spontaneously apply only 57–64% of this knowledge during reasoning. This reveals a significant performance gap rooted in a failure of *recall*, not a fundamental absence of knowledge. To address this, we introduce SocraticAgent, a zero-shot autonomous agent that emulates Socratic inquiry by guiding an LLM to first deconstruct a problem and comprehensively detail the internal knowledge required for its solution. Through a deterministic two-action cycle of (1) knowledge deconstruction and (2) grounded reasoning, it procedurally closes this recall gap without any model updates. Across a diverse suite of LLMs, SocraticAgent significantly improves reasoning accuracy, outperforming standard prompting and noisy external retrieval. Critically, our agentic, process-driven approach achieves performance competitive with expensive, data-dependent fine-tuning methods, but does so at inference time without any parametric changes. Our work demonstrates that a deliberative agentic process can serve as a powerful substitute for parametric memory adaptation.

## 💡 Key Findings

Our work reveals that LLMs, despite their prowess on complex benchmarks, rely on fragile, surface-level heuristics during reasoning, often failing to recall knowledge they demonstrably possess.

1.  **The Knowledge Recall Gap:** LLMs possess 90–97% of the facts needed for commonsense reasoning, yet spontaneously apply only 57–64% during a standard Chain-of-Thought process — a consistent ~30-percentage-point chasm between possession and application.
2.  **Process Beats Noisy Retrieval:** For `Gemini-2.5-Flash`, enabling native web search (Noisy RAG) *degraded* performance to 74.23%, while SocraticAgent's internal deliberation boosted it to **78.47%**. For commonsense reasoning, clean internal recall is more effective than unguided external retrieval.
3.  **Competitive with Fine-Tuning, at Zero Cost:** SocraticAgent with `LLaMA3.1-8B` achieves **77.44%** on StrategyQA, significantly closing the gap to the fine-tuned SOTA (82.3%) — without any data curation, training, or model-specific lock-in.
4.  **Process = Perfect Knowledge (for Small Models):** For `Qwen2.5-1.5B`, SocraticAgent achieves a mean accuracy of **76.55%**, *identical* to Oracle RAG with a perfect answer key. A superior internal process can be as valuable as flawless external knowledge.

## 🏗️ How It Works

SocraticAgent employs a deterministic two-action deliberation cycle, requiring no model updates or specialized training:

```
Question q
    │
    ▼
┌─────────────────────────────────────┐
│  Action 1: Deconstruct(q)           │
│                                     │
│  • Iteratively identify key         │
│    concepts via RefineConcepts      │
│  • Generate standalone definitions  │
│    for each via RefineExplanation   │
│  • LLM.Critique validates quality   │
│    at each step (ACCEPT / REVISE)   │
│                                     │
│  → Populates knowledge context K    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Action 2: GroundReasoning(q, K)    │
│                                     │
│  • Synthesize prompt with q + K     │
│  • LLM generates final answer       │
│    strictly grounded in recalled K  │
│                                     │
│  → Produces final answer A_final    │
└─────────────────────────────────────┘
```

The agent's actions exclusively manipulate the LLM's **internal cognitive state** — it performs no external tool-use, web search, or database lookup. The entire process runs at inference time with zero parametric changes.

## 📊 Main Results

Accuracy (%) on three commonsense reasoning benchmarks:

| Model | Method | CommonSenseQA | StrategyQA | TruthfulQA | Mean |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `LLaMA3.1-8B` (Fine-Tuned) | SOTA Parametric | 78.0 | 82.3 | 86.6 | 82.3 |
| `Qwen2.5-7B` (Fine-Tuned) | SOTA Parametric | 78.6 | 83.2 | 81.2 | 81.0 |
| | | | | | |
| `Gemini-2.5-Flash` | Vanilla Baseline | 68.96 | 76.38 | 79.88 | 75.07 |
| | Noisy RAG (w/ Web) | 67.40 | 76.64 | 78.66 | 74.23 |
| | **SocraticAgent (ours)** | **81.63** | **78.66** | **75.12** | **78.47** |
| | | | | | |
| `LLaMA3.1-8B-It` | Vanilla Baseline | 72.38 | 73.26 | 52.06 | 65.90 |
| | Oracle RAG (w/ Ref) | 87.36 | 79.86 | 86.19 | 84.47 |
| | **SocraticAgent (ours)** | **81.05** | **77.44** | **71.85** | **76.78** |
| | | | | | |
| `Qwen2.5-7B-It` | Vanilla Baseline | 79.12 | 70.98 | 63.23 | 71.11 |
| | Oracle RAG (w/ Ref) | 89.11 | 73.81 | 88.61 | 83.84 |
| | **SocraticAgent (ours)** | **81.62** | **77.57** | **69.24** | **76.14** |
| | | | | | |
| `Qwen2.5-1.5B-It` | Vanilla Baseline | 65.40 | 59.10 | 40.87 | 55.13 |
| | Oracle RAG (w/ Ref) | 85.19 | 71.79 | 72.68 | 76.55 |
| | **SocraticAgent (ours)** | **80.60** | **75.14** | **73.92** | **76.55** |

## 🚀 Getting Started

### Prerequisites

```bash
pip install langchain-core langchain-openai langgraph openai pydantic httpx
```

### Usage

1. Configure your API credentials in [`SocraticAgent Demo.py`](SocraticAgent%20Demo.py):

```python
API_KEY = "your_api_key"
BASE_URL = "your_api_base_url"
```

2. Run the demo:

```bash
python "SocraticAgent Demo.py"
```

The demo implements the full SocraticAgent pipeline using [LangGraph](https://github.com/langchain-ai/langgraph), including:
- **Knowledge Analyzer** — Iteratively identifies key concepts for a given question
- **Knowledge Explainer** — Generates and refines standalone explanations for each concept
- **Critic & Reviser loops** — Validates and improves both concepts and explanations via structured critique (ACCEPT/REVISE)
- **Grounded Reasoning** — Produces a final answer strictly grounded in the self-recalled knowledge

## 🧪 Experiment Infrastructure

The experiments in our paper were conducted using a separate benchmarking framework:

**🔗 [LLM_reasoning_benchmarking](https://github.com/kuri-leo/LLM_reasoning_benchmarking)** — A benchmarking framework for LLM reasoning evaluation, supporting dual-mode execution (online streaming & OpenAI Batch API), async scheduling, Pydantic v2 data contracts, and append-only JSONL persistence with idempotent resume.

## ✍️ How to Cite

If you find our work valuable for your research, we would greatly appreciate it if you cite our paper:

```bibtex
@inproceedings{yan2026socraticagent,
  title     = {SocraticAgent: An Autonomous Agent for Unlocking Latent Knowledge in LLMs},
  author    = {Yan, Yang and Lu, Yu and Xu, Renjun and Lan, Zhenzhong},
  booktitle = {Proc. of the 25th International Conference on Autonomous Agents and Multiagent Systems},
  pages     = {1977--1985},
  year      = {2026},
}
```
