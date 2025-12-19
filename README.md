# Reinforcement-Learning-Evaluation (ACE-inspired)

A tiny, deterministic **RL evaluation** task for an LLM. The model must **compress a paragraph without context collapse** (no lost facts, numbers, or units) and return strict JSON:
`{ rewrite, preserved_facts, at_risk_facts, key_insight, delta_update }`.
A reproducible **grader** validates constraints; an **evaluator** runs N trials and reports a **pass rate** (tunable to ~10–40%).

**Inspiration:** *Agentic Context Engineering (ACE): Evolving Contexts for Self-Improving Language Models* — Generator → Reflector → Curator. We mirror ACE by forcing faithful compression and collecting a `key_insight` (reflection) + `delta_update` (curation) to prevent context collapse over time.
Paper: [https://arxiv.org/abs/2510.04618](https://arxiv.org/abs/2510.04618)
(PDF included in `paper/ACE_Agentic_Context_Engineering.pdf`)

---

## What’s implemented

* **Task:** “Preserve facts under summarization pressure.”
* **Environment:** `ORIGINAL`, `FACTS`, `BANNED`, and concise **aliases**.
* **Reward model (grader):** deterministic checks for facts/aliases, numeric fidelity, banned terms, concision (≤60%), ACE-style fields.
* **Evaluator:** 1 API call per run → parse JSON → grade → report pass rate.

**RL-ready:** swap the static prompt for a policy (e.g., PPO, best-of-N) and optimize against the same deterministic reward.

---

## Structure

```
ace_task/
  data.py       # fixtures + alias map
  grader.py     # deterministic reward checks
  evaluate.py   # prompt build, model call, grading loop
  prompt.txt    # task spec + example
```

---

## Setup

```bash
python -m venv .venv
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install anthropic python-dotenv
```

Create `.env` in repo root:

```
ANTHROPIC_API_KEY=sk-...your key...
```

---

## Run

```bash
python -m ace_task.evaluate --runs 10
# or pick a model explicitly:
python -m ace_task.evaluate --runs 10 --model claude-3-5-haiku-latest
```

The script prints each run’s reason (pass/fail) and the final pass rate.

### Train (REINFORCE)

Run a tiny REINFORCE loop against the deterministic grader using any local HF model:

```bash
python -m ace_task.train_reinforce --model gpt2 --steps 5 --scenario report_long --device cpu
```

---

## Tuning the 10–40% pass-rate (as required)

* **`ace_task/evaluate.py`**: adjust `temperature` or small per-run word caps to vary outputs.
* **`ace_task/data.py`**: tighten/relax `ALIAS_MAP` (e.g., allow “inflation 2.1%”).
* **`ace_task/grader.py`**: keep concision at 60% (default); only adjust if needed for target band.

---

## RL framing

* **State:** `(ORIGINAL, FACTS, BANNED, aliases)` + prompt spec.
* **Action:** model emits strict JSON (`rewrite`, `key_insight`, `delta_update`, …).
* **Reward:** deterministic grader → pass/fail (or a shaped score if extended).
* **Episode:** one run.
* **ACE tie-in:** `key_insight` (Reflector) and `delta_update` (Curator) are the small, cumulative context improvements ACE advocates to prevent information loss.

### Next step: implement the RL policy

The immediate next step is to replace the static prompting with a learning **policy** trained against this deterministic reward. Because the environment already exposes **State (S)**, **Action (A)**, and **Reward (R)**, you can plug in **any RL algorithm**—e.g., PPO, policy gradients, best-of-N with rejection sampling, or offline preference optimization—to optimize the model (or prompt parameters) to **maximize pass-rate**. Practically, sample actions (JSON rewrites) from the current policy, score them with `grader.py`, and update the policy to increase rewards; the `key_insight` and `delta_update` fields can also serve as auxiliary signals or evolving context per ACE.

### Executive Summary

I designed a deterministic reinforcement learning evaluation task inspired by Agentic Context Engineering (ACE): Evolving Contexts for Self-Improving Language Models (Zhang et al., 2025). ACE’s Generator → Reflector → Curator loop aims to prevent context collapse—the loss of essential facts during summarization. I chose it because precise, fact-preserving compression mirrors real ML workflows such as experiment reports, paper digests, and release summaries.

The model’s goal is to rewrite a paragraph succinctly while preserving every factual, numerical, and unit-based element. It must output a structured JSON object containing four fields: rewrite, which holds the concise reformulation of the text; preserved_facts and at_risk_facts, which list what information remains intact or is in danger of being lost; key_insight, representing the Reflector’s analytical observation; and delta_update, representing the Curator’s actionable improvement rule derived from that reflection.
A deterministic grader enforces schema validity, concision, numeric fidelity, alias coverage, and fact preservation. It also ensures the Reflector/Curator fields appear, so outputs are correct and improvable.

The codebase includes:
data.py – defines inputs, banned terms, and aliases to maintain authenticity and control the 10–40% pass rate.

grader.py – the reward model, ensuring schema precision and anti-collapse checks.

evaluate.py – runs multiple trials, aggregates pass rate (~40% on claude-3-5-haiku-latest), and explains failures.

prompt.txt – mirrors the grader for full transparency and prompt–reward parity.

Conceptually, the task fits the classic reinforcement learning framework where the state consists of the original paragraph and the associated rules, the action is the model’s structured rewrite of the paragraph, the reward is provided by a deterministic grader that evaluates the output, and the episode encompasses a single forward step from input to graded output.

It teaches a valuable ML skill - context-safe summarization with quantitative verification - while offering measurable feedback and tunable difficulty.
Based on the requirement of the task, I did not train a reinforcement learning model yet. I have built a small, ACE-inspired RL evaluation environment: the model outputs strict JSON summaries, a deterministic grader scores them, and an evaluator reports pass-rate. The next step will be to replace static prompting with a learned policy and train against this reward using algorithms like PPO, policy gradients, best-of-N/rejection sampling, or DPO/preference optimization.
---

## Reference

Zhang et al., *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models*, 2025.
arXiv: [https://arxiv.org/abs/2510.04618](https://arxiv.org/abs/2510.04618) (PDF included).
