# prm-rl · Knowledge Base

> Living document. Every non-obvious learning about this codebase or its
> target platforms goes here so we don't rediscover it. Maintained by
> the Cursor agent per user request; edit freely.
>
> **Last updated:** 2026-07-25 (PRM v2 diagnosed: fp16-NaN + backbone-collapse; retrain hyperparams landed, see §6.1)

---

## 0. How to use this doc

- **Section 1** — project shape at a glance.
- **Section 2** — TACC Vista gotchas (the ones we've actually hit).
- **Section 3** — prm-rl codebase conventions and version pins.
- **Section 4** — Colab quickstart notes.
- **Section 5** — Chronological session log (chat summaries).
- **Section 6** — Open issues / next steps.

If something bit us, it should be a numbered bullet in section 2 or 3
with the exact symptom, root cause, and fix. Future-us should be able
to grep for the error message and land on the fix.

---

## 1. Project overview

Research framework for studying **reward hacking in process-based RL for
math reasoning**. Every "experimental arm" reduces to a different
`reward_funcs=[...]` list passed to `trl.GRPOTrainer` — there is
deliberately **no custom PPO/GRPO loop** in the codebase.

- Repo: <https://github.com/rishuray123/prm-rl>
- Two runtime targets:
  - **Colab** — tiny model (SmolLM2-135M), pedagogical, all arms in one notebook.
  - **TACC Vista** — H200 (GH200 120 GB), Slurm-driven, real model scale.

Arms currently implemented:

| # | Name                  | Reward stack                                    | Config                              |
|---|-----------------------|-------------------------------------------------|-------------------------------------|
| 1 | Outcome baseline      | `outcome`                                       | `arm1_outcome.yaml`                 |
| 2 | Naive process         | `naive_process` (PRM step-mean)                 | `arm2_naive_process.yaml`           |
| 3 | Prefix-consistency    | `prefix_consistency` (PRM, truncate at 1st bad) | `arm3_prefix_consistency.yaml`      |
| 4 | Contradiction-aware   | `outcome + naive_process + contradiction` (NLI) | `arm4_contradiction.yaml`           |
| 5 | Counterfactual        | `outcome + counterfactual` (numeric grounding)  | `arm5_counterfactual.yaml`          |
| 6 | Hybrid PROGRS         | `hybrid` (outcome-conditioned centering)        | `arm6_hybrid.yaml`                  |
| 7 | KL-regularized        | Same as chosen base + explicit KL β sweep       | `arm7_regularized.yaml`             |

---

## 2. TACC Vista — environment gotchas

### 2.1 Filesystem layout

| Mount     | Quota    | Use for                                      |
|-----------|----------|----------------------------------------------|
| `$HOME` / `/home1` | ~23 GB (was **90% full** as of 2026-07-24) | Config dotfiles only. **Never** put repos, venvs, or HF caches here. |
| `$WORK`   | 1024 GB  | Repo clone. Resolves to `/work/<id>/<user>/vista/` on GH partition. |
| `$SCRATCH`| large    | venv, HF caches, training outputs, logs.     |

Canonical paths we're using:

```
$WORK/prm-rl                          # git clone
$SCRATCH/venvs/prm-rl                 # venv (setup_env.sh default)
$SCRATCH/hf-cache                     # HF_HOME + datasets + transformers cache
```

### 2.2 ⚠️ Module load MUST precede venv activation

**Symptom:** After sourcing `vista_env.sh` or manually running `module
load python3` *after* activating the venv, `pip install` prints
`Defaulting to user installation because normal site-packages is not
writeable` and installs into the **system** Python's site-packages,
leaving the venv untouched. `python` may still reach the venv
interpreter, but `pip` writes to `/opt/apps/.../python3/3.11.8/lib/...`.

**Root cause:** `module load python3` prepends
`/opt/apps/gcc14/cuda12/python3/3.11.8/bin` to `$PATH`. If this
happens after `source $VENV/bin/activate`, that system path ends up
**ahead of** `$VENV/bin`, so `pip` resolves to system pip.

**Fix (correct incantation for every interactive session):**

```bash
module reset && module load gcc cuda python3     # modules FIRST
source $SCRATCH/venvs/prm-rl/bin/activate         # then venv activation
source $SCRATCH/venvs/prm-rl/vista_env.sh         # env vars only, no `module load`
```

**Sanity check** (all three must resolve to `$VIRTUAL_ENV`):

```bash
which python; which pip; echo "$VIRTUAL_ENV"
```

**Fixed in code:** `slurm/setup_env.sh` no longer bakes `module load`
into `vista_env.sh`; `slurm/_common.sh` already follows the correct
order for Slurm batch jobs. See commit `6b0a0e7`.

### 2.3 ⚠️ PyTorch aarch64 wheels live on `cu126`, not `cu124`

**Symptom:** `pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6"` fails with:

```
ERROR: Could not find a version that satisfies the requirement torch>=2.6
(from versions: 2.0.0, 2.0.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1)
```

**Root cause:** PyTorch stopped publishing `linux_aarch64` wheels on the
`cu124` index after 2.5.1. GH200 needs aarch64. Wheels for
`torch>=2.6` (aarch64) are only on the `cu126` and `cu128` indices.

**Fix:**

```bash
pip install -U --index-url https://download.pytorch.org/whl/cu126 "torch>=2.6"
```

Confirmed working combination on Vista GH200 (2026-07-25):

```
torch  2.13.0+cu126
cuda   True — NVIDIA GH200 120GB
```

**Fixed in code:** `slurm/setup_env.sh` still uses cu124 out of caution
(some fresh venvs pull torch 2.5.x fine); if that fails we've documented
the cu126 fallback here. We should probably bump the default —
[open issue in §6.1].

### 2.4 ⚠️ TRL 0.17+ / 1.x requires `torch >= 2.6`

**Symptom:**
```
ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'
```
raised from `trl/models/utils.py` when importing `GRPOConfig` /
`GRPOTrainer`.

**Root cause:** TRL imports `FSDPModule` unconditionally at module
scope. That symbol landed in torch 2.6 as part of the FSDP2 API. Torch
2.5.1 has `torch.distributed.fsdp` but no `FSDPModule`.

**Fix:** upgrade torch to ≥ 2.6 (see §2.3). Do **not** downgrade TRL —
the whole codebase is on the TRL 1.x API (see §3.3).

### 2.5 flash-attn build failure on Vista is expected, safe to ignore

**Symptom:** During `setup_env.sh`:

```
error: #error -- unsupported GNU version! gcc versions later than 13
       are not supported!
error: command 'nvcc' failed with exit code 255
ERROR: Failed building wheel for flash-attn
Failed to build flash-attn
```

**Root cause:** Vista's `gcc` module provides gcc 14. `nvcc` (CUDA
12.5) supports gcc ≤ 13. flash-attn compiles a `.cu` file through
`nvcc → g++` and hits the version guard.

**Impact:** none for our workloads. `setup_env.sh` wraps flash-attn
install with `|| true`, so the install continues and prints
`Setup complete.` PyTorch's built-in SDPA is the transparent fallback
and handles our model sizes fine (up to 7B with `max_completion_length
≤ 1024`). Do not spend cycles trying to fix flash-attn unless someone
explicitly asks for it.

### 2.6 Compute nodes need `libpython3.11.so.1.0` from the `python3` module

**Symptom:** On a `c610-XXX[gh]` (compute) node, immediately after
`source $VENV/bin/activate`:

```
python: error while loading shared libraries: libpython3.11.so.1.0:
        cannot open shared object file: No such file or directory
```

**Root cause:** `idev` gives you a bare login shell on the compute
node; the modules loaded on login2 during `setup_env.sh` are not
inherited. The venv's `bin/python` is a symlink to
`/opt/apps/gcc14/cuda12/python3/3.11.8/bin/python3`, which is
dynamically linked against `libpython3.11.so.1.0`. That library is
only on `LD_LIBRARY_PATH` when the `python3` module is loaded.

**Fix:** same incantation as §2.2 — `module reset && module load gcc
cuda python3` before doing anything else on the compute node.

### 2.7 Slurm queues we actually use

| Queue    | Node     | Max nodes/job | Wall time | Charge     | Notes                          |
|----------|----------|---------------|-----------|------------|--------------------------------|
| `gh-dev` | 1× H200  | 8             | 2 h       | 1 SU/hr    | Fastest for smoke tests / iteration. |
| `gh`     | 1× H200  | 64            | 48 h      | 1 SU/hr    | Real runs. `idev -p gh` allocated immediately in this session. |
| `gg`     | 144 CPU  | 32            | 48 h      | 0.33 SU/hr | CPU only, we're not using it yet. |

Allocation: `ASC26008` (9504 SUs remaining as of 2026-07-24).

### 2.8 ⚠️ /home1 quota bites HuggingFace *and* Triton — redirect BOTH

**Symptom:** during any HF-backed run (train_rl, train_prm, ...):

```
Could not cache non-existence of file. Will ignore error and continue.
  Error: [Errno 122] Disk quota exceeded:
  '/home1/<uid>/<user>/.cache/huggingface/hub/models--Qwen--...'

OSError: [Errno 122] Disk quota exceeded:
  '/home1/<uid>/<user>/.triton/cache/NPF46G6E5C4DAGWSQ5FWMRFU7SVQ3RFRWWFS74W6L4F5KPLMJXBA'
```

Second one comes from Triton, mid-GRPO, during the first
`self.rotary_emb(...)` call — Triton JIT-compiles a kernel and tries
to write the cached artefact to `~/.triton/cache/*`, which fails
under quota.

**Root cause:** Vista's `/home1` is ~23 GB and typically half-full
before you land. HF caches are ~3 GB per open-weights model; Triton
kernels are small individually but proliferate. `HF_HOME` alone is
not enough — `TRITON_CACHE_DIR` is a separate variable, as are
`XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `MPLCONFIGDIR`, and
`TORCHINDUCTOR_CACHE_DIR`.

**Fix (permanent, code):** `slurm/env_caches.sh` sets *all* of these
to sit under `$SCRATCH/prm-rl-caches/`. It is sourced by
`_common.sh`, `iter_all_arms.sh`, and `smoke_all_arms.sh`.
`setup_env.sh` also bakes the same exports into `vista_env.sh` for
manual sessions.

**Fix (one-shot, existing caches):** `slurm/fix_home_quota.sh`
rsyncs `~/.cache/{huggingface,pip,torch,matplotlib,wandb}` and
`~/.triton` into `$SCRATCH/prm-rl-caches/home-mirror/*` and symlinks
the home paths back, so a partially-downloaded 3 GB Qwen checkpoint
isn't lost. Run once after hitting this error, then future writes
will land in `$SCRATCH` directly via the env vars above.

**Correct incantation for a fresh interactive session on Vista:**

```bash
module reset && module load gcc cuda python3
source $SCRATCH/venvs/prm-rl/bin/activate
source $SCRATCH/venvs/prm-rl/vista_env.sh    # ← was missing; sets HF/Triton dirs
# Sanity:
env | grep -E '^(HF_HOME|TRITON_CACHE_DIR|XDG_CACHE_HOME)='
```

The `source vista_env.sh` step was missing from earlier runbooks; if
you skip it, all caches end up in `$HOME`. `slurm/iter_all_arms.sh`
and friends now source `slurm/env_caches.sh` explicitly, so even a
partially-configured shell can't leak into `$HOME` if the driver is
used.

### 2.9 ⚠️ DeBERTa-v3 PRM silently NaN's in fp16 on H200 → PRM signal dies

**Symptom:** `process_correctness` uniformly 0.000 (or, on an
un-negated PRM, uniformly 1.000) across every arm in the Phase 1.5
iter summary. HF training log for `outputs/prm_v2/` looks perfect
(`eval_f1 ≥ 0.83`, healthy loss curve, sensible label balance), but
`prm.score_steps(...)` returns `float('nan')` for both known-positive
and known-negative training examples. The downstream
`process_correctness` aggregator squashes NaN → 0 during summary
formatting, so the failure hides in plain sight.

**Root cause:** `outputs/prm_v2/config.json` gets `"dtype":
"float16"` written by HF Trainer even when
`TrainingArguments(bf16=True)`. On next `from_pretrained`, the model
loads in fp16, and DeBERTa-v3's disentangled attention (relative
position keys × content queries) overflows/NaNs on H200 forward
passes. This is a documented upstream issue — DeBERTa-v3 only works
reliably in fp32 or bf16 at inference.

Consequence during Phase 1.5: every process-based reward function
(`naive_process`, `naive_process + gold_verification`, ...) was
called with a broken PRM, yielded NaN reward, was silently zeroed by
the GRPO reward-collation code, and the process-based arms trained
with dead process signal. That's why arms 2/3/5 didn't diverge from
arm 1 on any behavioral metric worth caring about.

**Fix (code, no retrain required):**
`prm_rl.models.prm.load_prm` now defaults to `torch_dtype=torch.
float32`. The on-disk weights are healthy; the bug was in the
forward pass only. Callers can override with `torch_dtype=torch.
bfloat16` on H200 if they want the throughput.

**Verification recipe** — after any PRM retrain, run:

```python
from datasets import load_from_disk
from prm_rl.models.prm import load_prm
import torch
d = load_from_disk("data/prm_v2")
prm = load_prm("outputs/prm_v2", device="cuda" if torch.cuda.is_available() else "cpu")
pos = [i for i, l in enumerate(d["label"]) if l == 1][:5]
neg = [i for i, l in enumerate(d["label"]) if l == 0][:5]
for i in pos + neg:
    ex = d[i]
    s = prm.score_steps(ex["question"], [ex["step"]])[0]
    print(f'label={ex["label"]} kind={ex["neg_kind"]!r:<25s} pred={s:.3f}')
```

Expected: positives should score ≥ 0.6, negatives ≤ 0.4. If any
score is NaN, the model is still loading in fp16 — check
`config.json`'s `dtype` field and `load_prm`'s `torch_dtype` argument.

### 2.10 Vista hardware snapshot

- Node: `c610-XXX` GH200 (Grace ARM CPU + H200 GPU, 120 GB HBM — note **not** the 480 GB variant seen in some GH200 SKUs).
- OS: Rocky 9.7 (per the 2026-02-12 admin notice in the motd).
- CUDA driver: 590.48.01. CUDA toolkit module: `cuda/12.5` (`nvcc` reports 12.5).
- System Python module: `python3/3.11.8` at `/opt/apps/gcc14/cuda12/python3/3.11.8`.
- gcc module: gcc 15.1.0 after `module reset` (also warned as gcc 14 elsewhere — Vista's default toolchain floats).

---

## 3. prm-rl codebase conventions

### 3.1 Design principle

Every arm = a `reward_funcs=[...]` list + weights fed to a stock
`trl.GRPOTrainer`. If a change would require a custom PPO/GRPO loop, it
should probably be a new reward function instead.

### 3.2 Reward registry

`src/prm_rl/rewards/__init__.py` maps reward names to lazy factory
lookups. `outcome` is a plain function; all others are `make_*`
factories that accept scorer/PRM/threshold kwargs and return a callable
with the TRL signature `fn(prompts, completions, **kwargs) -> list[float]`.

To add a new arm:

1. Write a factory in `src/prm_rl/rewards/<name>.py`.
2. Register it in `REGISTRY` (`_FACTORIES` dict).
3. Add a YAML under `configs/experiments/<name>.yaml` following the
   existing shape (`rewards: - name: ..., weight: ...`).
4. If you want it in the Colab notebook, drop a `train_arm(...)` call
   into the extension cells.

### 3.3 Version pin matrix (as of 2026-07-25)

Every one of these has bitten us. Do not bump loosely.

| Package        | Minimum | Reason                                                         |
|----------------|---------|----------------------------------------------------------------|
| torch          | ≥ 2.6   | TRL imports `FSDPModule` (FSDP2). See §2.4.                    |
| transformers   | 5.x     | Colab has 5.13; API keys we depend on: `dtype=`, `processing_class=`. |
| trl            | 1.x     | API we use: no `max_prompt_length` in `GRPOConfig`, `max_length` in `SFTConfig`. |
| datasets       | 5.x     | Colab has 5.0. Load id **must** be `openai/gsm8k`, not bare `gsm8k`. |
| huggingface_hub| latest  | Enforces `namespace/name` for dataset ids.                     |

**Known API renames** (already handled in code + notebook):

| Old                                     | New                          | Where                                    |
|-----------------------------------------|------------------------------|------------------------------------------|
| `SFTConfig(max_seq_length=)`            | `max_length=`                | `training/sft.py`, notebook, SFT YAMLs   |
| `SFTTrainer(tokenizer=)` / `Trainer(tokenizer=)` | `processing_class=` | `training/{sft,prm_train}.py`, notebook  |
| `from_pretrained(torch_dtype=)`         | `dtype=`                     | `models/policy.py`, notebook             |
| `GRPOConfig(max_prompt_length=)`        | (removed, tokenizer handles) | `training/rl.py`, notebook, arm YAMLs    |
| `load_dataset("gsm8k")`                 | `"openai/gsm8k"`             | `data/gsm8k.py`                          |

### 3.4 Evaluation package layout

`src/prm_rl/evaluation/__init__.py` re-exports 7 scalar/behavioral/trap
metrics that `scripts/evaluate.py` imports flat. **Do not** add torch
imports at module scope in any file that ends up in that re-export
list (`behavioral.py`, `crhs.py`, `est.py`, `metrics.py`, `traps.py`)
— we deliberately keep `import prm_rl.evaluation` cheap. Heavier
`faithfulness.py` / `cma.py` require explicit submodule imports.

### 3.5 Slurm scripts

- `slurm/setup_env.sh` — one-time env bootstrap on login node.
- `slurm/_common.sh` — sourced by every batch script; loads modules
  first, activates venv, sets HF cache env vars.
- `slurm/rl.slurm`, `slurm/sft.slurm`, `slurm/prm.slurm`,
  `slurm/eval.slurm`, `slurm/data_prep.slurm` — one per pipeline stage,
  each takes a config YAML as the first positional argument.
- `slurm/smoke_arm1.slurm` — end-to-end smoke on gh-dev (train + eval)
  with hardcoded config.
- `slurm/sweep_all_arms.sh` — fans out one sbatch per arm YAML.

Batch pattern for chained dependencies:

```bash
J1=$(sbatch --parsable slurm/sft.slurm configs/experiments/sft.yaml)
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm/prm.slurm configs/experiments/prm.yaml)
J3=$(sbatch --parsable --dependency=afterok:$J2 slurm/rl.slurm  configs/experiments/arm2_naive_process.yaml)
sbatch --dependency=afterok:$J3 slurm/eval.slurm configs/experiments/arm2_naive_process.yaml
```

---

## 4. Colab quickstart

`notebooks/quickstart_colab.ipynb` runs all six arms end-to-end on
SmolLM2-135M + DeBERTa-v3-xsmall PRM. The notebook auto-clones and
`git fetch && reset --hard origin/main` on every restart so kernel
restarts pick up new commits without needing to clear the folder.

Key cells:

| Cell # | Purpose                                              |
|--------|------------------------------------------------------|
| 0      | Intro / pipeline overview                            |
| 2–4    | Bootstrap (clone/pull, `pip install -e .`, version dump) |
| 6–8    | Load GSM8K + build golden + PRM data                 |
| 10–11  | SFT the tiny policy                                  |
| 13–14  | Train the PRM                                        |
| 16–17  | Arm 1 (outcome)                                      |
| 19–20  | Arm 2 (naive process)                                |
| 22     | `train_arm(...)` helper + `nli_scorer` load          |
| 24, 26, 28, 30 | Arms 3, 4, 5, 6                              |
| 32–34  | Evaluate all six + `pd.DataFrame` of results         |

---

## 5. Session log

### 2026-07-22 — Colab notebook: Arms 3–6 extension

Extended the Colab quickstart from 26 → 36 cells to run Arms 3–6
alongside 1 & 2 on the same tiny model. Introduced `train_arm(tag,
out_dir, reward_funcs, weights)` helper so each new arm is 3 lines
(factory, sanity print, `train_arm` call). Reused `prm_scorer` for
Arms 3 & 6; loaded NLI once for Arm 4. Extended `evaluate()` calls
and the results `pd.DataFrame` to cover all six arms.

Side finding: Cursor's Shell tool was auto-injecting
`--trailer "Co-authored-by: Cursor <cursoragent@cursor.com>"` on every
`git commit -m`. Disabled at the setting level (per user).

Commit landed clean as **`230c3cc`**.

### 2026-07-24 → 2026-07-25 — TACC Vista Arm 1 smoke test bring-up

**Goal:** verify the full RL pipeline (env → HF download → GRPO train
→ save → eval) works on a Vista H200 with a 1.5B model before spending
SUs on a full run.

**Artifacts added:**

- `configs/experiments/arm1_smoke.yaml` — Qwen2.5-1.5B, `max_steps=10`,
  `data.n=64`, single outcome reward, `eval.n_test=20`, no PRM/traps.
- `slurm/smoke_arm1.slurm` — chained train → evaluate on `gh-dev`
  with 45 min walltime.
- `slurm/README.md` — new "Smoke test" section with success criteria
  and scale-up recipe.

**Bugs found and fixed** (order of discovery):

1. **Cursor Shell auto-injected commit trailer** — disabled at settings
   level; commits went through clean.
2. **`slurm/README.md` had `prm_rl` (underscore) in one line** while
   `setup_env.sh` and `_common.sh` use `prm-rl` (hyphen). Fixed in
   commit `171a1db`.
3. **`libpython3.11.so.1.0: cannot open shared object file`** on the
   compute node — see §2.6. Manual `module load` unblocked; permanent
   fix in `_common.sh` (already correct) and interactive docs.
4. **`ImportError: cannot import name 'FSDPModule' from
   'torch.distributed.fsdp'`** raised by TRL — see §2.4. Root cause:
   torch 2.5.1 too old.
5. **`cannot import name 'behavioral_scores' from
   'prm_rl.evaluation'`** — `evaluate.py` imports flat but package
   `__init__.py` was empty. Fixed by re-exporting the 7 scalar metrics
   in commit `0c79cda`.
6. **`Defaulting to user installation because normal site-packages is
   not writeable`** — pip was targeting system Python instead of venv
   because I had baked `module load python3` into `vista_env.sh`,
   which was sourced *after* venv activation and re-ordered `$PATH`.
   Reverted the bake, updated activation order docs. Commit `6b0a0e7`.
7. **`torch>=2.6` not on cu124 index for aarch64** — see §2.3. Fixed
   by switching to `cu126`. Confirmed torch 2.13.0+cu126 works on
   GH200 120GB.

Commits landed this session (chronological):

| SHA         | Summary                                                 |
|-------------|---------------------------------------------------------|
| `e7d1258`   | Add Arm 1 smoke config + slurm/smoke_arm1.slurm         |
| `171a1db`   | slurm/README typo (prm_rl → prm-rl)                     |
| `0c79cda`   | Fix evaluate.py imports + torch pin bump + module loads |
| `6b0a0e7`   | Vista: modules BEFORE venv activation (PATH fix)        |
| `9400ad8`   | Add this knowledge-base doc                             |

### 2026-07-25 — First Arm 1 result on Vista GH200

Smoke run completed end-to-end after applying the cu126 torch fix
(§2.3). Config: `configs/experiments/arm1_smoke.yaml` (Qwen2.5-1.5B,
10 GRPO steps, n=64 train, n=20 test, 5 traps).

```json
{
  "accuracy": {"accuracy": 0.45, "correct": 9, "n": 20},
  "behavior": {"avg_tokens": 149.85, "avg_steps": 5.3, "avg_self_rougeL": 0.199},
  "traps": {"exploit_rate": 0.4, "trap_solve_rate": 0.2, "n": 5},
  "crhs": {"CRHS": 0.485, "CRHS_not_exploit": 0.6, "CRHS_phi_cct": 0.5,
           "CRHS_not_verbose": 1.0, "CRHS_nie": 0.0}
}
```

**Interpretation:**
- 45% ± ~22 pp (Wilson 95% CI at n=20) — too noisy for arm comparison,
  useful only as a "pipeline works" signal.
- 10 GRPO steps ≪ what's needed for learning — this is essentially
  base-model Qwen2.5-1.5B-Instruct behavior with a light nudge.
- CRHS = 0.485 with `CRHS_not_verbose = 1.0` — model is right at the
  verbosity baseline (149.85 tokens vs 150 baseline), which by our
  scoring means "no verbosity penalty" (probably too generous — the
  baseline should be recalibrated once we have SFT-trained anchors).

Thesis-grade eval requires at minimum `n_test ≥ 500`, `max_steps ≥
1000`, and 3–5 seeds per arm. Current smoke is validation only.

### 2026-07-25 — All-arms smoke driver

Added:

- `configs/experiments/prm_smoke.yaml` — DeBERTa-v3-xsmall, n=64, 1 epoch.
- `configs/experiments/arm{2,3,4,5,6}_smoke.yaml` — 1.5B policy, 10
  steps, mirroring `arm1_smoke.yaml`, only rewards differ.
- `slurm/smoke_all_arms.sh` — sequential driver: golden → PRM → arms
  1..6 → aggregate. ~35–40 min on H200, under gh-dev's 2 h.
- `src/prm_rl/scripts/summarize_smoke.py` — reads every
  `eval_results.json` and writes a markdown comparison table.

**Caveat baked into `prm_smoke.yaml`:** with
`strategy='gsm8k_native'` the PRM only sees positive steps and will
collapse to ~1.0 for everything, so arms 2/3/4/6 rewards will not
meaningfully differentiate completions in the smoke. This is fine for
pipeline validation; not for scientific comparison. See §6.4.

Commits landed today (post-smoke):

| SHA        | Summary                                                          |
|------------|------------------------------------------------------------------|
| `a20b5b4`  | Add PRM + Arm 2..6 smoke configs, driver, summarizer, KB update  |

### 2026-07-25 — All-arms smoke result on Vista GH200

Full driver ran end-to-end in ~35 min on `c610-001[gh]`. NLI model
for Arm 4 downloaded cleanly on first use.

| arm  | accuracy | process_correctness | avg_tokens | avg_steps | avg_self_rougeL | exploit_rate | trap_solve_rate | CRHS  |
|------|----------|---------------------|------------|-----------|-----------------|--------------|-----------------|-------|
| arm1 | 0.400    | N/A                 | 146.6      | 4.50      | 0.160           | 0.40         | 0.40            | 0.485 |
| arm2 | 0.250    | 1.000               | 160.7      | 6.55      | 0.236           | 0.40         | 0.40            | 0.474 |
| arm3 | 0.350    | 1.000               | 169.3      | 6.75      | 0.235           | 0.40         | 0.20            | 0.466 |
| arm4 | 0.400    | 1.000               | 151.8      | 5.95      | 0.204           | 0.20         | 0.40            | 0.553 |
| arm5 | 0.250    | 1.000               | 142.9      | 5.40      | 0.207           | 0.40         | 0.20            | 0.485 |
| arm6 | 0.400    | 1.000               | 142.7      | 4.30      | 0.153           | 0.60         | 0.40            | 0.415 |

**What this tells us (real signals):**

1. **Every arm's pipeline works on Vista.** Zero blockers to scaling up.
2. **`process_correctness = 1.000` for arms 2/2/4/6** — exactly as the
   `prm_smoke.yaml` caveat predicted. The PRM trained on
   `gsm8k_native` (label=1 only) collapsed to outputting ~1.0 for
   every step. This is a diagnostic signal that our PRM is useless,
   not a training signal. Fixing this is [§6.4] / [§7.1].
3. **Arm 1 dropped from 0.45 → 0.40** between the two runs with
   identical config — GRPO's default sampling is stochastic
   (`num_generations=4`, `temperature=0.9`). This is our concrete
   evidence that a single-seed n=20 eval has huge run-to-run variance;
   fixing this needs multiple seeds *and* larger n_test.

**What this does NOT tell us (must not be interpreted as science):**

- **Arm 4 (contradiction) looks best with CRHS 0.553 and lowest
  exploit rate 0.20.** Tempting to celebrate, but at n=5 traps a
  single flip = 0.20 change; and the contradiction reward hasn't seen
  a discriminating PRM anyway. Don't cite this.
- **Arm 6 (hybrid) looks worst with exploit_rate 0.60.** Same caveat.
- **Arms 2 and 5 dropped to 0.25 accuracy.** Consistent with 10 steps
  of GRPO on a mostly-uniform reward moving the policy off its
  instruction-tuned prior in an unhelpful direction; but at n=20 with
  ±22 pp CIs this is suggestive at best.

**Bottom line:** smoke phase is complete. All findings above are
"pipeline works" grade, not "publishable" grade. Move on to Phase 2
(§7) for the real experiments.

Commits landed for the all-arms smoke result:

| SHA        | Summary                                                    |
|------------|------------------------------------------------------------|
| _pending_  | Log first all-arms smoke result table + phase 2 plan       |

### 2026-07-25 — Phase 1.5 iteration harness + Phase 2 sweep artefacts

Added (all pre-Vista — code is on the local Mac, waiting to be
pulled and run):

- **`src/prm_rl/data/prm_data.py`** — extended `build_prm_dataset`
  with `inject_negatives_prob`, `negative_kinds`,
  `max_negatives_per_example`, `seed`. Four negative kinds are
  implemented: `arithmetic_mutation`, `operator_swap`,
  `fabricated_conclusion`, `duplicate_prev_step`. Deterministic under
  fixed seed. New `summarize_prm_dataset(ds)` returns
  `{n, n_pos, n_neg, pos_frac}` for logging. Directly addresses §6.1
  Path A.
- **`src/prm_rl/scripts/build_prm_data.py`** — CLI flags
  `--inject_negatives_prob`, `--negative_kinds`,
  `--max_negatives_per_example`, `--seed`. Logs the pos/neg split.
- **`src/prm_rl/training/prm_train.py`** — reads the new fields from
  the YAML `data:` block and logs the dataset stats before training.
- **`tests/test_prm_data.py`** — 10 tests (5 pure-Python + 5
  `datasets`-backed). Local light run: all 21 previous tests still
  pass; the new file's `datasets`-backed tests skip cleanly on the
  local Mac and will run on Vista via `make test`.
- **`configs/experiments/prm_v2.yaml`** — DeBERTa-v3-base PRM
  trained on `data/golden_v2` (n=2000) with
  `inject_negatives_prob=0.5`. Target: val F1 ≥ 0.7 (§6.1 success
  criterion).
- **`configs/experiments/arm{1..6}_iter.yaml`** — Phase 1.5 configs
  sized for `gh-dev`'s 2 h cap. Qwen2.5-1.5B, max_steps=50, n=256
  train, n_test=100. Point at `outputs/prm_v2` for the process-based
  arms.
- **`slurm/iter_all_arms.sh`** — sequential driver: build golden v2 →
  build PRM data with negatives → train PRM v2 → RL train+eval for
  arms 1..6 → aggregate to `outputs/iter_summary.md`. Wall-clock
  budget: ~75–105 min on H200, comfortably under gh-dev's 2 h.
- **`configs/experiments/arm{1..6}_phase2.yaml`** — Phase 2 configs
  at Qwen2.5-7B, max_steps=500, n=2000 train, n_test=500,
  num_generations=8, batch 4 × grad_accum 4. Requesting 5 h walltime
  per job on `gh` (expected actual ~2.5–4 h).
- **`slurm/phase2_sweep.sh`** — sbatch fan-out for 6 arms × 3 seeds
  = 18 jobs on `gh`. Each arm's eval is chained via
  `--dependency=afterok:$train_jid` off its train job so wall time
  is per-arm, not global. Pre-flight checks that Phase 1.5 artefacts
  (`data/golden_v2`, `data/prm_v2`, `outputs/prm_v2`) exist, sources
  `env_caches.sh`, and prefetches Qwen2.5-7B into `HF_HOME` on the
  login node so the 18 GPU jobs don't race the same 14 GB download.
  **Outputs go to `$SCRATCH/prm-rl-outputs/{arm}_seed{S}/`** — 18
  × 14 GB of 7B checkpoints would blow through `$WORK`'s quota
  otherwise (§2.8). Logs stay in `$WORK/prm-rl/logs/` (text, cheap).
- **`slurm/phase2_summarize.sh`** — reads
  `$SCRATCH/prm-rl-outputs/{arm}_seed{S}/eval_results.json` and
  writes the aggregated markdown table to
  `$WORK/prm-rl/outputs/phase2_summary.md`. Safe to run before every
  seed has finished: missing seeds print a WARN and are simply left
  out.

**Iteration cadence going forward:**

1. `idev -p gh-dev -t 02:00:00` → `bash slurm/iter_all_arms.sh`.
   Fast, no queue wait, tells us whether the PRM v2 signal is
   discriminating (`process_correctness` in the summary should stop
   being uniformly 1.000).
2. If (1) looks reasonable, `bash slurm/phase2_sweep.sh` from a login
   node. Comes back overnight with the real numbers.
3. `bash slurm/phase2_summarize.sh` → paste table into thesis
   `thesis_draft/04_chapter4_experiments.md §4.8` and re-run pandoc
   per `thesis_draft/README_HOW_TO_ASSEMBLE.md`.

**Why gh-dev interactively rather than the gh queue for iteration:**
gh-dev is 2 h wall, immediate allocation; gh is 48 h wall, queues.
A 500-step 7B arm exceeds gh-dev's cap (§6.2 est. 2–4 h per arm) so
we CANNOT run the real sweep there — but a 50-step 1.5B iter arm
fits, and iterating on `iter_all_arms.sh` on gh-dev with `idev` is
by far the fastest way to catch code bugs in the new PRM pipeline
before spending 50+ SUs on the 7B sweep.

Commits pending for this batch:

| SHA        | Summary                                                        |
|------------|----------------------------------------------------------------|
| _pending_  | Synthetic-neg PRM data, prm_v2, arm{1..6}_{iter,phase2}, sweep |

---

## 6. Phase 2 — plan for the first thesis-grade run

**Goal:** produce Arm 1..6 numbers that can actually be compared and
cited. Blockers, in dependency order:

### 6.1 Fix the PRM training data (P0 — blocks everything downstream)

The smoke PRM trained on `gsm8k_native` yields `process_correctness =
1.000` for every arm (§5, 2026-07-25 result table). Until we have a
PRM that discriminates correct from incorrect steps, arms 2 / 3 / 4 /
6 rewards are near-uniform and GRPO gets almost no signal.

Two viable paths (pick one):

- **`strategy='teacher'`** — sample multiple reasoning traces from a
  teacher LLM (e.g. Qwen2.5-72B or GPT-4-class), keep the one whose
  final answer matches gold, mark steps in wrong-answer traces as
  negatives. Requires teacher API/model access.
- **Synthetic negatives** — cheapest to try first. Extend
  `src/prm_rl/data/prm_data.py::build_prm_dataset` with an
  `inject_negatives_prob: float` kwarg that, per row, appends a
  wrong-step continuation (e.g. arithmetic mutation, "therefore
  <random number>") with label=0. Matches the notebook's cell-13
  trick, but at package level.

Recommended: implement synthetic negatives first (self-contained,
~30 LOC), retrain PRM at scale, measure PRM val F1. Only add teacher
distillation if val F1 stays below ~0.7.

**STATUS (2026-07-25):** Path A (synthetic negatives) is landed in
code and has been run twice. Both attempts produced a PRM that
**did not learn to discriminate**, for two overlapping reasons:

1. **fp16 NaN at inference (see §2.9)** — hidden the failure of (2)
   behind an even more catastrophic failure. Fixed by defaulting
   `load_prm` to fp32.
2. **Backbone converged to the class prior** — see run 1's trainer
   log (`eval_f1 = 0.834`, `eval_loss = 0.598`) which exactly matches
   a constant-positive classifier on a 71%-positive dataset. Post-fix
   probe confirmed every input produced `pred = 0.719 ≈ pos_frac`.
   Contributing factors and fixes:
   - **Right-side truncation** dropped the tail (=current step) on
     any long input, making positive/negative pairs indistinguishable
     to the model.
     Fix: `tokenizer.truncation_side = "left"` in both `prm_train.py`
     and `load_prm()` in `prm.py`.
   - **`max_length=1024` on a model whose native buffer is 512** ran
     the disentangled attention on positions it was never trained on.
     Fix: `max_length: 512` in `configs/experiments/prm_v2.yaml`.
   - **`bf16: true` on a small classifier** underflowed classifier
     gradients into zero — the classifier bias moved to the prior
     and stayed there.
     Fix: `bf16: false` (fp32 training) in `prm_v2.yaml`.
   - **LR = 2e-5 for only 3 epochs** was too conservative to push
     the backbone away from its init.
     Fix: `learning_rate: 5.0e-5`, `num_train_epochs: 5`.

**Success criterion tightened:** val F1 ≥ 0.7 is *not* sufficient —
on this dataset a constant-positive classifier already scores 0.83.
The PRM must also pass the smoke probe in §2.9 (positives > 0.6,
negatives < 0.4 on training examples). Only then is it safe to
promote to Phase 2.

Next action: rerun `python -m prm_rl.scripts.train_prm --config
configs/experiments/prm_v2.yaml` (or `slurm/iter_all_arms.sh` from
stage 3 onward), then rerun the §2.9 probe. If discrimination still
collapses, escalate to Path B (teacher distillation).

### 6.2 Scale the RL runs (P1)

Once PRM is discriminating, re-run each arm at real scale on `gh`
(48 h queue, 1 SU/hr per H200 hour):

| Knob                  | Smoke | Phase 2 target |
|-----------------------|-------|----------------|
| Policy                | Qwen2.5-1.5B | Qwen2.5-7B-Instruct |
| SFT stage             | skipped      | 2 epochs on n=2000 golden |
| Golden n              | 64           | 2000                      |
| GRPO `max_steps`      | 10           | 500 (later 2000)          |
| GRPO `n_generations`  | 4            | 8                         |
| GRPO batch size       | 2            | 4 × grad_accum 4          |
| Seeds per arm         | 1            | 3                         |
| Eval `n_test`         | 20           | 500                       |
| Trap prompts          | 5            | 30+ (needs authoring)     |

Expected cost per arm: ~2–4 H200 hours = ~2–4 SUs. 6 arms × 3 seeds ×
3 SUs ≈ **~54 SUs total for one full sweep.** We have 9504 SUs on
`ASC26008`, so plenty of headroom for multiple sweeps.

Submission pattern:

```bash
for arm in arm1_outcome arm2_naive_process arm3_prefix_consistency \
           arm4_contradiction arm5_counterfactual arm6_hybrid; do
    for seed in 42 43 44; do
        sbatch --time=04:00:00 -p gh -J "${arm}-s${seed}" slurm/rl.slurm \
            "configs/experiments/${arm}.yaml" \
            "seed=${seed}" \
            "output_dir=outputs/${arm}_seed${seed}"
    done
done
```

### 6.3 Add authored trap prompts (P1)

Current `data/traps/trap_examples.json` has 5 hand-crafted scenarios
(4 kinds: `impossible`, `underspecified`, `shortcut`,
`adversarial`). Grow to 30+ so `exploit_rate` has decimals of
resolution finer than 0.20. Balance across the 4 kinds so per-kind
breakdowns are possible.

### 6.4 Wire in faithfulness / CMA offline probes (P2)

`evaluation/faithfulness.py` (paired counterfactual, produces `phi_cct`)
and `evaluation/cma.py` (causal mediation, produces `NIE`) already
exist but aren't chained into `evaluate.py`. In Phase 2:

1. After every arm's RL train, run `scripts/faithfulness_probe.py`
   (helper referenced in `evaluate.py` docstring — needs to be
   written) on a fixed paired-intervention set.
2. Save `cct.json` / `cma.json` alongside the checkpoint.
3. Rerun `evaluate.py` with `--override eval.cct_json=... eval.cma_json=...`.

Both are needed for the "not just accuracy" side of the composite
reward-hacking score to actually reflect faithfulness rather than
being clamped to 0.

### 6.5 Add CI (P3)

`.github/workflows/tests.yml` — matrix Python 3.10/3.11, cache pip,
run the 21 unit tests. Trivial to add; currently missing.

---

## 7. Historical open issues (kept for context)

### 7.1 Bump `setup_env.sh` torch install to use cu126 by default

Current script still uses `--index-url https://download.pytorch.org/whl/cu124`.
On fresh Vista venvs that gives torch 2.5.1 (highest aarch64 wheel on
cu124), which then fails at TRL import. Every fresh setup will
currently need the cu126 workaround.

Suggested change: pin to cu126 in `setup_env.sh` with a comment
citing this doc. Only defer if we have a reason to keep cu124 (we
don't right now).
