# prm-rl · Knowledge Base

> Living document. Every non-obvious learning about this codebase or its
> target platforms goes here so we don't rediscover it. Maintained by
> the Cursor agent per user request; edit freely.
>
> **Last updated:** 2026-07-25

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

### 2.8 Vista snapshot

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

Next step (in progress): actually complete the smoke run with the
upgraded torch and record the `eval_results.json` numbers.

---

## 6. Open issues / next steps

### 6.1 Bump `setup_env.sh` torch install to use cu126 by default

Current script still uses `--index-url https://download.pytorch.org/whl/cu124`.
On fresh Vista venvs that gives torch 2.5.1 (highest aarch64 wheel on
cu124), which then fails at TRL import. Every fresh setup will
currently need the cu126 workaround.

Suggested change: pin to cu126 in `setup_env.sh` with a comment
citing this doc. Only defer if we have a reason to keep cu124 (we
don't right now).

### 6.2 First real Arm 1 result on Vista

After the smoke passes, run a larger Arm 1 configuration and record
the `eval_results.json` here for baseline comparison against future
arms:

```bash
sbatch --time=04:00:00 --job-name=arm1-full -p gh slurm/rl.slurm \
    configs/experiments/arm1_smoke.yaml \
    model.name=Qwen/Qwen2.5-7B-Instruct \
    training.max_steps=500 \
    data.n=2000 \
    output_dir=outputs/arm1_qwen7b
```

### 6.3 CI

`.github/workflows/tests.yml` still not added. When we do, target
Python 3.10 + 3.11 and cache pip. The 21 unit tests are pure Python
and take ~2 min in total.

### 6.4 Better golden-set strategy

Move from `strategy='gsm8k_native'` to `'teacher'` or `'verifier'` once
we have an oracle model. That's when the PRM starts seeing genuine
negative steps and Arms 2 / 3 / 4 / 6 become discriminating instead of
trivially maxing out.

### 6.5 Faithfulness / CMA offline probes

`evaluation/faithfulness.py` and `evaluation/cma.py` exist but aren't
wired into any pipeline yet. Once we have real trained checkpoints,
run them on a paired-intervention set and feed the resulting Phi-CCT /
NIE numbers back into `composite_reward_hacking_score` (currently
passed as `0.0`).
