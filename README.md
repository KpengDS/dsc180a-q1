# DSC 180A Q1: Order and Selection Sensitivity in In-Context Learning

Individual UC San Diego DSC 180A Q1 research project for the November 6 Code Checkpoint.
I study how AG News classification accuracy changes when **the same eight
training demonstrations are reordered** versus when **the demonstration set is
replaced under a fixed positional ordering**. The project is based on
[Order Matters: Rethinking Prompt Construction in In-Context Learning](https://arxiv.org/abs/2511.09700).
Deliverables are reproducible Python code and a research report.

The main pipeline uses **Qwen2.5-1.5B Base** and
[AG News (`fancyzhx/ag_news`)](https://huggingface.co/datasets/fancyzhx/ag_news).
Completed work includes smoke tests, a 2×3×10 Base pilot, and a verified
5×5×100 intermediate experiment. The full 50,000-prediction experiment has
**not** been run. These are methodology-inspired adaptations, not exact replication.

## Repository layout

```text
src/                  Dataset inspection, prompt/parser, model runtime,
                      crossed experiment design, runner, offline verification
tests/               Offline unit and mocked integration tests
experiments/          Pinned JSON configurations
report/               Methodology, frozen protocols, concise result reports
results/              Selected plans, manifests, summaries, CSVs and environments
requirements.txt      Pinned direct dependencies
```

Raw predictions, duplicated run-time source snapshots, detailed local audit
files, datasets, model weights, caches, virtual environments, and secrets are
excluded from the checkpoint. They remain on the original machine. See
[results/README.md](results/README.md) for the artifact inclusion policy.

## Setup

Use Python **3.11–3.13**; the experiments were run with Python **3.13.1**.
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
Requirements pin Datasets 3.6.0, PyTorch 2.7.1, Transformers 4.52.4, and
Hugging Face Hub 0.33.4. They are direct pins, not a full transitive lock;
per-run `environment.txt` files record the installed packages.

Recorded Base snapshots:

- Model: `Qwen/Qwen2.5-1.5B`, commit `8faed761d45a263340a0528343f099c05c9a4323`.
- Dataset: `fancyzhx/ag_news`, commit `eb185aade064a813bc0b7f42de02595523103ca4`.

The tested hardware was an M3 MacBook Air with 16 GB unified memory, using
MPS/BF16 and batch size one. BF16 weights require roughly 3 GB plus runtime
memory shared with macOS and other applications. BF16/MPS needs a compatible
macOS/PyTorch installation. The runner supports `--device cpu` with FP32, but
that changes precision, increases memory needs, and is slower. Identical results
across devices and library versions are not guaranteed. First reproduction needs
internet access to download the pinned public dataset/model into ignored
`data/` and `models/` directories. Installing requirements does not download weights.

## Run automated tests (no model inference or downloads)

```bash
python -m unittest discover -s tests -v
```

The 14 tests cover balanced/reproducible selections, shared positional
permutations, prompts, strict parsing, completion boundaries, known sensitivity
values, artifact writing with mocked predictions, and paired-plan checks.

Inspect the dataset separately:

```bash
python -m src.inspect_ag_news --revision eb185aade064a813bc0b7f42de02595523103ca4
```

This downloads data when needed but does not load a model.

## Reproduce Base experiments

Run from the repository root. These commands **perform inference**; choose a
fresh output directory for each run. Existing directories are never overwritten.
The `reproduced_*` names below avoid colliding with original local runs, whose
small checkpoint metadata directories already exist in a clone.

Five-article Base smoke test:

```bash
python -m src.base_smoke_test --output-dir results/reproduced_base_smoke
```

Base pilot: two sets × three shared permutations × ten common test articles.
The paired-plan check requires the exact original Instruct pilot selections:

```bash
python -m src.experiment --config experiments/base_pilot.json \
  --paired-plan results/pilot/plan.json \
  --output-dir results/reproduced_base_pilot --verbose
```

Base intermediate: five sets × five shared permutations × 100 common test articles:

```bash
python -m src.experiment --config experiments/base_intermediate.json \
  --output-dir results/reproduced_base_intermediate
```

All Base configurations use seed 180, eight demonstrations (two per class),
canonical alphabetical label/text sorting, shared seeded positional permutations,
a plain completion prompt ending in `Label:`, 1,000-character article truncation,
greedy single-beam decoding, a 12-token cap, and frozen `base-completion-v1` parsing.
The first generated newline or EOS closes the answer. Token-cap termination
without a boundary fails parsing. Only case-insensitive exact canonical class
names with surrounding whitespace are accepted; invalid outputs count as incorrect.
No chat template is used for Base. The earlier Instruct pilot used a chat template
and a different parser. Its artifacts remain separate and unchanged.

A full rerun writes raw predictions, exact prompts, plans, manifests, environment
records, summaries, timing, and source snapshots to the fresh ignored directory.
No full experiment is triggered automatically. Available options:

```bash
python -m src.experiment --help
```

## Verify reproduced outputs without inference

After the reproduced Base run has completed and its dataset/tokenizer are cached:

```bash
python -m src.verify_base_results results/reproduced_base_pilot
python -m src.verify_base_results results/reproduced_base_intermediate \
  --baseline results/base_pilot
```

The offline verifier checks every raw completion, generated token sequence,
reconstructed prompt, label, boundary, score, CSV accuracy, and JSON metric against
the pinned cache. It checks the recorded MPS/BF16 protocol, so a CPU/FP32 run does
not satisfy this specific audit. It refuses to replace an existing
`verification.json`. The original raw records are intentionally omitted from the
checkpoint; running this audit on original metadata-only directories in a fresh
clone requires obtaining the local raw artifacts or reproducing the run first.

## Preliminary results

| Run | Design (sets × permutations × articles) | Mean accuracy | Order sensitivity (pp) | Selection sensitivity (pp) | Selection/order | Parse failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Instruct pilot | 2×3×10 | 88.33% | 4.71405 | 5.00000 | 1.06066 | 2/60 |
| Base pilot | 2×3×10 | 73.33% | 4.71405 | 0.00000 | 0.00000 | 0/60 |
| Base intermediate | 5×5×100 | **71.36%** | **2.28182** | **1.50924** | **0.66142** | **28/2,500** |

The intermediate run had 1,784 correct predictions, 39m 44s timed inference,
and 51m 52s wall time. Parser failures were 25 `Science/Tech` and three `Health`
outputs; the strict parser was not relaxed. All 2,500 records were verified
locally before preparing this checkpoint.

For configuration accuracy matrix `A`, order sensitivity is the mean row-wise
standard deviation; selection sensitivity is the mean column-wise standard
deviation. Both use `ddof=0`. The ratio is selection/order, or null if order is
zero. Multiply sensitivities by 100 for percentage points.

Read [methodology](report/methodology.md), the frozen
[Base protocol](report/base_smoke_protocol.md), and the
[intermediate protocol](report/base_intermediate_protocol.md). Detailed findings:
[Instruct pilot](report/pilot_results.md),
[Base pilot comparison](report/base_pilot_results.md),
[Base intermediate results](report/base_intermediate_results.md).

## Limitations

This work does not establish a general superiority of order or selection effects.
The intermediate study uses one seed, five sets/permutations, and 100 unique test
articles repeated across configurations; 2,500 outcomes are not independent test
examples. Its first two sets and first three permutations match the Base pilot,
but only three test articles overlap, so their aggregate comparison is not fully
paired. Base/Instruct comparisons also change prompt and parser protocols.

The paper uses 10×10×500 and describes oversampling for class balance; this project's
intermediate sample uses stratified sampling without replacement (25 per class).
Exact original prompts, parsing, seeds/indices, checkpoint versions, truncation,
numerical settings, sorting details, and `ddof` are unconfirmed. Section 4
ordering selection with a development set is outside this implementation.
The project therefore remains an explicitly documented adaptation.
