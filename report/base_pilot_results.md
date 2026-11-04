# Paired Base and Instruct sensitivity pilots

Only the small 2 sets × 3 permutations × 10 articles design was run for Base:
60 predictions, seed 180, eight balanced demonstrations, MPS/BF16 greedy
inference. Model `Qwen/Qwen2.5-1.5B` was pinned to
`8faed761d45a263340a0528343f099c05c9a4323`.

The Base plan was checked against `results/pilot/plan.json` before inference.
All canonical demonstration selections, actual ordered indices, positional
permutations, test indices and class counts match exactly. The dataset revision,
truncation length and standard-deviation convention also match. Reference plan
path and SHA-256 are recorded in the new manifest.

| Metric | Base | Original Instruct |
| --- | ---: | ---: |
| Mean configuration accuracy | 73.33% | 88.33% |
| Order sensitivity, percentage points | 4.71405 | 4.71405 |
| Selection sensitivity, percentage points | 0.00000 | 5.00000 |
| Selection/order ratio | 0.00000 | 1.06066 |
| Parse failures / 60 | 0 | 2 |
| Summed inference seconds | 53.498 | 56.849 |
| Run wall seconds, including startup | 62.254 | 66.243 |

Accuracy matrices, with sets as rows and shared permutations as columns:

```text
Base:      [[0.8, 0.7, 0.7],
            [0.8, 0.7, 0.7]]
Instruct:  [[1.0, 0.9, 0.9],
            [0.8, 0.9, 0.8]]
```

Sensitivity uses population standard deviations (`ddof=0`), averaged across
rows for order and across columns for selection. The Base column values are
identical across these two sets, giving zero observed selection sensitivity.
Its row values vary, so the zero ratio is defined rather than a zero-denominator
case. This does not establish absence of selection sensitivity in general.

Base was 15 percentage points below Instruct in mean accuracy on this paired
pilot. Across the 60 matched configuration/article outcomes, both were correct
42 times, only Base was correct twice, only Instruct was correct 11 times, and
both were incorrect five times. These are descriptive counts; the same ten test
articles repeat across configurations, so the 60 outcomes are not independent.
No significance test or general performance conclusion is warranted.

The comparison changes both model and prompt/output protocol. Base used the
frozen `base-completion-v1` plain prompt ending in `Label:`, first-newline/EOS
boundary, 12-token cap, and strict exact-class parser. Instruct used the chat
wrapper and its existing parser/generation policy. Parse failures count as
incorrect in each run under its own predefined policy. The results therefore
compare complete protocols and do not isolate a model-only causal effect. Neither
Instruct output nor previous scoring was changed. See
[base_smoke_protocol.md](base_smoke_protocol.md) and [methodology.md](methodology.md).

## Reproduce and verify

```bash
python -m unittest discover -s tests -v
python -m src.experiment --config experiments/base_pilot.json \
  --paired-plan results/pilot/plan.json --output-dir results/base_pilot --verbose
```

The output directory must not already exist; choose a fresh directory to repeat.
`--paired-plan` fails before model loading/inference if design indices or
protocol-independent design settings differ. All 14 automated tests passed,
including checks that pairing rejects changed test indices or truncation settings.

The saved records were independently checked for 60 unique configuration/article
keys, ten common test indices per configuration, valid Base completions, and
correct accuracy flags. All six CSV accuracies and the JSON summary metrics were
recomputed from individual predictions using direct standard-deviation formulas;
timing and failure counts were also checked. No inference or parser errors occurred.
SHA-256 checks confirmed all 48 prior result files were unchanged.

Artifacts in `results/base_pilot/` include `predictions.jsonl` (raw text,
completion text, termination, parse errors, timing and exact prompts),
`configurations.csv`, `summary.json`, `plan.json`, `manifest.json`, environment
and source snapshots, plus `comparison.json` and `verification.json`.
The source snapshot predates this results note. No 50,000-prediction experiment
was run, and no commit or push was performed.
