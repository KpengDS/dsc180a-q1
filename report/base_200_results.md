# Base sensitivity experiment with 200 test articles

The approved experiment completed using the existing pipeline without source,
prompt, parser, or methodology changes: **5 demonstration sets × 5 shared
permutations × 200 common test articles = 5,000 predictions**. Each demonstration
set contains eight training examples, two per class. The 200 unique test articles
are exactly balanced at 50 per AG News class. Seed 180, MPS/BF16, batch size one,
greedy decoding, 1,000-character truncation, and the frozen `base-completion-v1`
completion boundary and strict parser were retained.

Model: `Qwen/Qwen2.5-1.5B`, commit
`8faed761d45a263340a0528343f099c05c9a4323`.
Dataset: `fancyzhx/ag_news`, commit
`eb185aade064a813bc0b7f42de02595523103ca4`.
The existing `experiments/base_intermediate.json` was reused with the sole
configuration override `--num-test 200`. The complete effective configuration
is saved in `results/base_200/plan.json` and `manifest.json`. An equivalent
standalone configuration is provided in `experiments/base_200.json`.

## Verified results

| Metric | New 5×5×200 run | Previous 5×5×100 run |
| --- | ---: | ---: |
| Mean accuracy | **71.92%** | 71.36% |
| Order sensitivity, percentage points | **2.16867** | 2.28182 |
| Selection sensitivity, percentage points | **1.61177** | 1.50924 |
| Selection/order ratio | **0.74321** | 0.66142 |
| Parsing failures | **9/5,000 (0.18%)** | 28/2,500 (1.12%) |
| Timed inference | **4,991.977 s (83m 12s)** | 2,384.304 s (39m 44s) |
| Wall time | **5,010.399 s (83m 30s)** | 3,111.749 s (51m 52s) |

There were **3,596 correct predictions** out of 5,000. Accuracy is equivalently
the mean of the 25 equal-size configuration accuracies. Timed inference includes
generation and decoding; wall time includes loading, tokenization, output writing,
and other untimed work. Runtime is descriptive rather than a controlled hardware
benchmark.

Accuracy matrix (percent): rows are demonstration sets and columns are shared
positional permutations.

| Set | Permutation 0 | Permutation 1 | Permutation 2 | Permutation 3 | Permutation 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 77.0 | 74.5 | 75.0 | 71.5 | 69.5 |
| 1 | 73.0 | 71.5 | 70.0 | 71.0 | 68.0 |
| 2 | 71.5 | 74.0 | 71.5 | 68.0 | 71.0 |
| 3 | 76.0 | 72.5 | 71.5 | 69.5 | 70.0 |
| 4 | 76.0 | 72.0 | 69.5 | 73.5 | 70.5 |

Using unchanged population standard deviations (`ddof=0`), order sensitivity is
the mean row standard deviation, **0.021686665856687146**. Selection sensitivity
is the mean column standard deviation, **0.016117707872201587**. Their
selection/order ratio is **0.743208198932601**. The denominator is nonzero.

## Comparison and interpretation

All five demonstration sets and all five positional permutations match the
previous intermediate experiment exactly. The existing seeded sampler draws a
new balanced test sample when its size changes; it does not append 100 articles
to the previous sample. **32 articles overlap**, yielding 800 matched
configuration/article outcomes. All 800 have identical rendered prompts, raw
outputs, and predicted labels across the two runs.

Mean accuracy increased by **0.56 percentage points**. Observed order sensitivity
decreased slightly, selection sensitivity increased slightly, and the ratio
increased from 0.66142 to 0.74321. These are descriptive comparisons across
different evaluation samples, not evidence that increasing sample size improved
the model. The smaller parser-failure rate likewise reflects the new evaluation
sample under the same fixed parser, not a protocol improvement.

Observed order sensitivity exceeds selection sensitivity in this run, but the
study uses one seed, five sets and five permutations. The same 200 articles are
repeated across configurations, so the 5,000 outcomes are not independent test
examples. No significance or general dominance claim follows from these results.

## Parsing, tests, and preservation

All 5,000 completions terminated at a generated newline. Nine outputs failed the
unchanged strict parser: eight ` Science/Tech\n\n` outputs (test index 7348 in
five configurations and 6368 in three) and one ` Entertainment\n\n` output
(test index 5307). Neither spelling is a permitted canonical class. Each remains
a null prediction and counts as incorrect. No retries, alias additions,
rescoring, or evaluation adjustments occurred. No execution errors occurred.

All **14 existing automated tests passed before inference and again afterward**;
there were no test failures. The existing offline verifier audited every record
against the pinned cached dataset and tokenizer, checking prompts, true labels,
token counts, generated-token decoding, completion boundaries, parser decisions,
predictions, correctness, and timing. It independently recomputed all 25
configuration accuracies, the accuracy matrix, both sensitivities, and their
ratio, and checked CSV/JSON consistency. All checks passed.

Preservation hashes confirmed that all **137 pre-run files** under source,
tests, configurations, report documentation, and saved results remain
byte-identical. This new results note was written after that check. No existing
file was modified; no commit, push, or Gradescope submission was performed.

## Artifacts and reproduction

All new experiment artifacts are under `results/base_200/`: `plan.json`,
`manifest.json`, `predictions.jsonl`, `configurations.csv`, `summary.json`,
`environment.txt`, `source/`, `verification.json`, and `comparison.json`.
Small metadata and summary files in this directory are included in Git. Raw
predictions, detailed verification records, and source snapshots remain ignored.
Historic result files were not overwritten. The run-time source snapshot predates this results note.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.experiment \
  --config experiments/base_intermediate.json \
  --num-test 200 --output-dir results/base_200
.venv/bin/python -m src.verify_base_results results/base_200 \
  --baseline results/base_intermediate
```

The result directory and verification file already exist locally. For a repeat,
use a fresh output directory in both commands; the runner refuses overwrites.

This remains a methodology-inspired adaptation of the paper's Section 3, not an
exact replication. It uses 5×5×200 rather than 10×10×500, and stratified sampling
without replacement instead of the paper's described oversampling. Exact paper
prompts, parsing, seeds, checkpoint versions and several implementation details
remain unconfirmed. No development-set ordering optimization is implemented.
See [methodology.md](methodology.md), the frozen
[Base protocol](base_smoke_protocol.md), and the
[previous intermediate report](base_intermediate_results.md).
