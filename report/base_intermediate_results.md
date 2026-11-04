# Base intermediate sensitivity experiment

The authorized intermediate experiment completed: **5 demonstration sets × 5
shared permutations × 100 common test articles = 2,500 predictions**. Each set
contains eight training demonstrations, two per class. Seed 180, MPS/BF16,
greedy decoding, 12-token cap, and the frozen `base-completion-v1` plain prompt,
newline/EOS boundary and strict parser were retained without modification.

Configuration: `experiments/base_intermediate.json`. Model:
`Qwen/Qwen2.5-1.5B@8faed761d45a263340a0528343f099c05c9a4323`.
Dataset: `fancyzhx/ag_news@eb185aade064a813bc0b7f42de02595523103ca4`.
The 100 test articles are unique and exactly balanced, 25 per class.

## Results and comparison

| Metric | Intermediate (5×5×100) | Previous Base pilot (2×3×10) |
| --- | ---: | ---: |
| Mean configuration accuracy | **71.36%** | 73.33% |
| Order sensitivity, percentage points | **2.28182** | 4.71405 |
| Selection sensitivity, percentage points | **1.50924** | 0.00000 |
| Selection/order ratio | **0.66142** | 0.00000 |
| Parsing failures | **28/2,500 (1.12%)** | 0/60 |
| Timed inference | **2,384.304 s (39m 44s)** | 53.498 s |
| Wall time | **3,111.749 s (51m 52s)** | 62.254 s |

The mean accuracy is 1,784 correct outcomes out of 2,500, equivalently the mean
of the 25 equal-size configuration accuracies. The console rounds this to 71.4%;
the saved exact value is 0.7136. Timed inference averages 0.954 seconds per
prediction. Wall time additionally includes startup, tokenization, result writing,
and other work outside the timed generation/decoding section.

Accuracy matrix (percent), rows = demonstration sets, columns = shared permutations:

| Set | Permutation 0 | Permutation 1 | Permutation 2 | Permutation 3 | Permutation 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 74 | 74 | 71 | 71 | 71 |
| 1 | 72 | 74 | 72 | 72 | 66 |
| 2 | 70 | 75 | 72 | 66 | 69 |
| 3 | 72 | 76 | 72 | 66 | 71 |
| 4 | 73 | 72 | 72 | 71 | 70 |

Population standard deviations (`ddof=0`) were retained. Averaging row standard
deviations gives order sensitivity 0.0228181780; averaging column standard
deviations gives selection sensitivity 0.0150924094. Dividing selection by order
gives 0.6614204440. The denominator is nonzero.

The intermediate accuracy is approximately **1.97 percentage points below** the
previous Base pilot. Observed order sensitivity is lower and selection sensitivity
is now nonzero. The first two demonstration sets and first three positional
permutations match the pilot, but only **three test articles overlap** between the
100-article sample and the prior ten-article sample. The comparison therefore
changes evaluation coverage as well as the number of sets/permutations; it is
not fully paired and does not isolate the effect of experiment size.

Within this run, observed order sensitivity exceeds selection sensitivity. This
is a descriptive finding for one seed and sampled design, not a general or
statistically established dominance claim. The 100 articles repeat across 25
configurations, so the 2,500 outcomes are not independent test examples.
The pilot's zero selection sensitivity did not establish a general absence of
selection effects, and this run does not justify tuning the evaluation protocol.

## Parsing and execution

All 2,500 completions terminated at a generated newline. No execution errors,
EOS-only terminations, token-cap failures, selective retries, or rescoring occurred.

- Test index 18 returned ` Science/Tech\n\n` in all 25 configurations. `Science/Tech`
  differs from the frozen canonical `Sci/Tech`, so all 25 were parser failures.
- Test index 3422 returned ` Health\n\n` in three configurations. `Health` is not
  one of the four permitted labels, so all three were parser failures.

These 28 outputs remain null predictions and count as incorrect. No alias,
parser, prompt, or boundary changes were introduced after observing results.
Failure records and raw text are retained in `predictions.jsonl` and
`verification.json`. The shared model runtime, Base protocol, prompt builder,
and design code were hash-checked against the previous Base pilot source and
remain unchanged.

## Verification and preservation

All **14 automated tests passed before inference**. The separate offline audit
`src/verify_base_results.py` then checked every saved record against the pinned
cached dataset and tokenizer, without loading model weights or running inference:

- Exactly 2,500 unique set/permutation/article keys and 100 common test indices
  per configuration; balanced demonstrations and shared valid permutations.
- Ground-truth labels, exact reconstructed plain prompts, input token counts,
  decoded generated token IDs, completion boundaries, strict parser decisions,
  predicted labels, correctness flags, and per-record timing validity.
- Every configuration CSV and summary accuracy, count correct, and parser-failure
  count; independent direct-formula standard deviations and ratio; summed runtime.
- Pinned model/dataset revisions, MPS/BF16, deterministic greedy settings, frozen
  parser metadata, source selection prefixes, and prior artifact SHA-256 hashes.

All checks passed. All **77 prior result files remain byte-identical**. The
verifier was developed separately from the running evaluator and checked first
against the old Base pilot. It did not change predictions or evaluation outputs.
Its source is saved as `results/base_intermediate/verification_script.py`;
the evaluator's earlier run-time source snapshot is preserved separately.

Outputs under `results/base_intermediate/`: `plan.json`, `manifest.json`,
`predictions.jsonl`, `configurations.csv`, `summary.json`, `environment.txt`,
`source/`, `verification.json`, `verification_script.py`, and `comparison.json`.
The results note was written after the evaluator's source snapshot was taken.

## Differences from the original paper

This is a methodology-inspired Section 3 adaptation, not exact replication.
The paper uses **10 sets × 10 permutations × 500 test articles**, rather than
5×5×100. We balance test classes by stratified sampling without replacement;
the paper describes oversampling underrepresented classes. The Base model name
now matches the listed family/size, but the exact original checkpoint is not
verified. The paper's exact prompt, parser, completion boundary, decoding,
truncation, numerical settings, source indices, seeds, alphabetical normalization
and tie rules, dataset versions, and standard-deviation `ddof` are unconfirmed.
Our explicit choices remain fixed. Section 4 development-set ordering optimization
is outside this run. See [methodology.md](methodology.md) and the pre-run
[intermediate protocol](base_intermediate_protocol.md).

## Reproduce

```bash
python -m unittest discover -s tests -v
python -m src.experiment --config experiments/base_intermediate.json \
  --output-dir results/base_intermediate
python -m src.verify_base_results results/base_intermediate \
  --baseline results/base_pilot
```

The output directory and verification file must not already exist. For a repeat,
choose a new output directory and substitute it in the verification command.
The audit can also check prior result hashes with `--previous-hashes PATH`, given
a pre-run JSON mapping of file paths to SHA-256 values. This run's hash mapping
is retained in its verification report. No full 50,000-prediction experiment,
commit, or push was performed.
