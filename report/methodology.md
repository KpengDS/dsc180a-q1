# AG News order and selection sensitivity

## Relationship to the paper

Source: Li et al., [Order Matters: Rethinking Prompt Construction in In-Context
Learning, arXiv:2511.09700v1](https://arxiv.org/html/2511.09700v1), Sections 3–3.2.
The paper compares a crossed matrix of demonstration sets and permutations,
using a canonical classification order: alphabetical labels, then alphabetical
examples within each label. A fixed positional permutation is applied across
sets. Its sensitivity measures average row-wise and column-wise standard
deviations, with selection/order as the relative-importance ratio. AG News uses
eight shots. Section 3 uses 10 sets, 10 permutations, and 500 test examples;
classification evaluation sets are balanced through oversampling.

This implementation is a **methodology-inspired adaptation**, not an exact
replication. The reviewed text does not provide exact training/test indices,
random seeds, full prompt/chat templates, decoding and label-parser settings,
standard deviation `ddof`, alphabetical normalization/tie rules, dataset/model
commit hashes, or the detailed oversampling procedure. Appendix tables list
`Qwen/Qwen2.5-1.5B`; this project uses **Qwen2.5-1.5B-Instruct**, which also differs.
These details require author code or clarification before claiming replication.
Development-set ordering optimization from Section 4 is outside this framework.

## Explicit project protocol

Each distinct demonstration set contains eight **training-split** examples by
default, sampled without replacement, two per class. Distinct sets may overlap;
they may not be identical. Sets are not required to be mutually disjoint.
Canonical sorting uses the actual class names (`Business`, `Sci/Tech`, `Sports`,
`World`), then full, untruncated article strings using case-sensitive Python
Unicode ordering; the dataset index breaks exact text ties.

One list of unique random permutations of positions 0–7 is sampled and reused
for every set. If canonical indices are `s`, permutation `p` gives the prompt
indices `[s[position] for position in p]`. Canonical order is a reference, not an
extra evaluated configuration unless selected as the sole ordering strategy.
Sampling distinct permutations is uniform by rejection. Selection is independent
of ordering and test sampling: each uses a dedicated SHA-256-derived seed from
the master seed and the stream name. This prevents changing the number of sets
from changing the permutation list or test sample. Inference uses the master seed.

All configurations evaluate the **same** held-out test indices. The pilot uses
10 unique test articles with near-balanced class counts (3,3,2,2). Remainder
classes are chosen randomly; no oversampling or duplicate test articles occurs.
This is a deliberate deviation from exact class balance. Counts are recorded.
No test results are used to select demonstrations, tune prompts, or choose a
winning ordering. Ten examples produce coarse 10-percentage-point accuracy steps.

The existing prompt builder, label parser, and working Qwen runtime are shared
with the smoke test. Articles are truncated independently at 1,000 characters by
default, before prompt construction; canonical ordering uses the original texts.
This is character truncation, not token truncation. The rendered prompt includes
eight labeled examples plus an unlabeled query and the tokenizer's chat wrapper.
Greedy generation uses `do_sample=False`, one beam, at most 12 new tokens, model
`eval()`, and `inference_mode()`. MPS uses BF16, CPU uses FP32, with deterministic
PyTorch algorithms requested. Hardware/library changes can still change results.
Parse failures are explicitly retained as null predictions and scored incorrect.
Runtime/load failures mark the run failed, leaving partial records rather than
reporting a complete sensitivity result.

## Calculations implemented here

Let `A[i,j] = correct_predictions / N_test` for set `i`, shared positional
permutation `j`, and the common test sample. Accuracies are fractions in [0,1].
For a vector `x` of length `n`:

```text
std(x; ddof) = sqrt(sum((x[t] - mean(x))**2) / (n - ddof))
order sensitivity     = mean_i(std(A[i, :]; ddof))
selection sensitivity = mean_j(std(A[:, j]; ddof))
relative ratio         = selection sensitivity / order sensitivity
```

This project's explicit default is population standard deviation (`ddof=0`); `--std-ddof 1`
selects sample standard deviation and requires both matrix dimensions >= 2.
Sensitivities multiplied by 100 are percentage points. A zero order denominator
produces a JSON `null` ratio plus an explanation, including when both are zero.
The code records all row/column standard deviations and the accuracy matrix.
These are descriptive grouped variations, not independent causal variance
components, significance tests, or standard deviations of the per-set means.
With only two sets and three permutations, the pilot cannot establish which
source dominates or reproduce published sensitivity estimates.

## Configuration and artifacts

`experiments/pilot.json` specifies the authorized 2 × 3 × 10 pilot (60 total
predictions). CLI overrides support `num_sets`, `num_permutations`, `num_test`,
`per_class`, `seed`, `selection`, `ordering`, `test_sampling`, `max_chars`,
`std_ddof`, `device`, and pinned snapshot revisions. Choices:

- `selection=balanced` preserves per-class shots; `uniform` samples `4*per_class`
  examples without class constraints. In uniform mode, varying class counts can
  contribute to measured selection sensitivity; this is a separate protocol.
- `ordering=random` generates the shared unique permutation list;
  `canonical` requires exactly one permutation and provides a baseline. One
  permutation yields zero descriptive order sensitivity with `ddof=0`.
- `test_sampling=near_balanced` spreads the remainder across random classes;
  `uniform` samples test articles without class constraints.

A run creates an exclusive output directory and refuses to overwrite an existing
one. Use a new path for a repeat. `--plan-only` resolves/loads the dataset and
saves the selections, with no model loading or generation.

- `plan.json`: effective configuration, stream seeds, canonical demonstration
  indices, shared permutations, actual ordered indices, common test indices/counts.
- `manifest.json`: status, timestamps, model/dataset resolved commit hashes,
  dataset fingerprints, template, tokenizer chat template, generation settings,
  dtype, device, Python/platform and core library versions.
- `predictions.jsonl`: one row per configuration/article, with indices, true and
  predicted labels, correctness, parser error, raw generation, exact rendered
  prompt, token count, and timing. Rows are flushed as they complete.
- `configurations.csv`: accuracy, count correct, parse failures, set/permutation
  IDs, canonical and ordered indices; list-valued CSV fields contain JSON.
- `summary.json`: configuration results, accuracy matrix, sensitivities, ratio.
- `environment.txt` and `source/`: installed package versions and snapshots of
  source, tests, configuration, methodology, requirements, and README. Snapshots
  include source files that were untracked at run time.

Snapshot hashes are pinned to the previously verified local model and dataset.
The model is loaded once for the entire run, and inference uses batch size one.
Weights and downloaded data remain under ignored directories.
