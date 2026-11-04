# Small AG News sensitivity pilot

This is a framework validation run, not a full experiment or exact replication.
Configuration: `experiments/pilot.json`, master seed 180, two distinct balanced
8-shot demonstration sets, three shared permutations, ten common test articles.
Actual test class counts: World 3, Sports 2, Business 2, Sci/Tech 3.
The model was Qwen2.5-1.5B-Instruct on MPS/BF16, using pinned snapshots and
unchanged greedy generation. See `results/pilot/manifest.json` for exact versions.

| Demonstration set | Permutation 0 | Permutation 1 | Permutation 2 |
| --- | --- | --- | --- |
| 0 | 100% | 90% | 90% |
| 1 | 80% | 90% | 80% |

The mean configuration accuracy was 88.33%. Population grouped standard
deviations (`ddof=0`) gave:

- Order sensitivity: 0.0471404521, or **4.71 percentage points**.
- Selection sensitivity: 0.0500000000, or **5.00 percentage points**.
- Selection/order ratio: **1.06066**.

Of 60 predictions, two failed parsing: test index 808 in set 1 under permutations
0 and 2 returned `Human: World`. Both were recorded with null predicted labels
and scored incorrect according to the preexisting strict parser. No post hoc
parser changes or selective retries were made. These are prediction-format
failures, not execution failures. The run completed normally.

All six configurations used the same test indices and the same three positional
permutations. The result artifact was checked against every individual prediction
and all configuration accuracies recomputed. Eight automated tests passed,
including offline integration and known-value sensitivity calculations. Total
timed inference was about 56.85 seconds (excluding model/data loading).

The sample has only ten test examples, so accuracy moves in 10-point steps;
two sets and three permutations provide very limited coverage. The near-equal
sensitivities in this pilot do not establish a general conclusion about order
versus selection or reproduce the paper's published results. There was no
model training, development-set ordering selection, or full experiment.

Artifacts are in `results/pilot/`: `plan.json`, `manifest.json`,
`predictions.jsonl`, `configurations.csv`, `summary.json`, `environment.txt`,
and the source snapshot. The snapshot captures the implementation at run time;
this results note was written afterward. See [methodology.md](methodology.md)
for definitions, protocol choices, and replication gaps.
