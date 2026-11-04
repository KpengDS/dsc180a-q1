# Base intermediate experiment protocol

Configuration fixed before evaluation: `experiments/base_intermediate.json`.
Five distinct balanced demonstration sets × five shared positional permutations
× 100 common test articles = 2,500 predictions. Each set has eight training
examples, two per AG News class. Master seed: 180. The test sampler draws without
replacement with exactly 25 examples per class. The independent seeded streams
preserve the prior pilot's first two demonstration sets and first three
permutations. The larger test sample is generated anew; it is not required to
contain the prior ten articles, so aggregate comparisons are not fully paired.

The model, dataset, prompt, parser and generation pipeline are unchanged from the
Base pilot: Qwen/Qwen2.5-1.5B at commit
`8faed761d45a263340a0528343f099c05c9a4323`, AG News at
`eb185aade064a813bc0b7f42de02595523103ca4`, 1,000-character article truncation,
plain completion ending in `Label:`, frozen `base-completion-v1` strict parsing,
first-newline/EOS boundary, 12-token cap, MPS/BF16, greedy single-beam decoding,
batch size one and deterministic algorithms requested. Invalid completions count
as incorrect, without retries or parser adjustments. See
[base_smoke_protocol.md](base_smoke_protocol.md) for the frozen rules.

Accuracy is computed for each of the 25 configurations on the same 100 articles.
Order sensitivity averages row-wise accuracy standard deviations; selection
sensitivity averages column-wise standard deviations. Population standard
deviation (`ddof=0`) is retained. The ratio is selection/order, or null if order
sensitivity is zero. All raw generations, prompts, indices, parse errors, timing,
versions, and source snapshots are saved under `results/base_intermediate/`.

## Differences from the original paper

This remains a methodology-inspired adaptation of Section 3, not exact replication.
The paper's 10 sets × 10 permutations × 500 test articles are reduced to 5×5×100.
This project's evaluation sample is exactly balanced by stratified sampling without
replacement; the paper describes oversampling to balance evaluation classes.
The Base model family/size now aligns with the listed Qwen model, but the exact
paper checkpoint is unverified. Exact paper prompt, completion boundary, label
parser, decoding, truncation, numerical settings, source indices, seeds,
alphabetical tie/normalization rules, dataset versions and `ddof` remain
unconfirmed. These project choices are explicit and frozen; matching the model
name alone does not establish replication. Development-set ordering optimization
from Section 4 is not part of this run. See [methodology.md](methodology.md).

Only this intermediate run is authorized here; no 50,000-prediction experiment
is launched. Prior artifacts are protected from overwrite and checked by hashes.
No evaluation choices will be changed in response to the observed results.
