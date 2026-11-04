# Base smoke protocol v1

This is a separate five-example pipeline check, not a sensitivity experiment or
an exact paper replication. No previous Instruct artifact is rescored or replaced.

- Model: `Qwen/Qwen2.5-1.5B`, pinned to resolved Hub commit
  `8faed761d45a263340a0528343f099c05c9a4323` in `experiments/base_smoke.json`.
- Dataset: existing pinned AG News snapshot. The shared seeded design selects
  one eight-shot training set, two examples per class, and canonical alphabetical
  ordering. Five common test articles are sampled near-balanced without replacement.
  Exact selections and seeds are saved to `plan.json`.
- Prompt: the existing text prompt builder, applied directly as a completion;
  eight labeled articles and one query ending exactly with `Label:`.
  No system/role messages, chat template, or assistant prefix is applied.
  Articles keep the existing 1,000-character truncation.
- Generation: BF16 on MPS, greedy decoding, one beam, batch size one, model eval,
  inference mode, fixed seed, and requested deterministic algorithms. Twelve new
  tokens maximum; no retries.
- Boundary: stop at the first generated LF newline, EOS, or token limit. Decode
  only generated tokens when looking for newlines. An initial newline is a real
  boundary, not a blank line to skip. If a token crosses the newline, retain the
  entire decoded output but score only the part before its first newline.
- Parser: `base-completion-v1`. Strip surrounding whitespace and compare
  case-insensitively to exactly World, Sports, Business, or Sci/Tech. Reject empty
  strings, prefixes (including `Human:` and `Label:`), quotes, punctuation,
  explanations, and ambiguous labels. Reaching the token cap without newline or
  EOS is a failure even if the text resembles a label. Failures become null
  predictions and count as incorrect. This stricter policy is specific to this
  new run; the old Instruct parser is unchanged.
- Timing: synchronize MPS before and after generation; report per-prediction and
  total inference seconds (generation and decoding, excluding loading,
  tokenization, and result writing). Wall timestamps additionally include loading.

`src/base_smoke_test.py` delegates to the shared experimental runner, model
loader, and prediction writer. It runs one set × one ordering × five articles.
The default `results/base_smoke/` must not exist: the runner refuses overwrite.
Use `--output-dir` for a new run. Model ID, model revision, and prompt mode are
configurable via JSON or `--model-id`, `--model-revision`, `--prompt-mode`.
The old experiment defaults still select Instruct/chat. Do not pair a new model
ID with another model's pinned revision. CPU is available for portability but
was not the requested execution backend.

Artifacts include raw outputs, scored completion text, termination reasons,
generated token IDs, parser failures, exact rendered prompts, accuracy, timing,
model/dataset hashes, config, source snapshot, and installed environment. The
single-cell sensitivity summary is structurally supported by the shared runner
but has no substantive interpretation as a sensitivity estimate.

Any future parser, boundary, or prompt change requires a new protocol version
and a new result directory. The initial policy is fixed before evaluation;
observed failures will be reported rather than used to relax scoring.
