# Base smoke test result

The separate five-article Base smoke test completed successfully under the frozen
`base-completion-v1` policy, with seed 180 and MPS/BF16 greedy generation.
Model: `Qwen/Qwen2.5-1.5B`, commit
`8faed761d45a263340a0528343f099c05c9a4323`.

| Test index | True label | Predicted label | Raw generated text (escaped) |
| --- | --- | --- | --- |
| 1025 | Business | Business | ` Business\n\n` |
| 1176 | Sports | Sports | ` Sports\n\n` |
| 653 | World | World | ` World\n\n` |
| 2930 | Sci/Tech | Business | ` Business\n\n` |
| 2422 | Sci/Tech | Business | ` Business\n\n` |

Accuracy: **3/5 (60%)**. Parsing failures: **0**. Total timed inference:
**4.856 seconds** (approximately 0.971 seconds per article), excluding model
loading/download, tokenization, and result writing. Per-record times are saved.

All completions stopped on a generated newline; tokens included two newlines,
so the full raw text was retained while only the preceding first line was
parsed. No prompt, parser, boundary, or evaluation policy was changed after
observing predictions. The two classification mistakes were Sci/Tech articles
predicted as Business. No execution errors occurred.

The run uses one canonically ordered balanced set and a five-article near-balanced
test sample selected by the shared framework. Its samples differ from the original
Instruct smoke test; accuracies are not a controlled Base/Instruct comparison.
The single-cell sensitivity values in the shared summary are not meaningful
sensitivity estimates. No full sensitivity experiment was run.

All 13 automated tests passed, including strict parsing, balanced plain prompts,
newline stopping, and a mocked runtime asserting no chat template calls.
Saved model hash, dtype, decoding, prompt structure, count of predictions, and
accuracy were verified. All previous result files were checked by SHA-256 and
remained byte-identical.

See `results/base_smoke/` for `manifest.json`, `plan.json`, `predictions.jsonl`,
`configurations.csv`, `summary.json`, environment versions, and source snapshot.
The snapshot captures the implementation before this results note was written.
Protocol details are in [base_smoke_protocol.md](base_smoke_protocol.md).
