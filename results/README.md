# Checkpoint result artifacts

The checkpoint includes small, unchanged artifacts from the recorded runs:
`pilot/` (Instruct), `base_smoke/`, `base_pilot/`, and `base_intermediate/`.

| Included file | Purpose |
| --- | --- |
| `plan.json` | Selected demonstration/test indices, permutations, seeds, configuration |
| `manifest.json` | Resolved revisions, prompt/parser/generation settings, environment, status and timestamps |
| `summary.json` | Configuration accuracies, accuracy matrix, grouped sensitivities, counts and timing where recorded |
| `configurations.csv` | Compact configuration-level results |
| `environment.txt` | Installed package versions for the recorded run |
| `comparison.json` (where present) | Compact descriptive comparison with the earlier pilot |

Raw `predictions.jsonl` files are excluded, including the intermediate file
(7,138,341 bytes). The original one-file smoke outputs also contain raw prompts
and are excluded. Duplicated `source/` snapshots, detailed `verification.json`
audit/hash inventories and the duplicated verifier script are local-only.
Audit inventories may contain machine-specific paths. Nothing was deleted,
rewritten, sanitized, or rescored to prepare this submission.

The proposed checkpoint source, configs and plans support reproduction; they are not a
substitute for the raw prediction records when independently auditing a historic
run. Follow the root README to reproduce into fresh `results/reproduced_*`
directories and audit those outputs. New run directories are ignored by default.
Summaries in `report/` describe the already completed verification and limitations.

The `.gitignore` uses an explicit run-directory and filename allowlist so future
raw/generated outputs are not accidentally proposed for a commit.
