# Failure Mode Study Readiness

Project: Trustworthy Endoscopic AI Pipeline
Last Updated: 2026-03-14
Purpose: Verify whether stored artifacts are sufficient for an academic paper on automated labelling failure modes.

## 1) Most Important Documents and What They Store

- VERSION_REGISTRY.md
  - Complete version timeline, parameter deltas, run outcomes, and interpretation.
  - Primary longitudinal narrative for methods and results.

- docs/AUTOMATED_ANNOTATION_STRATEGIES.md
  - Method-level rationale and historical negative results (why prior methods failed).
  - Supports methodology section and ablation context.

- docs/PROJECT_REVIEW_AND_ROADMAP.md
  - Risk/gap framing, governance decisions, and execution priorities.
  - Useful for discussion/threats-to-validity and future work.

- results/logs/v13_advanced_20260313_154100.log
  - Full command-line trace of the first complete v13 advanced run.

- results/logs/v13_advanced_20260314_163000.log
  - Full command-line trace of an independently executed repeat v13 run.
  - Confirms reproducibility of the same operating point and key outcomes.

- results/runs/20260313_161640_generate/run_info.json
  - Exact configuration snapshot for the first complete v13 run.

- results/runs/20260314_171000_generate/run_info.json
  - Exact configuration snapshot for the repeat v13 run.

- results/runs/20260313_161640_generate/summary.json
  - Compact machine-readable outcome summary for v13 run A.

- results/runs/20260314_171000_generate/summary.json
  - Compact machine-readable outcome summary for v13 run B.

- results/runs/20260313_161640_generate/artifacts_manifest.json
  - Canonical artifact list for v13 run A.

- results/runs/20260314_171000_generate/artifacts_manifest.json
  - Canonical artifact list for v13 run B.

## 1b) Version Coverage Matrix (What Exists vs Missing)

| Version | Status | Structured run evidence in workspace | Coverage quality for paper |
|---|---|---|---|
| v1-v5 | Historical | No preserved per-run structured artifacts | Low (narrative-only) |
| v6 | Baseline historical | Metrics documented in VERSION_REGISTRY, no local structured run folder in current workspace | Medium (table-level) |
| v7-v8 | Historical | Narrative + metrics in VERSION_REGISTRY, no complete local structured run package | Medium (table-level) |
| v9 | Failed (OOM) | Narrative + OOM analysis in VERSION_REGISTRY | Medium (failure documented) |
| v10 | Evaluated | `results/runs/20260311_184717_session/*`, plus finetune run metadata | High |
| v11 | Evaluated | `results/runs/20260312_185336_generate`, `results/runs/20260313_112259_generate` | High |
| v12 | Evaluated | Full log + artifacts documented under v11/v12 section | High |
| v13 | Evaluated | `results/runs/20260313_161640_generate` and `results/runs/20260314_171000_generate` with summaries/manifests | High |

Immediate implication:
- The study currently has strongest reproducibility from v10 onward.
- For v1-v9, the paper should present them as historical design stages with
  reconstructed evidence from registry tables, not full artifact-complete runs.

## 2) Are Current Artifacts Enough for a Failure-Mode Paper?

Short answer: Almost, but not yet fully publication-grade.

What is already strong:
- Reproducible run traces with config + summary + log.
- Explicit negative controls (Normal/test FP reporting).
- Cross-version evolution and failure-analysis narrative.
- Visual diagnostics per class (20 images each) with overlays.

What is still missing for stronger academic rigor:
- Fixed gate specification per experiment (with immutable versioned gate schema).
- Statistical uncertainty (confidence intervals/bootstrapped intervals) for key rates.
- Explicit train/val/test split manifests saved as files per run.
- Environment fingerprint per run (git commit hash, Python/PyTorch/Ultralytics versions, GPU/CPU, CUDA).
- Figure index mapping (which image files are used in manuscript figures/tables).
- Formal annotation quality metrics beyond coverage/edge:
  - pseudo-label precision proxy,
  - Normal FP under thresholded operating points,
  - bbox-to-signal expansion distribution quantiles.

- Consolidated run index for all available experiments:
  - one document linking version → run_dir → log → summary.

## 3) Minimum Additions Recommended Before Writing

1. Save run environment metadata in each run folder:
   - git hash, package versions, hardware summary.
2. Save split manifests:
   - train/val/test filename lists per class.
3. Save a gate schema file:
   - criteria, thresholds, objective weights, and rationale.
4. Add one run-comparison table artifact:
   - version, selected params, coverage, Normal FP, edge rate, gate pass.
5. Archive a curated visualization panel list for paper figures.

6. Build and freeze an evidence index document:
   - all versions,
   - all run directories available,
   - exact metric source files.

## 3b) Artifact Preparation Checklist (Paper Package)

Required package per key version run (v10-v13):

1. `run_info.json` and `run_info.txt`
2. `summary.json` and `summary.txt`
3. `artifacts_manifest.json`
4. `quality_gate.json` snapshot copy in run folder
5. confidence CSV snapshots in run folder
6. visualization index (CSV with selected figure IDs)
7. environment metadata (`env.json`)

Recommended canonical folder for manuscript assets:

- `results/paper_artifacts/`
  - `tables/`
  - `figures/`
  - `run_snapshots/`
  - `stats/`

Current status (created in workspace):

- `results/paper_artifacts/RUN_INDEX.md`
- `results/paper_artifacts/tables/failure_taxonomy.md`
- `results/paper_artifacts/tables/results_table_v10_v13.md`
- `results/paper_artifacts/tables/v13_replication_table.md`
- `results/paper_artifacts/figures/FIGURE_INDEX.csv`
- `results/paper_artifacts/run_snapshots/GATE_SCHEMA_v1.json`
- `results/paper_artifacts/stats/v13_replication_check.json`

Snapshot freezing status (completed for v13 runs):

- `results/runs/20260313_161640_generate/snapshots/`
  - confidence CSV copies + `quality_gate.json` + `SHA256SUMS.txt`
- `results/runs/20260314_171000_generate/snapshots/`
  - confidence CSV copies + `quality_gate.json` + `SHA256SUMS.txt`

## 4) Current v13 Interpretation for Manuscript

- v13 outcomes are replicated across two independent complete runs (2026-03-13 and 2026-03-14).
- v13 resolves under-coverage but significantly increases Normal false positives.
- Therefore, current gate pass is not equivalent to trustworthy pseudo-label quality.
- This is a governance failure mode and should be reported as a central finding.

## 5) Current Evidence Index (Available Now)

- v10 session: `results/runs/20260311_184717_session/`
- v11 run A: `results/runs/20260312_185336_generate/`
- v11 run B / v12 context: `results/runs/20260313_112259_generate/`
- v13 run A: `results/runs/20260313_161640_generate/`
- v13 run B: `results/runs/20260314_171000_generate/`
- v13 log A: `results/logs/v13_advanced_20260313_154100.log`
- v13 log B: `results/logs/v13_advanced_20260314_163000.log`
- v13 snapshot A checksums: `results/runs/20260313_161640_generate/snapshots/SHA256SUMS.txt`
- v13 snapshot B checksums: `results/runs/20260314_171000_generate/snapshots/SHA256SUMS.txt`

## 6) Prioritized Execution Order

See `docs/NEXT_STEPS_PRIORITIZATION.md` for the ranked post-v13 action plan.
