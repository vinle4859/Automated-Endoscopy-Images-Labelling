# Next Steps Prioritization

Project: Trustworthy Endoscopic AI Pipeline  
Date: 2026-03-14  
Purpose: Define execution order after v13 replication and paper-artifact packaging.

## Priority 0 (Do Now)

1. Finalize failure-mode paper package for v10-v13 under `results/paper_artifacts/`.
2. Freeze evidence mapping: version -> run directory -> log -> summary -> table/figure IDs.
3. Add environment fingerprints (`env.json`) per key run where missing.
4. Lock a gate-v2 draft that includes explicit Normal-control constraints.

Definition of done:
- Core paper assets are complete and cross-referenced.
- Reproducibility metadata is present for all key runs.

## Priority 1 (Next Experimental Cycle)

1. Run gate-v2 comparative ablation on the current v13 operating point.
2. Report uncertainty intervals for key rates (coverage, Normal FP, edge rate).
3. Quantify bbox-to-signal expansion as a pseudo-label precision proxy.

Decision rule:
- If Normal FP remains above target after gate-v2 and threshold tuning, open v14.
- If gate-v2 can control Normal FP while preserving coverage, continue paper-first.

## Priority 2 (Conditional v14 Branch)

1. Prototype hybrid proposal-refinement: PatchCore proposals + Med-SAM refinement.
2. Compare against v13/gate-v2 using identical splits and reporting schema.
3. Keep v14 as a targeted mitigation study, not the default immediate path.

Entry criteria:
- Priority 1 completed and evidence shows unresolved specificity failure.

## Recommended Default Track

Paper-prep first, then conditional v14.

Rationale:
- Current evidence already supports a strong failure-mode contribution.
- Immediate value is highest in reproducibility, governance framing, and statistical rigor.
- v14 should be justified by a measured residual gap, not started by default.
