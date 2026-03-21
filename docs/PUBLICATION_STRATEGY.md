# Publication Strategy for Failure-Mode Study

Project: Automated Endoscopy Image Labelling
Last Updated: 2026-03-14
Purpose: Define realistic paper quality targets and evidence standards.

## 1) Recommended Target Path

Short-term target:
- High-quality conference/workshop paper focused on negative results and governance failures in automated pseudo-labelling.

Mid-term target:
- Journal extension with stronger statistical validation, external data, and expanded ablations.

## 2) Realistic Venue Tiering

Tier A (near-term, realistic with current dataset size):
- Applied medical AI conferences/workshops (MICCAI workshops, MIDL-style venues, EMBC tracks, imaging informatics workshops).
- Strength: clear failure taxonomy + reproducibility + practical implications for weak supervision pipelines.

Tier B (after one more robust experimental cycle):
- Mid/high-impact journal in medical imaging/biomedical AI.
- Typical expectation: stronger experimental breadth, confidence intervals, external validation, and clearer clinical framing.

## 3) Quartile/Impact Guidance

If targeting journal publication quality:
- Minimum recommended quality bar: Q2.
- Better strategic target after strengthening evidence package: Q1.

Practical note:
- Quartile and impact vary by indexing category and year; the safer planning approach is to target evidence quality first, then map to venue.

## 4) Evidence Requirements by Target

Conference-quality minimum:
1. Clear method history and failure taxonomy (v1-v13 timeline).
2. Reproducible run artifacts for key versions (v10-v13).
3. Negative-control analysis and gate-failure discussion.
4. Qualitative visualization panel with consistent selection protocol.

Journal-quality minimum:
1. All conference requirements.
2. Statistical uncertainty (bootstrapped confidence intervals on key rates).
3. External validation or temporal split robustness checks.
4. Formal pseudo-label quality proxies and sensitivity analyses.
5. Stronger reproducibility package (env fingerprints, immutable manifests, run index).

## 5) Current Position (as of v13)

Current strength:
- Excellent failure narrative and reproducible recent runs.

Current limitation:
- Gate can pass despite unsafe Normal FP (governance failure mode).

Interpretation:
- Strong conference-ready failure-mode story after artifact packaging cleanup.
- Not yet Q1 journal-ready without additional validation and statistics.

## 6) Recommended Next Milestones

1. Freeze a complete paper artifact bundle for v10-v13.
2. Implement gate-v2 (include Normal FP criterion) and run comparative ablation.
3. Add uncertainty intervals for key metrics (coverage, FP, edge rate).
4. Prepare figure/table index and manuscript-aligned naming.
