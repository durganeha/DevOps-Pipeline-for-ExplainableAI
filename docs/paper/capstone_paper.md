# Building a DevOps Pipeline for Explainable AI (XAI): An Automated System for Validating Model Fairness

*[Author Name], [Institution], [Department]*

---

## Abstract

*[150-250 words. State the problem (bias drift in deployed ML systems),
your approach (an automated CI/CD fairness gate + SHAP explainability +
CUSUM drift monitoring), and your key result (e.g. "the CUSUM detector
identified injected covariate drift within N batches of its onset, while
the CI fairness gate correctly blocked M% of intentionally biased model
variants in testing").]*

## 1. Introduction

- The problem: ML models in high-stakes domains (credit, hiring, criminal
  justice) can be biased, and that bias is often invisible until an
  incident or audit.
- The gap: standard DevOps/MLOps pipelines validate functional correctness
  (tests passing, latency, uptime) but rarely validate *fairness* as a
  first-class, automatically-enforced gate.
- The added complication: even a model that is fair at deployment time can
  become unfair later as the real-world population drifts — a dimension
  most fairness tooling does not address continuously.
- Your contribution: an end-to-end pipeline that (a) blocks unfair models
  in CI, (b) explains individual decisions via SHAP, and (c) continuously
  monitors deployed models for bias drift via CUSUM.

## 2. Related Work

*[Your existing literature survey slides map directly here. For each of your
~10 papers: 1-2 sentences on what they did, then a sentence connecting it to
your work — e.g. "Unlike [X], which evaluates fairness only at training
time, our pipeline enforces this check automatically at every CI run."]*

Suggested subsections:
- 2.1 Fairness metrics and mitigation (AIF360, Fairlearn, disparate impact
  literature)
- 2.2 Explainable AI methods (SHAP, LIME)
- 2.3 Concept/model drift detection (CUSUM, SPRT, sequential monitoring)
- 2.4 MLOps/DevOps for ML (CI/CD for ML systems, model governance)

## 3. Proposed System

### 3.1 Architecture
*[Insert docs/architecture_diagram.png. Describe the four layers: CI/CD
trigger → training & evaluation → fairness gate → deployment artifact, and
the separate always-on drift-monitoring loop.]*

### 3.2 Fairness Gate
- Metric: Disparate Impact, DI = P(favorable | unprivileged) / P(favorable
  | privileged); threshold 0.8 (the "80% rule").
- Also implemented: Equal Opportunity Difference, Statistical Parity
  Difference (for a fuller picture; DI is the primary gating metric).
- Enforcement: implemented as a GitHub Actions workflow
  (`.github/workflows/fairness-gate.yml`) that fails the CI job — and
  therefore blocks the merge — when DI falls below threshold.

### 3.3 Explainability Layer
- SHAP (`LinearExplainer`, since the baseline model is Logistic Regression)
  used for (a) global feature importance and (b) per-prediction explanation.
- *[If you swap to a tree-based model, mention `TreeExplainer` here
  instead, and re-run the SHAP experiments.]*

### 3.4 Drift Monitoring
- CUSUM (Cumulative Sum) control-chart method applied to the DI score
  computed over sequential batches of incoming data.
- One-sided lower CUSUM: S_t = max(0, S_{t-1} + (target − DI_t) − k); alarm
  when S_t exceeds threshold h.
- Models a realistic scenario: the deployed model is fixed, but the
  real-world feature distribution (e.g., income, credit history) shifts
  between demographic groups over time — a covariate-shift-driven fairness
  degradation, not a change in the model itself.

## 4. Experimental Setup

- **Dataset:** Loan Prediction Problem Dataset (Analytics Vidhya / Kaggle).
  *[Report final row count, feature list, and class balance once you've
  swapped in the real CSV at data/raw/loan_data.csv.]*
- **Protected attribute:** Gender.
- **Model:** Logistic Regression (scikit-learn), standardized features.
- **Drift simulation:** synthetic batches (n=12, 800 samples/batch) with a
  gradually widening covariate gap between groups starting at batch 5,
  simulating a real-world economic shift after deployment.
- **Baselines for comparison:** *[if time permits, compare against (a) no
  monitoring at all, (b) a naive fixed-window fairness check every N
  batches, to show CUSUM's earlier detection.]*

## 5. Results

*[Fill in with your actual numbers/plots. Suggested tables/figures:]*
- Table: DI, accuracy, and pass/fail outcome for N model variants (some
  intentionally biased, some fair) run through the fairness gate.
- Figure: `cusum_drift_plot.png` — DI per batch and CUSUM statistic,
  annotated with the detection point.
- Table: detection latency (batches until alarm) across a few different
  values of h (alarm threshold) and k (slack), to show the
  sensitivity/false-alarm tradeoff.
- Figure: `shap_summary.png` — global feature importance.

## 6. Discussion

- What the fairness gate catches vs. misses (e.g., DI alone doesn't capture
  Equal Opportunity violations — mention your other two metrics here).
- Limitations of CUSUM (requires choosing k/h; sensitive to batch size;
  assumes i.i.d. within a batch).
- Practical deployment considerations: false-alarm cost vs. detection
  latency tradeoff a real MLOps team would face.

## 7. Conclusion and Future Work

- Summary of contribution: a working, automatically-enforced fairness gate
  + explainability + drift monitoring, integrated directly into CI/CD.
- Future work: SPRT-t and Sequential Bayesian Factor methods as additional
  drift detectors; MLflow-based experiment/version tracking; automated
  retraining trigger on drift alarm; extending to non-binary protected
  attributes and intersectional fairness.

## References

*[Your existing ~10-paper literature survey list goes here, formatted per
your institution's required citation style (IEEE is common for CS capstones
— check with your guide).]*