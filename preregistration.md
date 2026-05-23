# Pre-registration — Step-5 Payoff Test

Frozen 2026-05-22, **before** running `experiment.step5_payoff`. Step-1
through Step-4 outputs are saved to `data/`. Inversion error from Step 3
is known and reported in `README.md`; this pre-registration is written
without observing any classifier outcomes.

## 1. What is being tested

Whether a failure-predictor trained on scenes drawn from the recovered
posterior's neighborhood (regime **B**) beats a predictor trained on
blind domain randomization (regime **A**) by a pre-specified margin on
a held-out test set that lies in the same local region as the
designated client failure.

## 2. Datasets (already produced and frozen in `data/step4_training_sets.npz`)

- **Regime A** — uniform DR over `PARAM_BOUNDS`, N = 1500.
- **Regime B** — uniform box around the ABC-posterior mean (per-axis
  half-width = 0.40 × axis range), N = 1500.
- **Regime C** — uniform box around the *true hidden* group-B params
  (same half-width), N = 1500.
- **Held-out test set** — `n_test_per_class = 400` failures + 400
  successes, both drawn from a 0.50 × axis-range window around the
  truth with the same group-A jitter and stratified to be class-balanced.
  Total test N = 800, success rate = 0.500 by construction.
- All four splits use disjoint random seeds (training seed 100, test
  seed branched from the same generator after training data is drawn).

## 3. Classifier

`sklearn.ensemble.RandomForestClassifier` with:
- `n_estimators = 200`
- `max_depth = None`
- `min_samples_leaf = 2`
- `class_weight = "balanced"` (so the >90% fail-rate training sets
  don't collapse to "always predict fail")
- `random_state = 0`

Inputs: the 6-d full parameter vector (`obj_x, obj_y, obj_yaw,
friction, mass, com_x`). Labels: 1 = success, 0 = failure. **Positive
class for metrics: fail = 0.**

Three classifiers are trained independently on regimes A, B, C with
the **same** test set.

## 4. Headline metric

**Balanced accuracy** on the held-out test set:

`BAcc = 0.5 * (TPR_success + TPR_fail)`

The test set is balanced 50/50, so BAcc = standard accuracy here, but
we state BAcc explicitly so the metric remains meaningful if a future
test set is reweighted.

Secondary reported metrics (no thresholds): standard accuracy, F1 of
failure class, ROC-AUC. These are exploratory.

## 5. Pre-registered thresholds

The thesis is considered:

- **Validated** if BOTH:
  - `BAcc(B) - BAcc(A) ≥ 0.10` (10 percentage points), AND
  - `BAcc(C) - BAcc(B) ≤ 0.05` (B within 5 pp of oracle ceiling).
- **Partially supported** if `BAcc(B) - BAcc(A) ≥ 0.05` but
  the second condition fails, OR vice versa.
- **Refuted** if `BAcc(B) - BAcc(A) < 0.05`.

The above are evaluated on a single (training-seed, test-seed) pair as
produced by Step 4. We additionally average across 5 classifier seeds
(`random_state ∈ {0..4}`) to reduce variance; the average is what is
compared against the thresholds.

## 6. What we will NOT do

- We will not change `WINDOW_FRAC`, `TEST_WINDOW_FRAC`, classifier
  hyperparameters, the metric, or the thresholds after observing the
  Step-5 result.
- We will not pick a different client failure if the result misses
  the threshold. If it misses, that is the reported result.
- We will not exclude classifier seeds with bad luck.

## 7. Step-6 robustness sweep (descriptive, no threshold)

We will rerun Step 3 with observation-position Gaussian noise of
σ ∈ {0, 0.5mm, 1mm, 2mm, 5mm, 10mm} added per-axis to the recorded
trajectory, then redo Steps 4+5 for each noise level (only the B
regime needs to be re-simulated; A and C are noise-independent for
their data sources). We plot:

- Inversion error (overall mean relative error from Step 3)
  vs noise σ.
- `BAcc(B) - BAcc(A)` margin vs noise σ.

The robustness plot is descriptive. We pre-register no pass/fail
threshold on Step 6 — its purpose is to characterize moat fragility.
