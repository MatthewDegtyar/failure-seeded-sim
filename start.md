# start.md — Failure-Seeded Simulation: Validation Experiment

## Purpose of this file
You are an AI coding agent. This file is your brief for building a **single validation
experiment**. Do not build the product. Do not build the full pipeline. Build the one
experiment that tests whether the core idea works. Read the whole file before writing code.

---

## 1. The thesis being tested

Robotic manipulation models fail on rare, hard-to-predict edge cases. The failure
manifold is thin and high-dimensional. We claim:

> When a failure happens, you can **invert it** — recover the simulator scene parameters
> that caused it — and **explode a synthetic neighborhood** around that recovered scene.
> Training data drawn from that neighborhood beats blind domain randomization at fixing
> the failure, by a meaningful margin.

If true → there is a company here. If false → we learn that cheaply. This experiment
tests **only that claim**, entirely in simulation (sim-to-sim).

**Out of scope for this experiment:** sim-to-real gap, real hardware, LLMs, perception
from images, per-client calibration, the full service. Do not build them.

---

## 2. Budget and infrastructure constraints

- Total cloud spend ceiling: **USD 200** (AWS credits available).
- This is a **CPU physics-sim** workload. Do **not** provision GPUs.
- Prefer running locally where possible; use AWS only to parallelize data generation.
- Use spot instances (c-series) for parallel sim runs. Tear them down when idle.
- Track spend. If projected spend exceeds USD 120, stop and report.

---

## 3. The simulated task (keep it tiny — this is mandatory)

A parallel-gripper grasp of a **single object** on a **flat table**.

Physics engine: **MuJoCo** (free, fast, well-supported) — fall back to PyBullet only if
MuJoCo setup blocks you.

The **scene parameter vector** has two groups:

**Geometric parameters (group A — treated as observable / near-known):**
- object position x, y
- object yaw

**Physical parameters (group B — treated as hidden / must be inferred):**
- friction coefficient
- object mass
- center-of-mass offset (one axis is enough to start)
- (optional) grasp approach angle

Total dimensionality: **6–8 numbers**. Do not exceed this. A larger scene space will
fail for boring reasons and teach us nothing.

**MuJoCo contact-parameter caveat:** MuJoCo's contact behavior is governed by `solref`
and `solimp`, which are NOT intuitive physical quantities (stiffness/damping). Do NOT
put contact compliance in group B for the first run — it does not map cleanly to a
physical parameter and the inversion will fit nonsense. Keep group B to **friction,
mass, and CoM offset** (all of which map to real MuJoCo physical properties). Contact
compliance is a stretch goal only after the basic loop works.

---

## 4. Experiment steps

### Step 1 — Map the failure manifold
- Sample N scenes across the parameter space.
- Run the grasp in MuJoCo, record success/failure for each.
- Confirm failures are **sparse and structured** (a thin region, not uniform noise).
- Deliverable: a dataset of (params, outcome) and a short note confirming structure.

### Step 2 — Designate one "client failure"
- Pick one specific **failed** configuration.
- Treat its parameters as **unknown ground truth** — store them, but downstream code
  is not allowed to read them.
- Produce the "observation": the multi-view trajectory of that failure
  (see Step 3 on observation richness).

### Step 3 — Inversion (THE CORE — split into 3a and 3b)

**3a — Geometric recovery.** Assume calibrated multi-view observation. For this
sim-to-sim experiment, hand the geometry (group A params) to the solver as
**near-perfect known constraints**. (Justification: real calibrated multi-view rigs
recover geometry directly via triangulation; simulating that as "known" is fair.)

**3b — Physical-parameter recovery (the hard half — this is the moat).**
- This is **system identification** in the literature — fitting simulator parameters
  so simulated motion matches observed motion. It is a well-established technique;
  the novelty of this project is *what we do with it* (failure-seeding), not the
  identification itself. Use standard methods, do not reinvent.
- With group A fixed, search **only** over group B (friction, mass, CoM offset).
- Method: black-box optimization — **CMA-ES** (preferred) or Bayesian optimization.
  CMA-ES is the standard choice for MuJoCo parameter fitting. Use the `cma` Python
  package. (A differentiable-MuJoCo / MJX gradient-based approach is faster and is a
  documented option *if* CMA-ES is too slow — but CMA-ES first; dimensionality is tiny.)
- Objective: propose group-B params → simulate → compare simulated **trajectory**
  to observed **trajectory** → minimize discrepancy (squared difference of object
  positions/velocities over time is a fine objective).
- **Use the full trajectory, not the end state.** Transient motion (acceleration,
  onset of tipping/sliding) carries the parameter information. End state alone is
  degenerate.
- Output a **posterior / distribution** over group-B params, not a point estimate.
  Different physical params can produce identical motion (degeneracy) — represent
  that honestly as a spread.

**Key metric — inversion error:** distance between recovered group-B posterior and
the true hidden values. Report this. If it is large, the project's crux has failed
cheaply — that is still a valid, valuable result. Report it plainly, do not bury it.

### Step 4 — Neighborhood exploration / amplification
- Take the recovered posterior from 3b and **explode** it: sample a cluster of scenes
  in a neighborhood around the recovered params, spanning the posterior's spread.
- Optionally run an outward search for nearby configs that also fail (maps the local
  failure-manifold extent). The simulator itself is the verifier — no adversarial
  network needed.

### Step 5 — The payoff test (proves the VALUE, not just the mechanism)

Train a small failure-predictor (params → fail/succeed; a small MLP or random forest)
under **three data regimes, with total sample count held EQUAL**:

- **Regime A — baseline:** uniform domain randomization across the whole space.
- **Regime B — our method:** scenes from the exploded neighborhood around the
  *recovered* config (Step 4).
- **Regime C — oracle ceiling:** scenes exploded around the *true* hidden config.

Evaluate all three on a **held-out test set of failures** drawn from the real failure
region (near the designated client failure).

**Claim is validated if:** B beats A by the pre-registered margin, AND B is close to C.
- B vs A = "failure-seeding beats blind randomization."
- B vs C = "how much our inversion error costs us."

### Step 6 — Robustness sweep
- Repeat Step 3b with increasing **observation noise** added to the trajectory.
- Plot inversion error and final B-vs-A margin as noise rises.
- This tells us how fragile the moat is.

---

## 5. Pre-registration (DO THIS BEFORE RUNNING STEP 5)

Before running the payoff test, write `preregistration.md` containing:
- The exact success threshold (e.g. "B beats A by >= 15% on held-out failure accuracy").
- The exact metrics and how they are computed.
- The held-out test set definition.

Decide these with nothing invested in the outcome. Do not edit this file after Step 5
results are known. If results miss the threshold, report that honestly.

---

## 6. Deliverables

1. Working code, organized: `sim/`, `inversion/`, `experiment/`, `analysis/`.
2. `preregistration.md` (frozen before Step 5).
3. `results.md`: inversion error, the three-regime comparison, the noise sweep, plots.
4. A short honest verdict: is the thesis supported, partially supported, or refuted?
5. Total AWS spend used.

---

## 7. Rules of engagement

- **Scope discipline:** if tempted to add scope, stop and flag it instead.
- **Honesty over optimism:** a clean refutation is a successful experiment. Do not
  tune the experiment until it passes.
- **Reproducibility:** fixed random seeds, logged configs, deterministic where possible.
- **Report early if:** MuJoCo setup blocks you, failures are NOT structured (Step 1),
  inversion error is severe, or spend approaches the cap.
- **Ask before:** changing the task, raising scene dimensionality, provisioning GPUs,
  or coupling the experiment to any specific robot-learning model.

---

## 8. Open assumptions to flag to the human

- Observation is assumed to be **multi-view trajectory video**, not a single snapshot.
  If implementation forces a snapshot-only assumption, STOP — physical-parameter
  recovery (3b) is not well-posed from a snapshot. Flag it.
- Geometry is assumed recoverable to near-perfect accuracy from multi-view. If that
  assumption needs relaxing, flag it before proceeding.