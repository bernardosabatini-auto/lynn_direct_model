# Higher-Order Interaction Detectability — Brief

Companion document to `masking_sim.py`. Written so a fresh Claude instance can
pick up the project cold.

## One-line goal
Test whether the Lynn (2026, *Nat. Phys.*) "direct dependencies" maximum-entropy
framework correctly detects when neuronal computation **requires** higher-order
(multi-input, non-separable) interactions — or whether it systematically
**under-reports** them. We identify two distinct reasons it does: (1) **input
correlation** and (2) **binarization of the signal**. Both are properties of how
the data is represented, not of the underlying computation.

## The claim we are stress-testing
For binary output `y` and inputs `x`, the maximum-entropy model matching only the
marginal rate `⟨y⟩` and the pairwise co-activities `⟨y x_i⟩` is logistic,

    P(y=1 | x) = σ(b + Σ_i w_i x_i)

i.e. exactly logistic regression of `y` on `x`. Lynn defines (in bits):

- `S_tot`  = marginal output entropy `H(⟨y⟩)`
- `S_dir`  = conditional entropy of the fitted logistic model `⟨ H[σ(η(x))] ⟩_x`
- variance explained by direct dependencies = `(S_tot − S_dir)/S_tot`

He reports ≈90% in mouse hippocampus and visual cortex and concludes neurons are
≈ perceptrons (direct dependencies suffice, interactions unnecessary).

**The paper never tests higher-order *detection* at scale.** Its only higher-order
probes are (a) a 2-input XOR toy (Fig. 1g, independent inputs) and (b) predicting
triplet+ co-activity rates from the pairwise model on real data (Fig. 4, which
assumes the conclusion and is tested only on positive co-activities). Its one
dynamical simulation (SI eq. S22) is **linear-in-inputs** (a time-delayed logistic
model with no interaction terms), so it cannot test higher-order detection. There
is no simulation with strong built-in multi-input interactions checked against the
method. That gap is what we fill.

## Mechanism — both failures run through one quantity
Write each input as mean + fluctuation, `x = μ + δ`. A pairwise interaction
factorizes as

    x_i x_j = μ_iμ_j + μ_iδ_j + μ_jδ_i + δ_iδ_j

Everything except the centered product `δ_iδ_j` is constant-or-linear and is
absorbed into the logistic weights for free. How much of even `δ_iδ_j` leaks into
the linear fit is set by the **input coskewness** `E[δ_i² δ_j] = Cov(δ_iδ_j, δ_i)`.

### Axis 1 — input correlation (`rho`)
- **Independent inputs** → `δ_iδ_j` is maximally orthogonal to the marginals →
  interaction is detectable; Lynn's metric correctly reports low direct-explained.
- **Correlated inputs** → the centered product acquires a linear projection →
  absorbed into effective linear weights → Lynn reports *high* direct-explained
  and the triplet test passes, even though the generator uses a strong interaction.

Same mechanism, opposite conclusion, driven only by input statistics.

### Axis 2 — representation (`input_mode`): binary vs analog
The coskewness `E[δ_i² δ_j]` depends on the representation:

- **Binary** `x∈{0,1}`: `x² = x`, so `δ_i² = (1−2p)x_i + p²` is *affine* in `x_i`.
  Coskewness collapses to an ordinary covariance, `E[δ_i²δ_j] = (1−2p)·Cov(x_i,x_j)`
  — nonzero whenever the rate `p ≠ ½` and inputs are correlated. The binary identity
  *creates* the leakage channel; sparse + correlated is its worst case.
- **Symmetric continuous** (zero-mean Gaussian): all odd joint moments vanish, so
  `E[δ_i²δ_j] = 0` at **every** correlation → the linear model captures *none* of the
  interaction.
- **Skewed analog** (lognormal; like ΔF/F or rate): nonzero coskewness → *partial*
  masking, growing with correlation.

Perfect-correlation limit (`x_i = x_j`) makes it concrete: binary gives
`x_i x_j = x_i` (fully absorbed); Gaussian gives `x_i x_j = x_i²` with
`Cov(x_i², x_i) = E[x_i³] = 0` (un-absorbable even at ρ=1).

**Second, separate consequence of `x²=x`:** every function on `{0,1}^n` is a
multilinear polynomial with no self-powers, so a single binary input can only act
linearly — `f(x_i) = f(0) + (f(1)−f(0))x_i`. Binarization therefore *erases all
single-input nonlinearity* (f–I curves, thresholds, dendritic supralinearities)
into a bare weight, leaving only multi-input interaction as the sole possible
nonlinearity — which Axis 1 then masks. Binarization is doubly favorable to the
"perceptron" conclusion.

This is not forced by biology: Lynn's headline 90% is from **calcium imaging**
(an analog signal he chooses to binarize) at coarse bins (V1 at 667 ms, where the
native quantity is a graded spike count). An analog treatment escapes both breaks.

### The conceptual error
Lynn conflates a **predictive** claim ("a linear model predicts the binarized
activity well") with a **mechanistic** one ("the neuron is a perceptron").
Correlated inputs and binarization are exactly the conditions that make that gap
large.

## What `masking_sim.py` does
Fixed generator across all conditions; output `y` is always binary (a neuron
fires or not). We vary only how the **inputs** are represented and correlated.

- Generator: **purely higher-order** — `η = strength · Σ_pairs sign · δ_iδ_j`,
  `linear_strength = 0`. So under independent inputs the linear model can explain
  ≈none of the recoverable information by construction (`phi_direct ≈ 0`).
- Inputs: one-factor latent model `z = √ρ·g + √(1−ρ)·ε`, mapped to
  `binary | gaussian | lognormal`. `rho` sweeps the correlation.
- Two models fit by IRLS/Newton (GPU matmuls): **direct** `[1, x]` (= Lynn's
  maxent model) and **interaction** `[1, x, all pairwise products]`.
- Because the generator is known, `S_true` is computed as an oracle.

Metrics (bits, held-out test set):

| metric | meaning |
|---|---|
| `phi_direct = (S_tot−S_dir)/(S_tot−S_true)` | **Lynn's direct share of recoverable info** (his implicit claim: ≈1) |
| `phi_int = (S_tot−S_int)/(S_tot−S_true)` | sanity: interaction model's share (≈1 in all regimes) |
| `nll_gap = NLL_direct − NLL_int` | predictive necessity of interactions (>0 ⇒ interactions help on held-out data) |
| `S_tot, S_true, S_dir, S_int, I_true` | raw entropies / total recoverable info |

## Predicted result (confirmed on CPU at small `L`)
- **gaussian**: `phi_direct ≈ 0` at *every* `rho`; `nll_gap` stays large →
  interactions never absorbed, always necessary.
- **lognormal** and **binary**: `phi_direct` **rises from ≈0 toward 1 as `rho`
  grows**; `nll_gap` shrinks → interaction progressively absorbed into linear
  weights. (binary rise scales with `(1−2p)`; sparser ⇒ stronger effect.)
- **`phi_int ≈ 1` in every cell** → the information is genuinely recoverable; only
  the *linear* model is fooled.

Headline figure (`masking_by_representation.png`, from `plot_modes`):
`phi_direct` vs `rho`, one line per `input_mode` — identical nonlinear mechanism,
conclusion flips from "interactions essential" to "perceptron" purely with the
representation.

## Run
```
python masking_sim.py        # run_mode_comparison over binary/gaussian/lognormal
                             #   + plot_modes -> masking_by_representation.png
```
Single-representation correlation sweep:
```python
from masking_sim import Config, run_sweep
run_sweep(Config(input_mode="binary"))
```

### Scaling on the DGX Spark (GH200)
- `device` auto-selects `cuda`. Use `dtype=torch.float32`.
- Push `L_train=L_test=1_000_000`, `K=32–64`, `n_pairs` up to `K(K−1)/2`.
  Interaction features are `1 + K + K(K−1)/2`; the `L×D` design matrix dominates
  memory (e.g. 1e6 × ~2000 × 4B ≈ 8 GB — fine in 128 GB unified).
- For very large `L`/`D`, set `hessian_chunk` (e.g. 100_000) to accumulate the
  IRLS Hessian in sample chunks and bound peak memory.
- Cheap to repeat across `seed` for error bars on `phi_direct(rho)`.

## Extensions (not yet implemented; orthogonal to the core point)
- **Triplet-prediction reproduction** of Lynn's Fig. 4 test, split by
  true-interaction vs random pairs, to show his detection tool lacks power on
  correlated binary data (predicted: low mispredict fraction even when
  interactions are essential).
- **Greedy optimal-input selection** (Lynn's `n*` procedure) with distractor
  inputs — input *discovery* is a separate question and does not bear on the
  linear-vs-interaction dissociation, which is why the core sim hands the model
  the true input set.
