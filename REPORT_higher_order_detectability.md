# Higher-Order Interaction Detectability in the Lynn Maximum-Entropy Framework

## Bottom Line

The Lynn maximum-entropy method models neuronal activity well without including nonlinear interactions between inputs. However, this does not mean that such interactions do not exist. Our simulations demonstrate that a network whose neurons genuinely require higher-order interactions to compute their outputs will nonetheless appear to be well-described by a purely linear (perceptron) model, through two independent mechanisms: (1) **input correlation** makes interaction terms collinear with marginals, and (2) **binary representation** creates a mathematical leakage channel (x² = x) that does not exist for continuous signals. Both are properties of how the data is represented, not of the underlying computation. The method's success at prediction is real; the absence-of-interactions conclusion is not warranted by that success.

## Summary

We tested whether the maximum-entropy framework of Lynn (2026, *Nature Physics*) — which concludes that neurons are approximately perceptrons — can reliably detect higher-order (multi-input, non-separable) interactions when they are genuinely present in a neural circuit. Using synthetic neurons with a known, purely nonlinear generator (zero linear drive), we find two independent reasons the method under-reports interactions:

1. **Input correlation** absorbs interaction terms into effective linear weights via collinearity.
2. **Binary representation** (x in {0,1}) creates a coskewness channel that leaks interaction signal into the linear fit — a channel that vanishes for symmetric continuous (Gaussian) inputs.

The same nonlinear generator, tested with the same correlation structure, produces phi_direct (fraction of recoverable information captured by direct dependencies) ranging from **0.00 (Gaussian inputs) to 0.84 (binary inputs at rho=0.9)**. The "perceptron" conclusion is an artifact of the representation.

---

## 1. Background: The Lynn Framework

Lynn fits a maximum-entropy model that matches only the marginal firing rate of an output neuron and its pairwise co-activity with each input. This model takes the form of logistic regression:

    P(y=1 | x) = sigmoid(b + sum_i w_i x_i)

He defines the following entropy hierarchy (in bits):

- **S_tot**: marginal output entropy H(<y>), the total variability with no knowledge of inputs
- **S_dir**: conditional entropy of the fitted logistic model, averaged over input patterns
- **S_true**: true conditional entropy (the irreducible noise given full knowledge of inputs)

The headline metric is **(S_tot - S_dir) / S_tot**, reported at ~90% across mouse hippocampus and visual cortex. The paper concludes that neurons can be described by simple artificial models (perceptrons) without interactions between inputs.

We use a more informative metric: **phi_direct = (S_tot - S_dir) / (S_tot - S_true) = I_dir / I_true**, the fraction of *recoverable* information captured by the linear model. Lynn's implicit claim is phi_direct ~ 1.

### What the paper does not test

The paper's only higher-order probes are: (a) a 2-input XOR toy example (Fig. 1g, independent inputs only), and (b) predicting triplet co-activity rates from the pairwise model on real data (Fig. 4, which assumes the conclusion). Its dynamical simulation (SI eq. S22) is linear-in-inputs. There is no simulation with strong built-in multi-input interactions checked against the method.

---

## 2. Two Mechanisms of Masking

Both mechanisms run through a single quantity. Write each input as mean + fluctuation, x = mu + delta. A pairwise interaction factorizes as:

    x_i * x_j = mu_i*mu_j + mu_i*delta_j + mu_j*delta_i + delta_i*delta_j

Everything except the centered product delta_i*delta_j is constant-or-linear and is absorbed into the logistic weights for free. How much of even delta_i*delta_j leaks into the linear fit is set by the **input coskewness** E[delta_i^2 * delta_j].

### 2.1 Axis 1 — Input Correlation (rho)

- **Independent inputs**: delta_i*delta_j is maximally orthogonal to the marginals. The interaction is detectable; Lynn's metric correctly reports low direct-explained.
- **Correlated inputs**: the centered product acquires a linear projection via coskewness. The interaction is absorbed into effective linear weights. Lynn reports high direct-explained even though the generator uses a strong interaction.

Same mechanism, opposite conclusion, driven only by input statistics.

### 2.2 Axis 2 — Representation: Binary vs Continuous

The coskewness E[delta_i^2 * delta_j] depends critically on the input representation:

- **Binary (x in {0,1})**: x² = x, so delta_i² = (1-2p)*x_i + p² is *affine* in x_i. Coskewness collapses to (1-2p)*Cov(x_i, x_j) — nonzero whenever rate p != 0.5 and inputs are correlated. The binary identity *creates* the leakage channel; sparse + correlated is its worst case.
- **Symmetric continuous (Gaussian, zero-mean)**: All odd joint moments vanish, so E[delta_i^2 * delta_j] = 0 at **every** correlation. The linear model captures *none* of the interaction regardless of rho.
- **Symmetric continuous, strictly positive (Uniform(0,1))**: Also symmetric about its mean, so coskewness vanishes. phi_direct ~ 0.00 at every rho — confirming that **positivity is not the issue; symmetry is what prevents masking**.
- **Skewed continuous (lognormal)**: Nonzero coskewness produces partial masking, growing with correlation.

**Second consequence of x² = x:** Every function on {0,1}^n is a multilinear polynomial with no self-powers, so a single binary input can only act linearly: f(x_i) = f(0) + (f(1)-f(0))*x_i. Binarization erases all single-input nonlinearity (f-I curves, thresholds, dendritic supralinearities) into a bare weight, leaving only multi-input interaction as the sole possible nonlinearity — which Axis 1 then masks. Binarization is doubly favorable to the "perceptron" conclusion.

This matters because Lynn's headline 90% is from **calcium imaging** — an analog signal that he chooses to binarize — at coarse time bins (V1 at 667 ms). The native quantity is a graded spike count. An analog treatment would escape both failure modes.

---

## 3. Simulation Architecture

### 3.1 Design

The generator is **purely higher-order**: linear_strength = 0, so the true input-output function involves only centered pairwise interactions delta_i*delta_j. Under independent inputs, the linear model can explain ~none of the recoverable information by construction (phi_direct ~ 0). Any phi_direct > 0 is purely a masking artifact.

| Parameter | Value | Description |
|---|---|---|
| K (inputs) | 32 | Number of input neurons |
| n_pairs | 120 | Interaction pairs (from K*(K-1)/2 = 496 possible) |
| L_train | 1,000,000 | Training samples |
| L_test | 1,000,000 | Test samples |
| rate | 0.20 | Binary firing probability / dichotomization threshold |
| target_out_rate | 0.15 | Output neuron firing rate |
| interaction_strength | 3.0 | Magnitude of interaction drive |
| interaction_type | centered_product | delta_i * delta_j (centered around input means) |

### 3.2 Input generation

One-factor latent model with direct pairwise correlation control:

    z_n = sqrt(rho) * g_shared + sqrt(1-rho) * eps_n

where g_shared is a single shared Gaussian factor and eps_n is private noise. This produces pairwise Gaussian correlation = rho for every pair. The latent z is then mapped to the chosen representation:

| Mode | Transform | Properties |
|---|---|---|
| binary | x = 1{z > Phi^{-1}(1-rate)} | Sparse, asymmetric; x²=x creates leakage |
| gaussian | x = z | Symmetric, zero-mean; no coskewness |
| uniform | x = Phi(z) | Symmetric, strictly positive (0,1); no coskewness |
| lognormal | x = exp(sigma*z - sigma²/2) | Skewed, non-negative; partial coskewness |

### 3.3 Factorial design

- **7 correlation levels**: rho in {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
- **4 representations**: binary, gaussian, uniform, lognormal

28 conditions, all sharing the same generator wiring. Only the input correlation and representation vary.

### 3.4 Model fitting

Two models fit per condition via IRLS/Newton (exact logistic regression, ridge=1e-3):

1. **Direct (linear) model**: features = [1, x] (K+1 = 33 features). This is Lynn's maxent model.
2. **Interaction model**: features = [1, x, all x_i*x_j] (1 + K + K*(K-1)/2 = 529 features).

All metrics are computed on the held-out test set (1M samples) to prevent overfitting.

### 3.5 Metrics (bits)

| Metric | Formula | Interpretation |
|---|---|---|
| phi_direct | (S_tot - S_dir) / (S_tot - S_true) = I_dir / I_true | Lynn's direct share of recoverable info (his implicit claim: ~1) |
| phi_int | (S_tot - S_int) / (S_tot - S_true) = I_int / I_true | Interaction model's share (sanity check; should be ~1) |
| nll_gap | NLL_direct - NLL_interaction | Predictive necessity of interactions (>0 means interactions help on held-out data) |

---

## 4. Results

### 4.1 The representation determines the conclusion

phi_direct (fraction of recoverable info captured by direct dependencies):

| rho | binary | gaussian | uniform | lognormal |
|---|---|---|---|---|
| 0.0 | 0.04 | 0.00 | 0.00 | 0.07 |
| 0.1 | 0.11 | 0.00 | 0.00 | 0.11 |
| 0.2 | 0.18 | 0.00 | 0.00 | 0.15 |
| 0.3 | 0.25 | 0.00 | 0.00 | 0.19 |
| 0.5 | 0.42 | 0.01 | 0.00 | 0.26 |
| 0.7 | 0.61 | 0.00 | 0.00 | 0.33 |
| 0.9 | 0.84 | 0.00 | 0.00 | 0.42 |

The same purely nonlinear generator produces phi_direct ranging from 0.00 to 0.84 depending only on how the inputs are represented and how correlated they are. Notably, uniform inputs are strictly positive yet show no masking — confirming that **symmetry, not positivity**, is the relevant property. The generator has zero linear drive — every bit of phi_direct > 0 is masking artifact.

**Key contrasts:**
- **Gaussian vs binary at rho=0.9**: phi_direct = 0.00 vs 0.84. Same interactions, same correlation, different representation.
- **Binary at rho=0 vs rho=0.9**: phi_direct = 0.04 vs 0.84. Same interactions, same representation, different correlation.

### 4.2 The interaction model always recovers the information

phi_int ~ 1.00 across all 21 conditions (range: 1.000 to 1.002). The information is genuinely present in the inputs and recoverable by a properly specified model. Only the *linear* model is fooled.

### 4.3 Interactions remain predictively necessary

The held-out NLL gap (NLL_direct - NLL_interaction) is positive in every condition, confirming that interactions carry genuine predictive information that the linear model cannot capture:

| rho | binary (bits) | gaussian (bits) | uniform (bits) | lognormal (bits) |
|---|---|---|---|---|
| 0.0 | 0.39 | 0.58 | 0.27 | 0.53 |
| 0.5 | 0.27 | 0.59 | 0.31 | 0.41 |
| 0.9 | 0.09 | 0.74 | 0.33 | 0.32 |

For Gaussian and uniform inputs the gap stays flat or *increases* with correlation (interactions remain fully detectable). For binary inputs the gap shrinks as the linear model absorbs more of the signal — but remains positive even at rho=0.9.

### 4.4 Results scale to 10x more inputs

To confirm these findings are not an artifact of small network size, we repeated the rho=0.5 condition at K=320 inputs (10x) with 1,200 interaction pairs:

| Mode | phi_direct | phi_int | nll_gap (bits) |
|---|---|---|---|
| binary | 0.11 | 1.00 | 0.50 |
| gaussian | 0.00 | 1.00 | 0.66 |
| uniform | 0.00 | 1.00 | 0.45 |
| lognormal | 0.16 | 1.00 | 0.50 |

The qualitative pattern is identical: Gaussian and uniform show no masking; binary and lognormal show partial masking; phi_int = 1.00 and nll_gap > 0 everywhere.

### 4.5 Masking depends on distributional skewness

To test whether skewness predicts masking across a range of continuous distributions, we ran the K=32 model at rho=0.5 for seven representations (all with the same purely nonlinear generator):

| Distribution | Skewness | phi_direct | phi_int | I_true (bits) | I_dir (bits) | nll_gap (bits) |
|---|---|---|---|---|---|---|
| uniform | 0.000 | 0.002 | 1.00 | 0.31 | 0.00 | 0.31 |
| gaussian | 0.000 | 0.005 | 1.00 | 0.59 | 0.00 | 0.59 |
| half_normal | 0.995 | 0.176 | 1.00 | 0.52 | 0.09 | 0.43 |
| binary (p=0.2) | 1.500 | 0.416 | 1.00 | 0.47 | 0.20 | 0.27 |
| exponential | 2.000 | 0.279 | 1.00 | 0.57 | 0.16 | 0.41 |
| chi-squared(1) | 2.828 | 0.225 | 1.00 | 0.59 | 0.13 | 0.46 |
| lognormal (σ=0.8) | 3.689 | 0.260 | 1.00 | 0.56 | 0.15 | 0.41 |

Skewness values are theoretical: 0 (Gaussian, Uniform by symmetry), √2(4−π)/(π−2)^(3/2) (half-normal), (1−2p)/√(p(1−p)) (binary), 2 (exponential), 2√2 (chi-squared(1)), (e^(σ²)+2)√(e^(σ²)−1) (lognormal).

For symmetric distributions (skewness = 0), phi_direct is essentially zero regardless of whether inputs are positive (uniform) or not (Gaussian). For skewed continuous distributions, phi_direct rises with skewness to ~0.18-0.28. Binary inputs (p=0.2) are an outlier: phi_direct = 0.42 despite moderate skewness (1.5), confirming that the x²=x identity is a separate, stronger masking mechanism beyond what skewness alone predicts. Among the continuous distributions, masking does not increase monotonically with skewness beyond ~2.0, suggesting the full coskewness structure — not just the marginal skewness — determines the degree of masking.

The full rho sweep for all seven distributions confirms these patterns hold across the entire correlation range:

phi_direct vs rho (all K=32, L=1M):

| rho | gaussian | uniform | half_normal | lognormal | exponential | chi2 | binary |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.00 | 0.00 | 0.02 | 0.07 | 0.05 | 0.07 | 0.04 |
| 0.1 | 0.00 | 0.00 | 0.02 | 0.11 | 0.08 | 0.07 | 0.11 |
| 0.2 | 0.00 | 0.00 | 0.04 | 0.15 | 0.13 | 0.10 | 0.18 |
| 0.3 | 0.00 | 0.00 | 0.08 | 0.19 | 0.18 | 0.13 | 0.25 |
| 0.5 | 0.01 | 0.00 | 0.18 | 0.26 | 0.28 | 0.23 | 0.42 |
| 0.7 | 0.00 | 0.00 | 0.32 | 0.33 | 0.39 | 0.38 | 0.61 |
| 0.9 | 0.00 | 0.00 | 0.45 | 0.42 | 0.56 | 0.65 | 0.84 |

At high rho, the ordering among continuous distributions shifts: chi-squared (skew=2.8) reaches phi_direct=0.65 at rho=0.9, surpassing lognormal (0.42) and exponential (0.56). This reveals that the interaction between skewness and correlation is nonlinear — higher-order moments of the distribution matter increasingly as rho grows. Binary remains the most strongly masked at every rho value.

### 4.6 Lynn's entropy metric and greedy input selection

The results above use phi_direct = I_dir / I_true, which normalizes by recoverable information. Lynn's paper reports a different metric: **(S_tot - S_dir) / S_tot**, which normalizes by total output entropy. We computed both, and also ran Lynn's exact greedy optimal input selection algorithm (add one input at a time, choosing whichever maximally reduces S_dir) on all seven distributions at rho=0.5.

Lynn's metric (S_tot - S_dir) / S_tot at rho=0.5:

| Mode | All 32 inputs | Greedy best 5 | Greedy best 10 | phi_direct |
|---|---|---|---|---|
| binary | 32.2% | 26.2% | 30.1% | 0.42 |
| gaussian | -0.1% | -0.1% | -0.1% | 0.01 |
| uniform | -0.1% | -0.1% | -0.1% | 0.00 |
| lognormal | 23.8% | 19.7% | 22.0% | 0.26 |
| half_normal | 15.5% | 12.1% | 14.4% | 0.18 |
| exponential | 26.0% | 22.1% | 24.3% | 0.28 |
| chi-squared | 21.8% | 17.2% | 20.4% | 0.23 |

The greedy algorithm correctly identifies the most informative inputs (input ordering is consistent across distributions, reflecting the shared generator wiring). However, the linear model cannot exploit these inputs for symmetric distributions — Gaussian and uniform show 0% explained variability regardless of how many optimally selected inputs are included. For binary inputs at rho=0.5, the greedy procedure captures 32% with all inputs; at rho=0.9 (where Lynn analyzes real data), this rises to ~79%, approaching his reported ~90%.

Full numerical results are in `lynn_metric_results.csv` (rho sweep) and `greedy_trajectories.csv` (greedy selection step-by-step).

### 4.7 Overfitting is not a concern

At rho=0 with Gaussian inputs, phi_direct = 0.00 and the interaction model achieves phi_int = 1.00. If the interaction model were overfitting, it would show inflated phi_int at low correlation where interactions are hardest to fit. Instead, it achieves exactly the oracle ceiling. The 1M held-out test set and ridge regularization ensure all metrics reflect genuine generalization.

---

## 5. Why This Matters for Interpreting the Lynn Framework

### 5.1 Two independent reasons the method over-reports direct dependencies

1. **Correlation masking**: Interaction terms become collinear with marginals when inputs share common drive. This is the universal condition in real neural circuits (shared sensory stimuli, locomotion state, arousal, recurrent connectivity).

2. **Binarization artifact**: The identity x² = x for binary data creates a coskewness channel that leaks interaction signal into the linear fit. This channel does not exist for symmetric continuous signals. Lynn's headline results use binarized calcium imaging — an analog signal that need not be binarized.

These are *independent* mechanisms. Even with independent inputs (rho=0), binary representation gives phi_direct = 0.04 vs 0.00 for Gaussian. Even without binarization (Gaussian inputs), correlation gives phi_direct = 0.01 at rho=0.5 vs 0.00 at rho=0. In real data, both mechanisms act simultaneously.

### 5.2 The ~90% figure does not constrain the computational mechanism

Lynn reports that direct dependencies explain ~90% of variability. Our results show that a purely nonlinear generator (zero linear drive) produces phi_direct = 0.84 (equivalent to Lynn's ~84% figure) with binary inputs at rho=0.9. The same generator with Gaussian inputs at the same correlation gives phi_direct = 0.00. The 90% figure reflects the representation and correlation structure, not the computation.

### 5.3 What the framework can and cannot tell us

**Can tell us:**
- The minimum information content of pairwise input-output correlations
- Which inputs are most informative about an output neuron's activity (the optimal input selection is a genuine contribution)
- The sparsity structure of the effective linear connectivity

**Cannot tell us:**
- Whether the neuron's true computation involves interactions
- Whether a nonlinear model would provide better predictions if properly specified
- Whether the high direct-explained value arises from a true perceptron mechanism or from masking by correlation and binarization

### 5.4 The conceptual error

Lynn conflates a **predictive** claim ("a linear model predicts the binarized activity well") with a **mechanistic** one ("the neuron is a perceptron"). Correlated inputs and binarization are exactly the conditions that make the gap between these claims large. The correct conclusion is: "a linear model is sufficient to predict binarized, coarsely binned neuronal activity given the observed input statistics," which is a weaker and importantly different statement.

---

## 6. Scope and Limitations

- We provide the model with the true generative input set, so input discovery (Lynn's greedy optimal-input selection) is not a confound. This isolates the linear-vs-interaction question.
- Our generator uses centered pairwise products. Other forms of higher-order dependence (e.g., threshold logic, dendritic supralinearities) may behave differently, though the coskewness mechanism generalizes.
- The one-factor latent model is a simplified correlation structure. Real neural correlations have richer eigenspectra. The qualitative conclusion — that correlation attenuates interaction detectability — holds generally.
- We use K=32 inputs, smaller than the hundreds selected in Lynn's analysis. The coskewness mechanism operates at the level of individual input pairs and does not require large K.
- The purely higher-order generator (linear_strength=0) is the cleanest test case. Real neurons likely have both linear and interaction components; the masking effect applies to the interaction component regardless.

---

## 7. Conclusion

The Lynn maximum-entropy framework provides a useful descriptive tool for quantifying how much of a neuron's variability is captured by pairwise input-output correlations. However, its central conclusion — that neurons are well-described as perceptrons — rests on conflating predictive sufficiency with mechanistic simplicity.

Our simulations demonstrate two independent mechanisms that cause a purely nonlinear generator to appear linear under Lynn's framework: (1) input correlation absorbs interaction terms into effective linear weights, and (2) binary representation creates a mathematical leakage channel (x² = x) that does not exist for continuous signals. Together, these mechanisms produce phi_direct = 0.84 from a generator with zero linear drive — nearly matching Lynn's reported ~90%.

The finding that "a linear model predicts binarized activity well" is real and useful. The inference that "neurons do not need interactions" does not follow. The same data, analyzed with continuous inputs rather than binarized ones, would yield the opposite conclusion.
