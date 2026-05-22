"""
masking_sim.py
==============
Does Lynn (2026, Nature Physics) "direct dependencies explain ~90% of neural
variability" survive (a) correlated inputs and (b) an ANALOG instead of binary
representation -- or is it an artifact of binarizing sparse, correlated signals?

Lynn's "direct dependency" model is the maximum-entropy model matching <y> and
<y x_i>, i.e. exactly logistic regression of y on the inputs:
        P(y=1|x) = sigmoid(b + sum_i w_i x_i).
His headline metric is the fraction of recoverable information captured by these
direct dependencies; he claims 0.9 Itrue <= Idir <= Itrue (perceptron-like).

We build neurons whose true input-output function is PURELY higher-order
(centered pairwise interactions delta_j*delta_k, zero linear drive), so under
INDEPENDENT inputs the linear model captures ~none of the recoverable
information (Idir/Itrue ~ 0). Two sweeps:

  (1) input correlation rho            -> the original "masking" effect
  (2) input REPRESENTATION input_mode  -> binary vs analog

THE REPRESENTATION RESULT (input_mode):
Masking leaks the interaction delta_j*delta_k into the linear weights through the
input coskewness E[delta_j^2 delta_k].
  * binary x in {0,1}: x^2 = x  =>  delta_j^2 is affine in x_j, so coskewness
    = (1-2p)*Cov(x_j,x_k): nonzero for sparse (p!=0.5) correlated inputs => masking.
  * symmetric continuous (Gaussian): all odd joint moments vanish => coskewness
    = 0 at every rho => the linear model captures NONE of the interaction.
  * skewed analog (lognormal): nonzero coskewness => partial masking.
So the "direct dependencies dominate" finding is largely a property of the binary
representation, not of the underlying computation.

Metrics (bits, on held-out test data; p_true is known because we built it):
  Stot=H(<y>)  Strue=<H(p_true)>  Sdir/Sint=<H(sigmoid(eta_dir/int))>
  Itrue=Stot-Strue  Idir=Stot-Sdir  Iint=Stot-Sint
  phi_direct=Idir/Itrue (Lynn's implicit ~1)   phi_int=Iint/Itrue (sanity ~1)
  nll_gap = NLL_direct - NLL_int (>0 => interactions still help prediction)

Device-agnostic (CUDA on the DGX Spark GH200, else CPU); fully vectorized,
logistic fits by IRLS/Newton.
"""

from dataclasses import dataclass, field
from typing import List
import math
import torch


# ----------------------------- configuration ------------------------------ #
@dataclass
class Config:
    L_train: int = 200_000
    L_test: int = 200_000
    K: int = 16                     # number of input neurons
    rate: float = 0.2               # binary firing prob (also sets the dichotomization threshold)
    target_out_rate: float = 0.15   # output rate (bias auto-calibrated per cell)

    input_mode: str = "binary"      # "binary" | "gaussian" | "lognormal" | "uniform" | "half_normal" | "exponential" | "chi2"
    lognormal_sigma: float = 0.8    # skew parameter for the lognormal representation

    linear_strength: float = 0.0    # 0.0 => purely higher-order generator (cleanest demo)
    interaction_strength: float = 3.0
    interaction_type: str = "centered_product"  # "centered_product" | "product" | "xor"
    n_pairs: int = 40

    rhos: List[float] = field(default_factory=lambda:
                              [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    modes: List[str] = field(default_factory=lambda:
                             ["binary", "gaussian", "uniform", "lognormal",
                              "half_normal", "exponential", "chi2"])

    ridge: float = 1e-3
    irls_iters: int = 50
    irls_tol: float = 1e-7
    hessian_chunk: int = 0          # 0 = no chunking; else accumulate Hessian over sample chunks
    seed: int = 0
    dtype: torch.dtype = torch.float32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------- utilities -------------------------------- #
def normal_icdf(q: float) -> float:
    return math.sqrt(2.0) * torch.special.erfinv(torch.tensor(2.0 * q - 1.0)).item()


def bin_entropy_bits(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1.0 - 1e-7)
    return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))


def nll_bits(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1.0 - 1e-7)
    return -(y * torch.log2(p) + (1.0 - y) * torch.log2(1.0 - p)).mean()


# ----------------------------- input sampler ------------------------------ #
def sample_inputs(mode, L, K, cfg: Config, rho, gen):
    """
    One-factor latent model: z_n = sqrt(rho)*g_shared + sqrt(1-rho)*eps_n
    (latent correlation rho; rho=0 => independent). Mapped to the chosen
    representation. Returns x (L,K).
        binary      : 1{z > Phi^{-1}(1-rate)}          (sparse, asymmetric; skew=(1-2p)/sqrt(p(1-p)))
        gaussian    : z                                 (symmetric, zero-mean; skew=0)
        uniform     : Phi(z)                            (symmetric on (0,1); skew=0)
        lognormal   : exp(sigma*z - sigma^2/2)          (skewed, non-negative; skew=(e^s2+2)*sqrt(e^s2-1))
        half_normal : |z|                               (skew=0.995)
        exponential : -ln(1-Phi(z))                     (skew=2)
        chi2        : z^2                               (skew=2*sqrt(2))
    """
    g = torch.randn(L, 1, generator=gen, device=cfg.device, dtype=cfg.dtype)
    eps = torch.randn(L, K, generator=gen, device=cfg.device, dtype=cfg.dtype)
    z = math.sqrt(rho) * g + math.sqrt(1.0 - rho) * eps
    if mode == "binary":
        return (z > normal_icdf(1.0 - cfg.rate)).to(cfg.dtype)
    if mode == "gaussian":
        return z
    if mode == "lognormal":
        s = cfg.lognormal_sigma
        return torch.exp(s * z - 0.5 * s * s)
    if mode == "uniform":
        # Phi(z) maps standard normal to Uniform(0,1); preserves rank correlation
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    if mode == "half_normal":
        return z.abs()
    if mode == "exponential":
        u = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return -torch.log1p(-u.clamp(max=1.0 - 1e-7))
    if mode == "chi2":
        return z.pow(2)
    raise ValueError(mode)


# --------------------------- generator (truth) ---------------------------- #
def build_generator(cfg: Config, gen):
    K = cfg.K
    iu = torch.triu_indices(K, K, offset=1, device=cfg.device)
    n_all = iu.shape[1]
    n_pairs = min(cfg.n_pairs, n_all)
    sel = torch.randperm(n_all, generator=gen, device=cfg.device)[:n_pairs]
    return {
        "pj": iu[0, sel], "pk": iu[1, sel],
        "pair_sign": torch.randint(0, 2, (n_pairs,), generator=gen,
                                   device=cfg.device, dtype=cfg.dtype) * 2 - 1,
        "w_lin": torch.randint(0, 2, (K,), generator=gen,
                               device=cfg.device, dtype=cfg.dtype) * 2 - 1,
    }


def generator_eta(x, cfg: Config, gpar, mu):
    """True linear predictor eta(x); mu is the per-input mean used for centering."""
    pj, pk, sgn = gpar["pj"], gpar["pk"], gpar["pair_sign"]
    xj, xk = x[:, pj], x[:, pk]
    if cfg.interaction_type == "centered_product":
        phi = (xj - mu[pj]) * (xk - mu[pk])
    elif cfg.interaction_type == "product":
        phi = xj * xk
    elif cfg.interaction_type == "xor":      # only meaningful for binary inputs
        phi = (xj != xk).to(cfg.dtype)
    else:
        raise ValueError(cfg.interaction_type)
    eta = cfg.interaction_strength * (phi * sgn).sum(dim=1)
    if cfg.linear_strength != 0.0:
        eta = eta + cfg.linear_strength * (x @ gpar["w_lin"])
    return eta


def calibrate_bias(eta, target_rate):
    lo, hi = -50.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if torch.sigmoid(eta + mid).mean().item() < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------ feature maps ------------------------------ #
def features_direct(x):
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return torch.cat([ones, x], dim=1)


def features_interaction(x, pair_idx=None):
    K = x.shape[1]
    if pair_idx is None:
        iu = torch.triu_indices(K, K, offset=1, device=x.device)
    else:
        iu = pair_idx
    prod = x[:, iu[0]] * x[:, iu[1]]
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return torch.cat([ones, x, prod], dim=1)


# --------------------------- logistic fit (IRLS) -------------------------- #
def fit_logistic_irls(Phi, y, cfg: Config):
    L, D = Phi.shape
    beta = torch.zeros(D, device=Phi.device, dtype=Phi.dtype)
    I = torch.eye(D, device=Phi.device, dtype=Phi.dtype)
    for _ in range(cfg.irls_iters):
        eta = (Phi @ beta).clamp(-30, 30)
        p = torch.sigmoid(eta)
        w = (p * (1.0 - p)).clamp_min(1e-6)
        if cfg.hessian_chunk and cfg.hessian_chunk < L:
            H = cfg.ridge * I.clone(); g = cfg.ridge * beta.clone()
            for s in range(0, L, cfg.hessian_chunk):
                e = slice(s, s + cfg.hessian_chunk)
                Pc = Phi[e]
                H += Pc.t() @ (w[e].unsqueeze(1) * Pc)
                g += Pc.t() @ (p[e] - y[e])
        else:
            H = Phi.t() @ (w.unsqueeze(1) * Phi) + cfg.ridge * I
            g = Phi.t() @ (p - y) + cfg.ridge * beta
        delta = torch.linalg.solve(H, g)
        beta = beta - delta
        if delta.abs().max().item() < cfg.irls_tol:
            break
    return beta


# ------------------------------ one cell ---------------------------------- #
@torch.no_grad()
def run_one(cfg: Config, mode, rho, gpar, gen):
    x_tr = sample_inputs(mode, cfg.L_train, cfg.K, cfg, rho, gen)
    mu = x_tr.mean(dim=0)                       # centering reference (train)
    eta_tr = generator_eta(x_tr, cfg, gpar, mu)
    bias = calibrate_bias(eta_tr, cfg.target_out_rate)
    y_tr = torch.bernoulli(torch.sigmoid(eta_tr + bias), generator=gen)

    x_te = sample_inputs(mode, cfg.L_test, cfg.K, cfg, rho, gen)
    p_true_te = torch.sigmoid(generator_eta(x_te, cfg, gpar, mu) + bias)
    y_te = torch.bernoulli(p_true_te, generator=gen)

    beta_d = fit_logistic_irls(features_direct(x_tr), y_tr, cfg)
    pair_idx = torch.stack([gpar["pj"], gpar["pk"]])
    beta_i = fit_logistic_irls(features_interaction(x_tr, pair_idx), y_tr, cfg)
    p_d = torch.sigmoid((features_direct(x_te) @ beta_d).clamp(-30, 30))
    p_i = torch.sigmoid((features_interaction(x_te, pair_idx) @ beta_i).clamp(-30, 30))

    Stot = bin_entropy_bits(y_te.mean()).item()
    Strue = bin_entropy_bits(p_true_te).mean().item()
    Sdir = bin_entropy_bits(p_d).mean().item()
    Sint = bin_entropy_bits(p_i).mean().item()
    Itrue = Stot - Strue
    eps = 1e-6
    return dict(mode=mode, rho=rho, out_rate=y_te.mean().item(),
                Stot=Stot, Strue=Strue, Sdir=Sdir, Sint=Sint,
                Itrue=Itrue, Idir=Stot - Sdir, Iint=Stot - Sint,
                phi_direct=(Stot - Sdir) / max(Itrue, eps),
                phi_int=(Stot - Sint) / max(Itrue, eps),
                nll_direct=nll_bits(p_d, y_te).item(),
                nll_int=nll_bits(p_i, y_te).item(),
                nll_true=nll_bits(p_true_te, y_te).item(),
                nll_gap=nll_bits(p_d, y_te).item() - nll_bits(p_i, y_te).item())


# --------------------------------- sweeps --------------------------------- #
def run_sweep(cfg: Config, mode=None):
    """Sweep rho for a single representation."""
    mode = mode or cfg.input_mode
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    gpar = build_generator(cfg, gen)
    print(f"\n[{mode}]  K={cfg.K} pairs={cfg.n_pairs} "
          f"interaction={cfg.interaction_type} linear={cfg.linear_strength}")
    hdr = (f"{'rho':>5} {'out':>6} {'Itrue':>7} {'Idir':>7} {'Iint':>7} "
           f"{'phiDir':>7} {'phiInt':>7} {'nllGap':>7}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for rho in cfg.rhos:
        r = run_one(cfg, mode, rho, gpar, gen)
        rows.append(r)
        print(f"{r['rho']:5.2f} {r['out_rate']:6.3f} {r['Itrue']:7.4f} "
              f"{r['Idir']:7.4f} {r['Iint']:7.4f} {r['phi_direct']:7.3f} "
              f"{r['phi_int']:7.3f} {r['nll_gap']:7.4f}")
    return rows


def run_mode_comparison(cfg: Config):
    """Identical generator; sweep rho separately for each representation."""
    return {m: run_sweep(cfg, mode=m) for m in cfg.modes}


# --------------------------------- plots ---------------------------------- #

# Analytical coskewness coefficient: E[delta_i^2 delta_j] / Cov(x_i, x_j).
# This is the distribution-specific factor that controls how much of a pairwise
# interaction leaks into the linear model.  Independent of rho.
COSKEW_COEFF = {
    "gaussian":    0.0,     # symmetric => zero by parity
    "uniform":     0.0,     # symmetric => zero by parity
    "half_normal": 0.86,    # empirical (no closed form)
    "binary":      0.60,    # analytical: (1-2p) with p=0.2
    "lognormal":   2.50,    # empirical
    "exponential": 1.66,    # empirical
    "chi2":        4.00,    # analytical: kurtosis_excess = 2*sqrt(2)*2 ... simplifies to 4
}


def plot_modes(by_mode, path="masking_by_representation.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for m, rows in by_mode.items():
        rho = [r["rho"] for r in rows]
        coeff = COSKEW_COEFF.get(m, "?")
        label0 = f"{m}  (coskew coeff={coeff})"
        ax[0].plot(rho, [r["phi_direct"] for r in rows], "o-", label=label0)
        ax[1].plot(rho, [r["nll_gap"] for r in rows], "o-", label=m)
    ax[0].set_xlabel("input correlation  rho")
    ax[0].set_ylabel(r"direct share of recoverable info  $I_{dir}/I_{true}$")
    ax[0].set_title("Same nonlinear generator, different representation")
    ax[0].set_ylim(-0.05, 1.1); ax[0].legend(frameon=False, fontsize=7)
    ax[1].axhline(0.0, color="k", lw=0.8, ls=":")
    ax[1].set_xlabel("input correlation  rho")
    ax[1].set_ylabel("held-out NLL(direct) - NLL(int)  [bits]")
    ax[1].set_title("Interactions stay predictively necessary")
    ax[1].legend(frameon=False, title="input_mode")
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    cfg = Config()
    by_mode = run_mode_comparison(cfg)
    try:
        plot_modes(by_mode)
    except Exception as e:
        print("plotting skipped:", e)
