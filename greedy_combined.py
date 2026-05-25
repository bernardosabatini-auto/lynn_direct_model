"""
Two analyses at K=32, rho=0.5:
  Part 1: Fit full model (linear + all products), examine which features matter.
  Part 2: Greedy selection from combined linear + product candidate pool.
"""
import torch, math, csv
from masking_sim import (Config, sample_inputs, build_generator, generator_eta,
                         calibrate_bias, bin_entropy_bits, nll_bits, fit_logistic_irls,
                         features_direct, features_interaction)


def run_analysis(L_train=200_000, L_test=200_000, n_greedy=30):
    cfg = Config(L_train=L_train, L_test=L_test, K=32, n_pairs=120,
                 interaction_strength=3.0, hessian_chunk=200_000)
    rho = 0.5
    K = cfg.K
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    gpar = build_generator(cfg, gen)

    iu = torch.triu_indices(K, K, offset=1, device=cfg.device)
    n_products = iu.shape[1]  # 496
    true_pairs = set((gpar['pj'][i].item(), gpar['pk'][i].item())
                     for i in range(gpar['pj'].numel()))

    print(f"K={K}, n_products={n_products}, total candidates={K + n_products}")
    print(f"True interaction pairs: {len(true_pairs)}")

    # Feature labels
    labels = [f'x_{i}' for i in range(K)]
    for p in range(n_products):
        i, j = iu[0, p].item(), iu[1, p].item()
        is_true = (i, j) in true_pairs
        labels.append(f'x_{i}*x_{j}' + ('*' if is_true else ''))

    greedy_rows = []

    for mode in cfg.modes:
        gen_run = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 1)
        x_tr = sample_inputs(mode, cfg.L_train, cfg.K, cfg, rho, gen_run)
        mu = x_tr.mean(dim=0)
        eta_tr = generator_eta(x_tr, cfg, gpar, mu)
        bias = calibrate_bias(eta_tr, cfg.target_out_rate)
        y_tr = torch.bernoulli(torch.sigmoid(eta_tr + bias), generator=gen_run)
        x_te = sample_inputs(mode, cfg.L_test, cfg.K, cfg, rho, gen_run)
        p_true_te = torch.sigmoid(generator_eta(x_te, cfg, gpar, mu) + bias)
        y_te = torch.bernoulli(p_true_te, generator=gen_run)

        S_tot = bin_entropy_bits(y_te.mean()).item()
        S_true = bin_entropy_bits(p_true_te).mean().item()

        with torch.no_grad():
            # --- PART 1: Full model fit ---
            beta_d = fit_logistic_irls(features_direct(x_tr), y_tr, cfg)
            p_d = torch.sigmoid((features_direct(x_te) @ beta_d).clamp(-30, 30))

            Phi_tr_all = features_interaction(x_tr)
            beta_i = fit_logistic_irls(Phi_tr_all, y_tr, cfg)
            Phi_te_all = features_interaction(x_te)
            p_i = torch.sigmoid((Phi_te_all @ beta_i).clamp(-30, 30))

            S_dir = bin_entropy_bits(p_d).mean().item()
            S_int = bin_entropy_bits(p_i).mean().item()
            nll_d = nll_bits(p_d, y_te).item()
            nll_i = nll_bits(p_i, y_te).item()

            product_weights = beta_i[K+1:].abs()
            linear_weights = beta_i[1:K+1].abs()

            top20_idx = product_weights.topk(20).indices
            n_true_top10 = sum(1 for idx in product_weights.topk(10).indices
                              if (iu[0, idx].item(), iu[1, idx].item()) in true_pairs)
            n_true_top20 = sum(1 for idx in top20_idx
                              if (iu[0, idx].item(), iu[1, idx].item()) in true_pairs)

        I_true = S_tot - S_true
        lynn_d = (S_tot - S_dir) / S_tot
        lynn_i = (S_tot - S_int) / S_tot

        print(f"\n{'='*70}")
        print(f"  {mode}")
        print(f"  PART 1 - Full model:")
        print(f"    lynn_direct={lynn_d:.3f}  lynn_interaction={lynn_i:.3f}  "
              f"nll_gap={nll_d-nll_i:.4f}")
        print(f"    mean|w_linear|={linear_weights.mean():.4f}  "
              f"mean|w_product|={product_weights.mean():.4f}")
        print(f"    true pairs in top-10 products: {n_true_top10}/10  "
              f"top-20: {n_true_top20}/20")

        # --- PART 2: Greedy from combined pool ---
        print(f"  PART 2 - Greedy selection ({n_greedy} steps from {K+n_products} candidates):")

        prods_tr = x_tr[:, iu[0]] * x_tr[:, iu[1]]
        prods_te = x_te[:, iu[0]] * x_te[:, iu[1]]
        all_f_tr = torch.cat([x_tr, prods_tr], dim=1)
        all_f_te = torch.cat([x_te, prods_te], dim=1)

        selected = []
        remaining = list(range(K + n_products))
        ones_tr = torch.ones(cfg.L_train, 1, device=x_tr.device, dtype=x_tr.dtype)
        ones_te = torch.ones(cfg.L_test, 1, device=x_te.device, dtype=x_te.dtype)

        with torch.no_grad():
            for step in range(n_greedy):
                best_sdir = float('inf')
                best_feat = -1
                for c in remaining:
                    trial = selected + [c]
                    Phi = torch.cat([ones_tr, all_f_tr[:, trial]], dim=1)
                    beta = fit_logistic_irls(Phi, y_tr, cfg)
                    Phi_t = torch.cat([ones_te, all_f_te[:, trial]], dim=1)
                    p_pred = torch.sigmoid((Phi_t @ beta).clamp(-30, 30))
                    sdir = bin_entropy_bits(p_pred).mean().item()
                    if sdir < best_sdir:
                        best_sdir = sdir
                        best_feat = c

                selected.append(best_feat)
                remaining.remove(best_feat)
                lynn = (S_tot - best_sdir) / S_tot
                is_product = best_feat >= K
                feat_type = 'PRODUCT' if is_product else 'LINEAR'
                is_true = False
                if is_product:
                    pidx = best_feat - K
                    is_true = (iu[0, pidx].item(), iu[1, pidx].item()) in true_pairs

                greedy_rows.append([mode, rho, step+1, best_feat, labels[best_feat],
                                   feat_type, is_true, best_sdir, lynn])

                if step < 10 or step == n_greedy - 1:
                    tp = ' [TRUE]' if is_true else ''
                    print(f"    n={step+1:3d}  {feat_type:7s}  {labels[best_feat]:14s}  "
                          f"lynn={lynn:.4f}{tp}")

    # Save CSV
    with open('greedy_combined_K32.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'rho', 'n', 'feature_idx', 'label', 'feat_type',
                    'is_true_pair', 'S_dir', 'lynn_metric'])
        w.writerows(greedy_rows)
    print('\nsaved greedy_combined_K32.csv')


if __name__ == "__main__":
    run_analysis()
