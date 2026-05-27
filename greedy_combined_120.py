"""
Greedy selection from combined linear + product pool, 120 steps.
Runs for both generators (linear-only and interaction-only) across all 7 distributions.
"""
import torch, math, csv
from masking_sim import (Config, sample_inputs, build_generator, generator_eta,
                         calibrate_bias, bin_entropy_bits, nll_bits, fit_logistic_irls,
                         features_direct, features_interaction)

N_GREEDY = 120
L_TRAIN = 200_000
L_TEST = 200_000
K = 32
RHO = 0.5

def run_greedy(cfg, gpar, gen_seed, generator_type):
    iu = torch.triu_indices(K, K, offset=1, device=cfg.device)
    n_products = iu.shape[1]
    true_pairs = set((gpar['pj'][i].item(), gpar['pk'][i].item())
                     for i in range(gpar['pj'].numel()))

    labels = [f'x_{i}' for i in range(K)]
    for p in range(n_products):
        i, j = iu[0, p].item(), iu[1, p].item()
        is_true = (i, j) in true_pairs
        labels.append(f'x_{i}*x_{j}' + ('*' if is_true else ''))

    all_rows = []
    summary_rows = []

    for mode in cfg.modes:
        gen_run = torch.Generator(device=cfg.device).manual_seed(gen_seed)
        x_tr = sample_inputs(mode, cfg.L_train, cfg.K, cfg, RHO, gen_run)
        mu = x_tr.mean(dim=0)
        eta_tr = generator_eta(x_tr, cfg, gpar, mu)
        bias = calibrate_bias(eta_tr, cfg.target_out_rate)
        y_tr = torch.bernoulli(torch.sigmoid(eta_tr + bias), generator=gen_run)
        x_te = sample_inputs(mode, cfg.L_test, cfg.K, cfg, RHO, gen_run)
        p_true_te = torch.sigmoid(generator_eta(x_te, cfg, gpar, mu) + bias)
        y_te = torch.bernoulli(p_true_te, generator=gen_run)

        S_tot = bin_entropy_bits(y_te.mean()).item()
        S_true = bin_entropy_bits(p_true_te).mean().item()
        I_recov = S_tot - S_true

        with torch.no_grad():
            # All-linear model
            beta_d = fit_logistic_irls(features_direct(x_tr), y_tr, cfg)
            p_d = torch.sigmoid((features_direct(x_te) @ beta_d).clamp(-30, 30))
            S_dir_lin = bin_entropy_bits(p_d).mean().item()

            # All-interaction model (true pairs only)
            pair_idx = torch.stack([gpar['pj'], gpar['pk']])
            beta_i = fit_logistic_irls(features_interaction(x_tr, pair_idx), y_tr, cfg)
            p_i = torch.sigmoid((features_interaction(x_te, pair_idx) @ beta_i).clamp(-30, 30))
            S_dir_int = bin_entropy_bits(p_i).mean().item()

        # Greedy from combined pool
        prods_tr = x_tr[:, iu[0]] * x_tr[:, iu[1]]
        prods_te = x_te[:, iu[0]] * x_te[:, iu[1]]
        all_f_tr = torch.cat([x_tr, prods_tr], dim=1)
        all_f_te = torch.cat([x_te, prods_te], dim=1)
        ones_tr = torch.ones(cfg.L_train, 1, device=x_tr.device, dtype=x_tr.dtype)
        ones_te = torch.ones(cfg.L_test, 1, device=x_te.device, dtype=x_te.dtype)

        selected = []
        remaining = list(range(K + n_products))

        print(f'\n  {generator_type} / {mode}')
        with torch.no_grad():
            for step in range(N_GREEDY):
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

                all_rows.append([generator_type, mode, RHO, step+1, best_feat,
                                labels[best_feat], feat_type, is_true, best_sdir, lynn])

                if step < 5 or step == N_GREEDY - 1 or (step+1) % 20 == 0:
                    tp = ' [TRUE]' if is_true else ''
                    print(f'    n={step+1:3d}  {feat_type:7s}  {labels[best_feat]:14s}  '
                          f'lynn={lynn:.4f}{tp}')

        n_lin = sum(1 for r in all_rows if r[0]==generator_type and r[1]==mode and r[6]=='LINEAR')
        n_prod = sum(1 for r in all_rows if r[0]==generator_type and r[1]==mode and r[6]=='PRODUCT')
        n_true = sum(1 for r in all_rows if r[0]==generator_type and r[1]==mode and r[7]==True)
        S_dir_greedy = all_rows[-1][8]
        lynn_greedy = all_rows[-1][9]

        summary_rows.append([generator_type, mode, S_tot, S_true, I_recov,
                            S_dir_lin, (S_tot-S_dir_lin)/S_tot, S_tot-S_dir_lin,
                            S_dir_int, (S_tot-S_dir_int)/S_tot, S_tot-S_dir_int,
                            S_dir_greedy, lynn_greedy, S_tot-S_dir_greedy,
                            n_lin, n_prod, n_true])

    return all_rows, summary_rows


def main():
    # Shared config
    base_cfg = Config(L_train=L_TRAIN, L_test=L_TEST, K=K, n_pairs=120,
                      hessian_chunk=200_000)

    gen = torch.Generator(device=base_cfg.device).manual_seed(base_cfg.seed)
    gpar = build_generator(base_cfg, gen)
    gen_seed = base_cfg.seed + 1

    all_traj = []
    all_summary = []

    # 1) Interaction-only generator
    cfg_int = Config(L_train=L_TRAIN, L_test=L_TEST, K=K, n_pairs=120,
                     interaction_strength=3.0, linear_strength=0.0,
                     hessian_chunk=200_000)
    print('='*70)
    print('INTERACTION-ONLY GENERATOR')
    print('='*70)
    traj, summ = run_greedy(cfg_int, gpar, gen_seed, 'interaction')
    all_traj.extend(traj)
    all_summary.extend(summ)

    # 2) Linear-only generator
    cfg_lin = Config(L_train=L_TRAIN, L_test=L_TEST, K=K, n_pairs=120,
                     interaction_strength=0.0, linear_strength=3.0,
                     hessian_chunk=200_000)
    print('\n' + '='*70)
    print('LINEAR-ONLY GENERATOR')
    print('='*70)
    traj, summ = run_greedy(cfg_lin, gpar, gen_seed, 'linear')
    all_traj.extend(traj)
    all_summary.extend(summ)

    # Save trajectories
    with open('results/greedy_combined_120steps.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['generator','mode','rho','n','feature_idx','label','feat_type',
                    'is_true_pair','S_dir','lynn_metric'])
        w.writerows(all_traj)
    print('\nsaved results/greedy_combined_120steps.csv')

    # Save summary
    with open('results/greedy_comparison_120steps.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['generator','mode','S_tot','S_true','I_recoverable',
                    'S_dir_linear_all32','lynn_linear_all32','I_linear_all32',
                    'S_dir_interaction_model','lynn_interaction_model','I_interaction_model',
                    'S_dir_greedy_n120','lynn_greedy_n120','I_greedy_n120',
                    'n_linear_selected','n_product_selected','n_true_pairs_selected'])
        w.writerows(all_summary)
    print('saved results/greedy_comparison_120steps.csv')

    # Print summary
    print('\n' + '='*90)
    print(f"{'gen':>11s} {'mode':>14s} | {'I_recov':>7s} | {'lynn_lin':>8s} | {'lynn_int':>8s} | "
          f"{'lynn_g120':>9s} | {'#lin':>4s} | {'#prod':>5s} | {'#true':>5s}")
    print('-'*90)
    for r in all_summary:
        print(f"{r[0]:>11s} {r[1]:>14s} | {r[4]:7.4f} | {r[6]:8.3f} | {r[9]:8.3f} | "
              f"{r[12]:9.4f} | {r[14]:4d} | {r[15]:5d} | {r[16]:5d}")


if __name__ == "__main__":
    main()
