# PLAN-V2 STATUS: see ../IMPLEMENTATION_STATUS.md#s2----softmax-propagation-and-ablation.
from .common import *
from .mapping import mapping_rows

def calibration(states, wo, wi, bias, args, logger=None):
    rows = []
    configuration = 0
    for kind in ("orf", "iid"):
        for m in (256, 512, 1024, 2048, 4096):
            for tau in (0.7, 1.0, 1.3):
                for seed in (101, 202, 303, 404, 505):
                    configuration += 1
                    if logger: logger.info("S2 calibration progress: configuration %d/150 (%s, m=%d, tau=%s, seed=%d).", configuration, kind, m, tau, seed)
                    k = (
                        build_kernel_state(
                            wo,
                            wi,
                            bias,
                            feature_count=m,
                            temperature=tau,
                            seed=seed,
                            chunk_size=args.kernel_chunk_size,
                        )
                        if kind == "orf"
                        else iid_kernel(
                            wo, wi, bias, m, tau, seed, args.kernel_chunk_size
                        )
                    )
                    for s in states:
                        f, _ = exact(s.vector, wo, wi, bias, tau)
                        h, ok = kernel_map(s.vector, k)
                        rows.append(
                            {
                                **base(s),
                                "feature_family": kind,
                                "m": m,
                                "tau": tau,
                                "seed": seed,
                                "rel_l2": float(
                                    (h - f).norm() / f.norm().clamp_min(1e-8)
                                ),
                                "denom_valid": ok,
                            }
                        )
    return rows
def run(states, wo, wi, bias, kernel, args, logger):
    logger.info("S2: softmax propagation measurements begin (%d states).", len(states))
    rows, single = mapping_rows(states, wo, wi, bias, kernel, args, True, logger)
    logger.info("S2: softmax propagation measurements complete.")
    return rows, single

def plot(rows):
    figure_dir = RESULT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for source, color in (("prompt", "tab:blue"), ("reply", "tab:orange")):
        q=[r for r in rows if r["source"] == source]
        axes[0].scatter([r["l1"] for r in q], [r["f_rel_l2"] for r in q], s=8, alpha=.4, label=source, color=color)
        axes[1].scatter([r["tv"] for r in q], [r["f_rel_l2"] for r in q], s=8, alpha=.4, label=source, color=color)
        axes[2].scatter([r["entropy"] for r in q], [r["kl_p_phat"] for r in q], s=8, alpha=.4, label=source, color=color)
    axes[0].set(xlabel="||p-p_hat||_1", ylabel="relative L2(F)", title="Softmax to embedding")
    axes[1].set(xlabel="TV(p,p_hat)", ylabel="relative L2(F)", title="TV propagation")
    axes[2].set(xlabel="exact entropy", ylabel="KL(p||p_hat)", title="Difficulty conditioning")
    [ax.legend() for ax in axes]; fig.tight_layout(); fig.savefig(figure_dir / "s2_error_propagation.pdf"); plt.close(fig)