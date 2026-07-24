from .common import *


def calibration(states, wo, wi, bias, args, logger=None):
    rows = []
    for family in ("orf", "iid"):
        for feature_count in (256, 512, 1024, 2048, 4096):
            for temperature in (0.7, 1.0, 1.3):
                for seed in (101, 202, 303, 404, 505):
                    if logger:
                        logger.info(
                            "S2 calibration %s m=%d tau=%s seed=%d",
                            family,
                            feature_count,
                            temperature,
                            seed,
                        )
                    kernel = (
                        build_kernel_state(
                            wo,
                            wi,
                            bias,
                            feature_count=feature_count,
                            temperature=temperature,
                            seed=seed,
                            chunk_size=args.kernel_chunk_size,
                        )
                        if family == "orf"
                        else iid_kernel(
                            wo,
                            wi,
                            bias,
                            feature_count,
                            temperature,
                            seed,
                            args.kernel_chunk_size,
                        )
                    )
                    for state in states:
                        exact_value, _ = exact(
                            state.vector, wo, wi, bias, temperature
                        )
                        approximate, valid = kernel_map(state.vector, kernel)
                        rows.append(
                            {
                                **base(state),
                                "feature_family": family,
                                "m": feature_count,
                                "tau": temperature,
                                "seed": seed,
                                "rel_l2": float(
                                    (approximate - exact_value).norm()
                                    / exact_value.norm().clamp_min(1e-8)
                                ),
                                "denom_valid": valid,
                            }
                        )
    return rows


def plot(rows):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(
        [row["l1"] for row in rows],
        [row["f_rel_l2"] for row in rows],
        s=8,
        alpha=0.4,
    )
    axes[1].scatter(
        [row["tv"] for row in rows],
        [row["f_rel_l2"] for row in rows],
        s=8,
        alpha=0.4,
    )
    axes[2].scatter(
        [row["entropy"] for row in rows],
        [row["kl_p_phat"] for row in rows],
        s=8,
        alpha=0.4,
    )
    axes[0].set(xlabel="||p-p_hat||_1", ylabel="relative L2(F)")
    axes[1].set(xlabel="TV(p,p_hat)", ylabel="relative L2(F)")
    axes[2].set(xlabel="exact entropy", ylabel="KL(p||p_hat)")
    figure.tight_layout()
    save_figure(figure, "s2_error_propagation")
    plt.close(figure)
