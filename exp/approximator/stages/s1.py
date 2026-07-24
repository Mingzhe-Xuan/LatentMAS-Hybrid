from .common import *
import time


def performance(states, wo, wi, bias, kernel, args, logger):
    chosen = states[:500]
    if not chosen:
        return []
    for index in range(200):
        exact(
            chosen[index % len(chosen)].vector,
            wo,
            wi,
            bias,
            args.kernel_temperature,
        )
        kernel_map(chosen[index % len(chosen)].vector, kernel)
    if wo.is_cuda:
        torch.cuda.synchronize()
    rows = []
    for method, function in (
        (
            "exact",
            lambda query: exact(query, wo, wi, bias, args.kernel_temperature),
        ),
        ("kernel", lambda query: kernel_map(query, kernel)),
    ):
        for state in chosen:
            if wo.is_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            function(state.vector)
            if wo.is_cuda:
                torch.cuda.synchronize()
            rows.append(
                {
                    **base(state),
                    "method": method,
                    "latency_us": (time.perf_counter_ns() - started) / 1000,
                }
            )
    logger.info("S1 performance timing complete.")
    return rows


def plot(rows):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(
        [row["entropy"] for row in rows],
        [row["f_rel_l2"] for row in rows],
        s=8,
        alpha=0.4,
    )
    axes[1].scatter(
        [row["confidence"] for row in rows],
        [row["f_rel_l2"] for row in rows],
        s=8,
        alpha=0.4,
    )
    axes[0].set(xlabel="exact entropy", ylabel="relative L2 error")
    axes[1].set(xlabel="exact confidence", ylabel="relative L2 error")
    figure.tight_layout()
    save_figure(figure, "s1_error_conditioning")
    plt.close(figure)
