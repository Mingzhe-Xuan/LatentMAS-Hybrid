# PLAN-V2 STATUS: see ../IMPLEMENTATION_STATUS.md#s1----exact-f-fidelity-and-performance.
from .common import *
import time
from .mapping import mapping_rows


def run(states, wo, wi, bias, kernel, args, logger):
    logger.info(
        "S1: exact-F and single-kernel measurements begin (%d states).", len(states)
    )
    rows, single = mapping_rows(states, wo, wi, bias, kernel, args, False, logger)
    logger.info(
        "S1: measurements complete: %d mapping rows, %d single-kernel rows.",
        len(rows),
        len(single),
    )
    return rows, single


def performance(states, wo, wi, bias, kernel, args, logger):
    chosen = []
    for source in SOURCES:
        chosen.extend([state for state in states if state.source == source][:500])
    if not chosen:
        return []
    logger.info(
        "S1 performance: warm-up 200 calls; timing %d balanced states per method.",
        len(chosen),
    )
    for index in range(200):
        exact(chosen[index % len(chosen)].vector, wo, wi, bias, args.tau)
        kernel_map(chosen[index % len(chosen)].vector, kernel)
    if wo.is_cuda:
        torch.cuda.synchronize()
    rows = []
    for method, fn in (
        ("exact", lambda q: exact(q, wo, wi, bias, args.tau)),
        ("kernel", lambda q: kernel_map(q, kernel)),
    ):
        for index, state in enumerate(chosen, start=1):
            if wo.is_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            fn(state.vector)
            if wo.is_cuda:
                torch.cuda.synchronize()
            rows.append(
                {
                    **base(state),
                    "method": method,
                    "latency_us": (time.perf_counter_ns() - started) / 1000,
                }
            )
            if index == len(chosen):
                logger.info("S1 performance: %s timing complete.", method)
    return rows


def plot(rows):
    figure_dir = RESULT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        ("entropy", "Exact entropy"),
        ("confidence", "Exact confidence"),
        ("prompt_length", "Prompt length"),
        ("reply_length", "Reply length"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (field, label) in zip(axes.flat, fields):
        for source, marker in (("prompt", "o"), ("reply", "x")):
            q = [
                r
                for r in rows
                if r["source"] == source and np.isfinite(r.get(field, np.nan))
            ]
            if q:
                ax.scatter(
                    [r[field] for r in q],
                    [r["f_rel_l2"] for r in q],
                    s=8,
                    alpha=0.35,
                    marker=marker,
                    label=source,
                )
        ax.set_xlabel(label)
        ax.set_ylabel("relative L2 error")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "s1_error_conditioning.pdf")
    plt.close(fig)
