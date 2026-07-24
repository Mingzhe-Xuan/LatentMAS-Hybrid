from .common import *


def _summary(values):
    values = np.asarray(values)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def run(states, wo, wi, args, logger):
    del args
    logger.info("S0: complete-trajectory hidden norms (%d states).", len(states))
    rows = []
    for state in states:
        rows.append({**base(state), "hidden_norm": float(state.vector.norm())})
    plot(rows, wo, wi)
    return rows


def weight_norm_summary(wo, wi):
    return [
        {
            "embedding": "refiner_output_embedding",
            "mapping": "refiner_to_judger",
            **_summary(wo.norm(dim=1).float().cpu().numpy()),
        },
        {
            "embedding": "judger_input_embedding",
            "mapping": "refiner_to_judger",
            **_summary(wi.norm(dim=1).float().cpu().numpy()),
        },
    ]


def hidden_norm_summary(rows):
    summaries = []
    groups = sorted({(row["role"], row["state_kind"]) for row in rows})
    for role, state_kind in groups:
        norms = [
            row["hidden_norm"]
            for row in rows
            if row["role"] == role and row["state_kind"] == state_kind
        ]
        summaries.append(
            {"role": role, "state_kind": state_kind, **_summary(norms)}
        )
    return summaries


def plot(rows, wo, wi):

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(
        wo.norm(dim=1).detach().cpu().numpy(),
        bins=60,
        histtype="step",
        label="Refiner W_out",
    )
    axis.hist(
        wi.norm(dim=1).detach().cpu().numpy(),
        bins=60,
        histtype="step",
        label="Judger W_in",
    )
    axis.set(xlabel="row L2 norm", ylabel="token count")
    axis.legend()
    figure.tight_layout()
    save_figure(figure, "s0_embedding_norm_hist")
    plt.close(figure)

    groups = sorted({(row["role"], row["state_kind"]) for row in rows})
    columns = 3
    row_count = max(1, (len(groups) + columns - 1) // columns)
    figure, axes = plt.subplots(
        row_count, columns, figsize=(5 * columns, 3.2 * row_count), squeeze=False
    )
    for axis, (role, state_kind) in zip(axes.flat, groups):
        norms = [
            row["hidden_norm"]
            for row in rows
            if row["role"] == role and row["state_kind"] == state_kind
        ]
        axis.hist(norms, bins=min(30, max(5, len(norms))), color="tab:blue", alpha=0.8)
        axis.set_title(f"{role} × {state_kind}", fontsize=9)
        axis.set_xlabel("hidden L2 norm")
        axis.set_ylabel("state count")
    for axis in list(axes.flat)[len(groups) :]:
        axis.set_visible(False)
    figure.tight_layout()
    save_figure(figure, "s0_hidden_norm_hist")
    plt.close(figure)
