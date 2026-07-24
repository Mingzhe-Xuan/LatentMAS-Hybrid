from .common import *


def s0(states, wo, wi, args):
    rows = []
    for s in states:
        r = base(s)
        r.update(hidden_norm=float(s.vector.norm()))
        rows.append(r)
    histogram_ecdf(rows, "hidden_norm", "s0")
    return rows


def weight_norm_summary(wo, wi):
    """Summarize embedding norms without retaining a row for every token."""
    summaries = []
    for embedding_name, embedding_matrix in (
        ("source_output_embedding", wo),
        ("target_input_embedding", wi),
    ):
        norms = embedding_matrix.norm(dim=1).float().cpu().numpy()
        summaries.append(
            {
                "embedding": embedding_name,
                "count": len(norms),
                "mean": float(norms.mean()),
                "std": float(norms.std()),
                "min": float(norms.min()),
                "p01": float(np.quantile(norms, 0.01)),
                "p05": float(np.quantile(norms, 0.05)),
                "p50": float(np.quantile(norms, 0.50)),
                "p95": float(np.quantile(norms, 0.95)),
                "p99": float(np.quantile(norms, 0.99)),
                "max": float(norms.max()),
            }
        )
    return summaries


def hidden_norm_summary(rows):
    """Describe the sampled hidden-state norm distribution by source."""
    summaries = []
    for source in SOURCES:
        norms = np.array(
            [row["hidden_norm"] for row in rows if row["source"] == source]
        )
        if not len(norms):
            continue
        summaries.append(
            {
                "source": source,
                "count": len(norms),
                "mean": float(norms.mean()),
                "std": float(norms.std()),
                "min": float(norms.min()),
                "p01": float(np.quantile(norms, 0.01)),
                "p05": float(np.quantile(norms, 0.05)),
                "p25": float(np.quantile(norms, 0.25)),
                "p50": float(np.quantile(norms, 0.50)),
                "p75": float(np.quantile(norms, 0.75)),
                "p95": float(np.quantile(norms, 0.95)),
                "p99": float(np.quantile(norms, 0.99)),
                "max": float(norms.max()),
            }
        )
    return summaries


def run(states, wo, wi, args, logger):
    logger.info("S0: norm diagnostics start for %d sampled states.", len(states))
    rows = s0(states, wo, wi, args)
    plot(rows, wo, wi)
    logger.info("S0: norm diagnostics completed with %d state rows.", len(rows))
    return rows


def plot(rows, wo, wi):
    figure_dir = RESULT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        wo.norm(dim=1).detach().cpu().numpy(),
        bins=60,
        histtype="step",
        label="W_out (source output embedding)",
    )
    ax.hist(
        wi.norm(dim=1).detach().cpu().numpy(),
        bins=60,
        histtype="step",
        label="W_in (target input embedding)",
    )
    ax.set_title("Embedding L2-norm distributions")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Token count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "s0_embedding_norm_hist.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for source, label in (
        ("prompt", "Prompt hidden states"),
        ("reply", "Reply hidden states"),
    ):
        norms = [row["hidden_norm"] for row in rows if row["source"] == source]
        if norms:
            ax.hist(norms, bins=60, histtype="step", label=label)
    ax.set_title("Hidden-state L2-norm distributions")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("State count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "s0_hidden_norm_hist.pdf")
    plt.close(fig)
