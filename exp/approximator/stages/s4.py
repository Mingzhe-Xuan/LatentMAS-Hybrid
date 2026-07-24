# PLAN-V2 STATUS: see ../IMPLEMENTATION_STATUS.md#s4----communication-space-geometry.
from .common import *

def s4(states, wo, wi, bias, k, args):
    lin = build_linear_state(wo, wi, ridge=1e-5)
    rows = []
    for s in states:
        f, _ = exact(s.vector, wo, wi, bias, args.tau)
        h, _ = kernel_map(s.vector, k)
        l = s.vector.to(wo.device) @ lin.matrix
        l = l * (lin.target_norm / l.norm().clamp_min(1e-6))
        logits = wo @ (s.vector.to(wo.device) / args.tau)
        if bias is not None: logits = logits + bias
        p = torch.softmax(logits, 0)
        entropy = float(-(p * p.clamp_min(1e-30).log()).sum())
        for name, v in (("exact", f), ("linear", l), ("kernel", h)):
            rows.append(
                {
                    **base(s),
                    "method": name,
                    "entropy": entropy,
                    "embedding": v.cpu().tolist(),
                }
            )
    return rows
def plot_s4(rows, args):
    # Sample state identities first, so each selected state contributes all three
    # methods and prompt/reply have equal weight in the PCA fitting population.
    rng = random.Random(args.probe_seed)
    chosen = []
    for src in SOURCES:
        by_state = {}
        for r in rows:
            if r["source"] == src:
                by_state.setdefault(
                    (r["item_id"], r["position"], r["turn_id"], r["agent_id"]), []
                ).append(r)
        keys = list(by_state)
        rng.shuffle(keys)
        for key in keys[: min(2000, len(keys))]:
            chosen.extend(by_state[key])
    X = np.array([r["embedding"] for r in chosen], dtype=np.float32)
    X -= X.mean(0)
    _, _, v = np.linalg.svd(X, full_matrices=False)
    Z = X @ v[:2].T
    for r, z in zip(chosen, Z):
        r["pc1"], r["pc2"] = float(z[0]), float(z[1])
    entropy = np.array([r["entropy"] for r in chosen]); cuts = np.quantile(entropy, [.25, .5, .75])
    for r in chosen: r["entropy_quartile"] = int(np.searchsorted(cuts, r["entropy"], side="right") + 1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    exact_by_key={(r["item_id"],r["source"],r["position"],r["turn_id"],r["agent_id"]):r for r in chosen if r["method"]=="exact"}
    colors=["#440154", "#31688e", "#35b779", "#fde725"]
    for ax, m in zip(axs, ("exact", "linear", "kernel")):
        q = [r for r in chosen if r["method"] == m]
        if m != "exact":
            ax.scatter([r["pc1"] for r in exact_by_key.values()], [r["pc2"] for r in exact_by_key.values()], c="lightgray", s=5, alpha=.35, label="exact")
        for source, marker in (("prompt", "o"), ("reply", "^")):
            for quartile in range(1,5):
                z=[r for r in q if r["source"]==source and r["entropy_quartile"]==quartile]
                ax.scatter([r["pc1"] for r in z], [r["pc2"] for r in z], color=colors[quartile-1], marker=marker, s=8, alpha=.7)
        if m != "exact":
            for r in q[:100]:
                key=(r["item_id"],r["source"],r["position"],r["turn_id"],r["agent_id"]); e=exact_by_key.get(key)
                if e: ax.annotate("", xy=(r["pc1"],r["pc2"]), xytext=(e["pc1"],e["pc2"]), arrowprops={"arrowstyle":"->","color":"gray","alpha":.2,"lw":.4})
        ax.set_title(m); ax.set_xlabel("PC1")
    axs[0].set_ylabel("PC2")
    (RESULT / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(RESULT / "figures" / "s4_shared_pca.pdf")
    plt.close(fig)
    write_rows(
        [{k: v for k, v in r.items() if k != "embedding"} for r in chosen],
        "s4_pca_coordinates",
    )
    if args.s4_tsne:
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise RuntimeError("--s4_tsne requires scikit-learn") from exc
        if len(X) <= 50:
            raise ValueError("S4 t-SNE needs more than 50 sampled mapping rows")
        Zt = TSNE(
            n_components=2,
            init="pca",
            perplexity=50,
            learning_rate="auto",
            n_iter=1500,
            random_state=101,
        ).fit_transform(X)
        for r, z in zip(chosen, Zt):
            r["tsne1"], r["tsne2"] = float(z[0]), float(z[1])
        write_rows(
            [{k: v for k, v in r.items() if k != "embedding"} for r in chosen],
            "s4_tsne_coordinates",
        )


def run(states, wo, wi, bias, kernel, args, logger):
    logger.info("S4: B-space mappings and shared PCA start for %d states.", len(states))
    rows = s4(states, wo, wi, bias, kernel, args)
    logger.info("S4: mappings completed (%d rows); fitting shared PCA.", len(rows))
    plot_s4(rows, args)
    logger.info("S4: shared PCA%s completed.", " and t-SNE" if args.s4_tsne else "")
    return rows