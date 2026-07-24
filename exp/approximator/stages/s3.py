# PLAN-V2 STATUS: see ../IMPLEMENTATION_STATUS.md#s3----fixed-text-orf-variance.
from .common import *

def select_s3(states, n):
    out = []
    for item in range(n):
        ps = [s for s in states if s.item_id == item and s.source == "prompt"]
        rs = [s for s in states if s.item_id == item and s.source == "reply"]
        if ps:
            out.append(ps[-1])
        out += [rs[i] for i in positions(len(rs), 16)]
    return out


def s3(states, wo, wi, bias, args, logger=None):
    selected = select_s3(states, args.s3_max_questions)
    rows = []
    for m in (512, 1024, 2048):
        if logger: logger.info("S3 progress: feature count m=%d started.", m)
        for tau_index, tau in enumerate(np.arange(0.5, 2.01, 0.1), start=1):
            if logger: logger.info("S3 progress: m=%d, tau=%0.1f (%d/16).", m, tau, tau_index)
            accum = {j: {"f": [], "kernel": []} for j in range(len(selected))}
            for seed in range(1001, 1001 + args.s3_replicates):
                k = build_kernel_state(
                    wo,
                    wi,
                    bias,
                    feature_count=m,
                    temperature=float(tau),
                    seed=seed,
                    chunk_size=args.kernel_chunk_size,
                )
                for j, s in enumerate(selected):
                    h, _ = kernel_map(s.vector, k)
                    _, p = exact(s.vector, wo, wi, bias, float(tau))
                    accum[j]["f"].append(h.cpu())
                    x = s.vector.to(wo.device) / tau
                    for band, i in rank_ids(p, s, args):
                        accum[j]["kernel"].append(
                            (
                                band,
                                float(
                                    positive_features(wo[i : i + 1], k.omega)[0]
                                    @ positive_features(x[None], k.omega)[0]
                                ),
                            )
                        )
            for j, s in enumerate(selected):
                st = torch.stack(accum[j]["f"])
                f, _ = exact(s.vector, wo, wi, bias, float(tau))
                var = st.var(0, unbiased=True).mean()
                mean = st.mean(0)
                r = base(s)
                r.update(
                    m=m,
                    tau=float(tau),
                    kind="F",
                    variance=float(var),
                    std=float(var.sqrt()),
                    relative_std=float(var.sqrt() / f.cpu().norm().clamp_min(1e-8)),
                    bias2=float((mean - f.cpu()).square().mean()),
                    mse=float((mean - f.cpu()).square().mean() + var),
                )
                rows.append(r)
                bands = {}
                for band, v in accum[j]["kernel"]:
                    bands.setdefault(band, []).append(v)
                for band, v in bands.items():
                    a = np.array(v)
                    rows.append(
                        {
                            **base(s),
                            "m": m,
                            "tau": float(tau),
                            "kind": "kernel",
                            "rank_band": band,
                            "variance": float(a.var(ddof=1)),
                            "std": float(a.std(ddof=1)),
                            "relative_std": float(
                                a.std(ddof=1) / (abs(a.mean()) + 1e-8)
                            ),
                        }
                    )
    return rows


def plot_s3(rows):
    fig, ax = plt.subplots(figsize=(7, 4))
    for source in SOURCES:
        q = [
            r
            for r in rows
            if r["kind"] == "F" and r["source"] == source and r["m"] == 2048
        ]
        for tau in sorted(set(r["tau"] for r in q)):
            v = [r["variance"] for r in q if r["tau"] == tau]
            if v:
                ax.scatter(
                    [tau],
                    [np.median(v)],
                    label=source if tau == 0.5 else None,
                    marker="o" if source == "prompt" else "x",
                )
    ax.set_yscale("log")
    ax.set_xlabel("tau")
    ax.set_ylabel("median F variance")
    ax.legend()
    (RESULT / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(RESULT / "figures" / "s3_variance_tau.pdf")
    plt.close(fig)
def run(states, wo, wi, bias, args, logger):
    logger.info("S3: fixed-text variance sweep starts; 3 feature counts x 16 temperatures x %d ORF seeds.", args.s3_replicates)
    rows = s3(states, wo, wi, bias, args, logger)
    logger.info("S3: variance sweep completed with %d rows.", len(rows))
    return rows
def plot_forest(rows):
    figure_dir = RESULT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    q=[r for r in rows if r["kind"] == "F" and r["m"] == 2048 and abs(r["tau"] - 1.0) < 1e-6]
    grouped={}
    for row in q: grouped.setdefault(row["item_id"], {}).setdefault(row["source"], []).append(row["variance"])
    points=[]
    for item, values in grouped.items():
        if "prompt" in values and "reply" in values:
            points.append((item, np.mean(values["reply"]) / max(np.mean(values["prompt"]), 1e-30)))
    points=sorted(points)[:50]
    if not points: return
    fig, ax=plt.subplots(figsize=(7, max(4, len(points)*.13)))
    ax.hlines(range(len(points)), 1, [ratio for _, ratio in points], color="tab:purple", alpha=.7)
    ax.scatter([ratio for _, ratio in points], range(len(points)), s=14, color="tab:purple")
    ax.axvline(1, color="black", linestyle="--"); ax.set_xscale("log"); ax.set_yticks(range(len(points)), [str(item) for item,_ in points]); ax.set_xlabel("reply / prompt F variance (m=2048, tau=1)"); ax.set_ylabel("question id")
    fig.tight_layout(); fig.savefig(figure_dir / "s3_reply_prompt_forest.pdf"); plt.close(fig)