from .common import *


def run(states, wo, wi, bias, kernel, args, logger):
    linear = build_linear_state(wo, wi, ridge=1e-5)
    rows = []
    for state in states:
        exact_value, probabilities = exact(
            state.vector, wo, wi, bias, args.kernel_temperature
        )
        kernel_value, _ = kernel_map(state.vector, kernel)
        linear_value = state.vector.to(wo.device) @ linear.matrix
        linear_value *= linear.target_norm / linear_value.norm().clamp_min(1e-6)
        entropy = float(
            -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
        )
        for method, value in (
            ("exact", exact_value),
            ("linear", linear_value),
            ("kernel", kernel_value),
        ):
            rows.append(
                {
                    **base(state),
                    "method": method,
                    "entropy": entropy,
                    "embedding": value.cpu().tolist(),
                }
            )
    logger.info("S4: generated exact/linear/kernel Refiner-to-Judger outputs.")
    summary = plot_s4(rows, args)
    return rows, summary


def plot_s4(rows, args):
    by_state = {}
    for row in rows:
        key = (
            row["item_id"],
            row["position"],
            row["turn_id"],
            row["agent_id"],
        )
        by_state.setdefault(key, []).append(row)
    all_keys = list(by_state)
    keys = [
        key
        for key, state_rows in by_state.items()
        if len(state_rows) == 3
        and all(
            np.isfinite(np.asarray(row["embedding"], dtype=np.float32)).all()
            for row in state_rows
        )
    ]
    random.Random(args.probe_seed).shuffle(keys)
    chosen = [
        row for key in keys[: min(2000, len(keys))] for row in by_state[key]
    ]
    if not chosen:
        return {
            "input_state_count": len(all_keys),
            "valid_state_count": 0,
            "invalid_state_count": len(all_keys),
            "mapped_embedding_count": 0,
        }
    matrix = np.asarray([row["embedding"] for row in chosen], dtype=np.float32)
    centered = matrix - matrix.mean(0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ components[:2].T
    if coordinates.shape[1] < 2:
        coordinates = np.pad(
            coordinates, ((0, 0), (0, 2 - coordinates.shape[1]))
        )
    for row, coordinate in zip(chosen, coordinates):
        row["pc1"], row["pc2"] = float(coordinate[0]), float(coordinate[1])

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for axis, method in zip(axes, ("exact", "linear", "kernel")):
        subset = [row for row in chosen if row["method"] == method]
        axis.scatter(
            [row["pc1"] for row in subset],
            [row["pc2"] for row in subset],
            c=[row["entropy"] for row in subset],
            cmap="viridis",
            s=8,
            alpha=0.7,
        )
        axis.set_title(method)
        axis.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    figure.tight_layout()
    save_figure(figure, "s4_shared_pca")
    plt.close(figure)
    write_rows(
        [{key: value for key, value in row.items() if key != "embedding"} for row in chosen],
        contextual_stem("s4_pca_coordinates"),
    )

    explained = np.square(singular_values)
    explained /= explained.sum().clip(min=np.finfo(np.float64).eps)
    pc1_ratio = float(explained[0]) if len(explained) else 0.0
    pc2_ratio = float(explained[1]) if len(explained) > 1 else 0.0
    summary = {
        "input_state_count": len(all_keys),
        "valid_state_count": len(keys),
        "invalid_state_count": len(all_keys) - len(keys),
        "mapped_embedding_count": len(chosen),
        "pca": {
            "pc1_explained_variance_ratio": pc1_ratio,
            "pc2_explained_variance_ratio": pc2_ratio,
            "pc1_pc2_cumulative_ratio": pc1_ratio + pc2_ratio,
            "by_method": {},
        },
        "paired_geometry": {},
    }
    for method in ("exact", "linear", "kernel"):
        subset = [row for row in chosen if row["method"] == method]
        summary["pca"]["by_method"][method] = {
            "pc1": clustered_metric(subset, "pc1", args),
            "pc2": clustered_metric(subset, "pc2", args),
            "entropy": clustered_metric(subset, "entropy", args),
        }

    by_all_state = {}
    for row in rows:
        key = (
            row["item_id"],
            row["position"],
            row["turn_id"],
            row["agent_id"],
        )
        by_all_state.setdefault(key, {})[row["method"]] = row
    for first, second in (
        ("exact", "linear"),
        ("exact", "kernel"),
        ("linear", "kernel"),
    ):
        pair_rows = []
        for methods in by_all_state.values():
            if first not in methods or second not in methods:
                continue
            left = np.asarray(methods[first]["embedding"], dtype=np.float64)
            right = np.asarray(methods[second]["embedding"], dtype=np.float64)
            left_norm = np.linalg.norm(left)
            right_norm = np.linalg.norm(right)
            absolute = np.linalg.norm(left - right)
            pair_rows.append(
                {
                    "item_id": methods[first]["item_id"],
                    "absolute_l2_error": float(absolute),
                    "relative_l2_error": float(absolute / max(left_norm, 1e-12)),
                    "cosine_similarity": float(
                        np.dot(left, right) / max(left_norm * right_norm, 1e-12)
                    ),
                }
            )
        summary["paired_geometry"][f"{first}_vs_{second}"] = {
            metric: clustered_metric(pair_rows, metric, args)
            for metric in (
                "absolute_l2_error",
                "relative_l2_error",
                "cosine_similarity",
            )
        }

    if args.s4_tsne:
        try:
            from sklearn.manifold import TSNE
        except ImportError as error:
            raise RuntimeError("--s4_tsne requires scikit-learn") from error
        if len(matrix) <= 50:
            raise ValueError("S4 t-SNE needs more than 50 mapping rows")
        estimator = TSNE(
            n_components=2,
            init="pca",
            perplexity=50,
            learning_rate="auto",
            max_iter=1500,
            random_state=101,
        )
        tsne = estimator.fit_transform(matrix)
        for row, coordinate in zip(chosen, tsne):
            row["tsne1"], row["tsne2"] = float(coordinate[0]), float(coordinate[1])
        write_rows(
            [
                {key: value for key, value in row.items() if key != "embedding"}
                for row in chosen
            ],
            contextual_stem("s4_tsne_coordinates"),
        )
        summary["tsne"] = {
            "kl_divergence": float(estimator.kl_divergence_),
            "by_method": {},
        }
        for method in ("exact", "linear", "kernel"):
            subset = [row for row in chosen if row["method"] == method]
            summary["tsne"]["by_method"][method] = {
                "tsne1": clustered_metric(subset, "tsne1", args),
                "tsne2": clustered_metric(subset, "tsne2", args),
            }
    return summary
