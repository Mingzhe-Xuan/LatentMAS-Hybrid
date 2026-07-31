from .common import *


ALIGNMENTS = ("linear", "kernel")
JOINT_SPACES = (
    ("hidden", "hidden"),
    ("embedding", "exact"),
)
SPACE_STYLES = {
    "hidden": {"color": "#4C78A8", "marker": "o", "label": "hidden state"},
    "embedding": {
        "color": "#F58518",
        "marker": "o",
        "label": "embedding state (exact)",
    },
    "aligned": {"color": "#54A24B", "marker": "x", "label": "aligned state"},
}


def run(states, wo, wi, bias, kernel, args, logger):
    source_dimension = int(wo.shape[1])
    target_dimension = int(wi.shape[1])
    if source_dimension != target_dimension:
        raise ValueError(
            "S4 raw-hidden joint visualization requires equal source-hidden "
            f"and target-embedding dimensions, got {source_dimension} and "
            f"{target_dimension}."
        )

    linear = build_linear_state(wo, wi, ridge=1e-5)
    rows = []
    for state in states:
        hidden_value = state.vector.to(wo.device, dtype=wo.dtype)
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
            ("hidden", hidden_value),
            ("exact", exact_value),
            ("linear", linear_value),
            ("kernel", kernel_value),
        ):
            rows.append(
                {
                    **base(state),
                    "method": method,
                    "entropy": entropy,
                    "embedding": value.detach().float().cpu().tolist(),
                }
            )
    logger.info(
        "S4: generated hidden/exact/linear/kernel Refiner-to-Judger vectors."
    )
    summary = plot_s4(rows, args)
    return rows, summary


def _state_key(row):
    return (
        row["item_id"],
        row["position"],
        row["turn_id"],
        row["agent_id"],
    )


def _valid_states(rows, probe_seed):
    by_state = {}
    for row in rows:
        by_state.setdefault(_state_key(row), {})[row["method"]] = row
    required = {"hidden", "exact", "linear", "kernel"}
    keys = []
    dimensions = set()
    for key, methods in by_state.items():
        if not required.issubset(methods):
            continue
        arrays = [
            np.asarray(methods[method]["embedding"], dtype=np.float32)
            for method in required
        ]
        if not all(array.ndim == 1 and np.isfinite(array).all() for array in arrays):
            continue
        state_dimensions = {array.shape[0] for array in arrays}
        if len(state_dimensions) != 1:
            continue
        dimensions.update(state_dimensions)
        keys.append(key)
    if len(dimensions) > 1:
        raise ValueError(
            "S4 joint visualization found inconsistent vector dimensions: "
            f"{sorted(dimensions)}"
        )
    random.Random(probe_seed).shuffle(keys)
    selected_keys = keys[: min(2000, len(keys))]
    return by_state, keys, selected_keys, (next(iter(dimensions)) if dimensions else 0)


def _joint_entries(by_state, selected_keys, alignment):
    entries = []
    specification = JOINT_SPACES + (("aligned", alignment),)
    for key in selected_keys:
        methods = by_state[key]
        for space, method in specification:
            row = methods[method]
            entries.append(
                {
                    **{
                        name: value
                        for name, value in row.items()
                        if name not in ("embedding", "method")
                    },
                    "alignment": alignment,
                    "space": space,
                    "source_method": method,
                    "embedding": row["embedding"],
                }
            )
    return entries


def _pca(entries):
    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError("S4 joint PCA requires scikit-learn") from error
    matrix = np.asarray([row["embedding"] for row in entries], dtype=np.float32)
    component_count = min(2, matrix.shape[0], matrix.shape[1])
    estimator = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=101,
    )
    coordinates = estimator.fit_transform(matrix)
    if coordinates.shape[1] < 2:
        coordinates = np.pad(
            coordinates, ((0, 0), (0, 2 - coordinates.shape[1]))
        )
    ratios = np.zeros(2, dtype=np.float64)
    ratios[: len(estimator.explained_variance_ratio_)] = (
        estimator.explained_variance_ratio_
    )
    coordinate_rows = []
    for row, coordinate in zip(entries, coordinates):
        coordinate_rows.append(
            {
                **{key: value for key, value in row.items() if key != "embedding"},
                "reducer": "pca",
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "pc1": float(coordinate[0]),
                "pc2": float(coordinate[1]),
            }
        )
    return coordinate_rows, {
        "pc1_explained_variance_ratio": float(ratios[0]),
        "pc2_explained_variance_ratio": float(ratios[1]),
        "pc1_pc2_cumulative_ratio": float(ratios.sum()),
    }


def _tsne(entries):
    try:
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise RuntimeError("--s4_tsne requires scikit-learn") from error
    matrix = np.asarray([row["embedding"] for row in entries], dtype=np.float32)
    if len(matrix) <= 50:
        raise ValueError("S4 joint t-SNE needs more than 50 vectors")
    estimator = TSNE(
        n_components=2,
        init="pca",
        perplexity=50,
        learning_rate="auto",
        max_iter=1500,
        random_state=101,
    )
    coordinates = estimator.fit_transform(matrix)
    coordinate_rows = []
    for row, coordinate in zip(entries, coordinates):
        coordinate_rows.append(
            {
                **{key: value for key, value in row.items() if key != "embedding"},
                "reducer": "tsne",
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "tsne1": float(coordinate[0]),
                "tsne2": float(coordinate[1]),
            }
        )
    return coordinate_rows, float(estimator.kl_divergence_)


def _scatter_joint(axis, rows, title, xlabel, ylabel):
    for space in ("hidden", "embedding", "aligned"):
        subset = [row for row in rows if row["space"] == space]
        style = SPACE_STYLES[space]
        axis.scatter(
            [row["x"] for row in subset],
            [row["y"] for row in subset],
            c=style["color"],
            marker=style["marker"],
            label=style["label"],
            s=12 if space != "aligned" else 16,
            alpha=0.55 if space == "hidden" else 0.7,
            linewidths=0.8 if space == "aligned" else 0,
        )
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(True, alpha=0.18)
    axis.legend(fontsize=8)


def _plot_joint_reduction(alignment, pca_rows, pca_ratios, tsne_rows=None):
    panel_count = 2 if tsne_rows is not None else 1
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(7 * panel_count, 5),
        squeeze=False,
    )
    _scatter_joint(
        axes[0, 0],
        pca_rows,
        (
            "Joint PCA\n"
            f"PC1 {pca_ratios['pc1_explained_variance_ratio']:.1%}, "
            f"PC2 {pca_ratios['pc2_explained_variance_ratio']:.1%}"
        ),
        "PC1",
        "PC2",
    )
    if tsne_rows is not None:
        _scatter_joint(
            axes[0, 1],
            tsne_rows,
            "Joint t-SNE",
            "t-SNE 1",
            "t-SNE 2",
        )
    figure.suptitle(
        f"S4 {alignment.capitalize()} alignment: hidden vs embedding vs aligned"
    )
    figure.tight_layout()
    save_figure(figure, f"s4_{alignment}_joint_reduction")
    plt.close(figure)


def _space_summary(rows, first_coordinate, second_coordinate, args):
    output = {}
    for space in ("hidden", "embedding", "aligned"):
        subset = [row for row in rows if row["space"] == space]
        output[space] = {
            first_coordinate: clustered_metric(subset, first_coordinate, args),
            second_coordinate: clustered_metric(subset, second_coordinate, args),
            "entropy": clustered_metric(subset, "entropy", args),
        }
    return output


def _paired_geometry(rows, args):
    by_state = {}
    for row in rows:
        by_state.setdefault(_state_key(row), {})[row["method"]] = row
    output = {}
    for first, second in (
        ("hidden", "exact"),
        ("hidden", "linear"),
        ("hidden", "kernel"),
        ("exact", "linear"),
        ("exact", "kernel"),
        ("linear", "kernel"),
    ):
        pair_rows = []
        for methods in by_state.values():
            if first not in methods or second not in methods:
                continue
            left = np.asarray(methods[first]["embedding"], dtype=np.float64)
            right = np.asarray(methods[second]["embedding"], dtype=np.float64)
            if left.shape != right.shape or not (
                np.isfinite(left).all() and np.isfinite(right).all()
            ):
                continue
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
        output[f"{first}_vs_{second}"] = {
            metric: clustered_metric(pair_rows, metric, args)
            for metric in (
                "absolute_l2_error",
                "relative_l2_error",
                "cosine_similarity",
            )
        }
    return output


def plot_s4(rows, args):
    by_state, valid_keys, selected_keys, dimension = _valid_states(
        rows, args.probe_seed
    )
    if not selected_keys:
        input_state_count = len({_state_key(row) for row in rows})
        return {
            "input_state_count": input_state_count,
            "valid_state_count": 0,
            "invalid_state_count": input_state_count,
            "selected_state_count": 0,
            "mapped_embedding_count": 0,
            "joint_point_count_per_alignment": 0,
        }

    pca_coordinate_rows = []
    tsne_coordinate_rows = []
    pca_by_alignment = {}
    tsne_by_alignment = {}
    reductions = {}
    for alignment in ALIGNMENTS:
        entries = _joint_entries(by_state, selected_keys, alignment)
        pca_rows, pca_ratios = _pca(entries)
        pca_coordinate_rows.extend(pca_rows)
        pca_by_alignment[alignment] = {
            **pca_ratios,
            "fit": "joint_hidden_embedding_aligned",
            "by_space": _space_summary(pca_rows, "pc1", "pc2", args),
        }
        tsne_rows = None
        if args.s4_tsne:
            tsne_rows, kl_divergence = _tsne(entries)
            tsne_coordinate_rows.extend(tsne_rows)
            tsne_by_alignment[alignment] = {
                "fit": "joint_hidden_embedding_aligned",
                "kl_divergence": kl_divergence,
                "by_space": _space_summary(
                    tsne_rows, "tsne1", "tsne2", args
                ),
            }
        reductions[alignment] = (pca_rows, pca_ratios, tsne_rows)

    write_rows(pca_coordinate_rows, contextual_stem("s4_joint_pca_coordinates"))
    if tsne_coordinate_rows:
        write_rows(
            tsne_coordinate_rows,
            contextual_stem("s4_joint_tsne_coordinates"),
        )
    for alignment, (pca_rows, ratios, tsne_rows) in reductions.items():
        _plot_joint_reduction(alignment, pca_rows, ratios, tsne_rows)

    input_state_count = len({_state_key(row) for row in rows})
    summary = {
        "input_state_count": input_state_count,
        "valid_state_count": len(valid_keys),
        "invalid_state_count": input_state_count - len(valid_keys),
        "selected_state_count": len(selected_keys),
        "mapped_embedding_count": len(selected_keys) * 4,
        "vector_dimension": dimension,
        "joint_point_count_per_alignment": len(selected_keys) * 3,
        "preprocessing": (
            "joint fit on raw vectors; global PCA centering only; "
            "no per-space scaling or standardization"
        ),
        "spaces": {
            "hidden": "raw Refiner hidden state immediately before logits",
            "embedding": "exact probability-weighted target input embedding",
            "aligned": "linear- or kernel-aligned target-space state",
        },
        "pca": {
            "fit": "separate joint fit for each alignment",
            "by_alignment": pca_by_alignment,
        },
        "paired_geometry": _paired_geometry(rows, args),
    }
    if args.s4_tsne:
        summary["tsne"] = {
            "fit": "separate joint fit for each alignment",
            "parameters": {
                "init": "pca",
                "perplexity": 50,
                "learning_rate": "auto",
                "max_iter": 1500,
                "random_state": 101,
            },
            "by_alignment": tsne_by_alignment,
        }
    return summary