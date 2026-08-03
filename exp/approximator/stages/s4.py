from .common import *


ALIGNMENTS = ("linear", "kernel")
REFERENCES = ("exact", "text")
SPACE_STYLES = {
    "hidden": {"color": "#4C78A8", "marker": "o"},
    "reference": {"color": "#F58518", "marker": "o"},
    "aligned": {"color": "#54A24B", "marker": "x"},
}


def run(states, text_states, wo, wi, bias, kernel, args, logger):
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
                    "trajectory_kind": "latent_mas_hybrid",
                    "method": method,
                    "entropy": entropy,
                    "embedding": value.detach().float().cpu().tolist(),
                }
            )
    for state in text_states:
        vector = state["vector"]
        if not (
            vector.ndim == 1
            and int(vector.shape[0]) == target_dimension
            and torch.isfinite(vector).all()
        ):
            raise ValueError(
                "S4 TextMAS token embeddings must be finite one-dimensional "
                f"Judger-space vectors of size {target_dimension}."
            )
        rows.append(
            {
                **{
                    key: value
                    for key, value in state.items()
                    if key != "vector"
                },
                "source": "refiner_text_mas_token_embedding",
                "trajectory_kind": "text_mas",
                "method": "text",
                "entropy": None,
                "embedding": vector.detach().float().cpu().tolist(),
            }
        )
    logger.info(
        "S4: generated hidden/exact/linear/kernel vectors and loaded %d TextMAS tokens.",
        len(text_states),
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


def _valid_states(rows):
    by_state = {}
    for row in rows:
        if row["method"] == "text":
            continue
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
            "S4 joint visualization found inconsistent latent vector dimensions: "
            f"{sorted(dimensions)}"
        )
    return by_state, keys, (next(iter(dimensions)) if dimensions else 0)


def _select_balanced(rows, valid_keys, probe_seed, include_text):
    rng = random.Random(probe_seed)
    keys_by_item = {}
    for key in valid_keys:
        keys_by_item.setdefault(key[0], []).append(key)
    text_by_item = {}
    for row in rows:
        if row["method"] == "text":
            text_by_item.setdefault(row["item_id"], []).append(row)

    if not include_text:
        keys = list(valid_keys)
        rng.shuffle(keys)
        return keys[: min(2000, len(keys))], {}

    item_ids = sorted(set(keys_by_item) & set(text_by_item))
    rng.shuffle(item_ids)
    selected_keys = []
    selected_text = {}
    for item_id in item_ids:
        latent = list(keys_by_item[item_id])
        text = list(text_by_item[item_id])
        rng.shuffle(latent)
        rng.shuffle(text)
        count = min(len(latent), len(text), 2000 - len(selected_keys))
        for key, text_row in zip(latent[:count], text[:count]):
            selected_keys.append(key)
            selected_text[key] = text_row
        if len(selected_keys) == 2000:
            break
    return selected_keys, selected_text


def _entry(row, *, alignment, reference, space, source_method):
    vector = np.asarray(row["embedding"], dtype=np.float32)
    return {
        **{
            name: value
            for name, value in row.items()
            if name not in ("embedding", "method")
        },
        "alignment": alignment,
        "reference": reference,
        "space": space,
        "source_method": source_method,
        "embedding_norm": float(np.linalg.norm(vector)),
        "embedding": row["embedding"],
    }


def _joint_entries(by_state, selected_keys, selected_text, alignment, reference):
    entries = []
    for key in selected_keys:
        methods = by_state[key]
        reference_row = methods["exact"] if reference == "exact" else selected_text[key]
        for space, method, row in (
            ("hidden", "hidden", methods["hidden"]),
            ("reference", reference, reference_row),
            ("aligned", alignment, methods[alignment]),
        ):
            entries.append(
                _entry(
                    row,
                    alignment=alignment,
                    reference=reference,
                    space=space,
                    source_method=method,
                )
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
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))
    ratios = np.zeros(2, dtype=np.float64)
    ratios[: len(estimator.explained_variance_ratio_)] = estimator.explained_variance_ratio_
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


def _space_label(space, reference):
    return {
        "hidden": "hidden state",
        "reference": f"{reference} state",
        "aligned": "aligned state",
    }[space]


def _scatter_joint(axis, rows, reference, title, xlabel, ylabel):
    for space in ("hidden", "reference", "aligned"):
        subset = [row for row in rows if row["space"] == space]
        style = SPACE_STYLES[space]
        axis.scatter(
            [row["x"] for row in subset],
            [row["y"] for row in subset],
            c=style["color"],
            marker=style["marker"],
            label=_space_label(space, reference),
            s=12 if space != "aligned" else 16,
            alpha=0.55 if space == "hidden" else 0.7,
            linewidths=0.8 if space == "aligned" else 0,
        )
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(True, alpha=0.18)
    axis.legend(fontsize=8)


def _plot_joint_reduction(alignment, reference, pca_rows, pca_ratios, tsne_rows=None):
    panel_count = 2 if tsne_rows is not None else 1
    figure, axes = plt.subplots(1, panel_count, figsize=(7 * panel_count, 5), squeeze=False)
    _scatter_joint(
        axes[0, 0],
        pca_rows,
        reference,
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
            axes[0, 1], tsne_rows, reference, "Joint t-SNE", "t-SNE 1", "t-SNE 2"
        )
    figure.suptitle(
        f"S4 {alignment.capitalize()} alignment: {reference} vs hidden vs aligned"
    )
    figure.tight_layout()
    save_figure(figure, f"s4_{alignment}_{reference}_joint_reduction")
    plt.close(figure)


def _space_summary(rows, first_coordinate, second_coordinate, args):
    output = {}
    for space in ("hidden", "reference", "aligned"):
        subset = [row for row in rows if row["space"] == space]
        output[space] = {
            first_coordinate: clustered_metric(subset, first_coordinate, args),
            second_coordinate: clustered_metric(subset, second_coordinate, args),
            "entropy": clustered_metric(subset, "entropy", args),
            "embedding_norm": clustered_metric(subset, "embedding_norm", args),
        }
    return output


def _distribution_geometry(entries, args):
    by_item = {}
    for row in entries:
        by_item.setdefault(row["item_id"], {}).setdefault(row["space"], []).append(
            np.asarray(row["embedding"], dtype=np.float64)
        )
    output = {}
    for first, second in (
        ("hidden", "reference"),
        ("hidden", "aligned"),
        ("reference", "aligned"),
    ):
        metric_rows = []
        for item_id, spaces in by_item.items():
            if first not in spaces or second not in spaces:
                continue
            left = np.mean(spaces[first], axis=0)
            right = np.mean(spaces[second], axis=0)
            metric_rows.append(
                {"item_id": item_id, "centroid_l2_distance": float(np.linalg.norm(left - right))}
            )
        output[f"{first}_vs_{second}"] = {
            "centroid_l2_distance": clustered_metric(
                metric_rows, "centroid_l2_distance", args
            )
        }
    return output


def _paired_geometry(rows, args, selected_keys=None):
    selected_keys = None if selected_keys is None else set(selected_keys)
    by_state = {}
    for row in rows:
        if row["method"] == "text":
            continue
        key = _state_key(row)
        if selected_keys is not None and key not in selected_keys:
            continue
        by_state.setdefault(key, {})[row["method"]] = row
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
            if left.shape != right.shape or not (np.isfinite(left).all() and np.isfinite(right).all()):
                continue
            left_norm = np.linalg.norm(left)
            right_norm = np.linalg.norm(right)
            absolute = np.linalg.norm(left - right)
            pair_rows.append(
                {
                    "item_id": methods[first]["item_id"],
                    "absolute_l2_error": float(absolute),
                    "relative_l2_error": float(absolute / max(left_norm, 1e-12)),
                    "cosine_similarity": float(np.dot(left, right) / max(left_norm * right_norm, 1e-12)),
                }
            )
        output[f"{first}_vs_{second}"] = {
            metric: clustered_metric(pair_rows, metric, args)
            for metric in ("absolute_l2_error", "relative_l2_error", "cosine_similarity")
        }
    return output


def plot_s4(rows, args):
    by_state, valid_keys, dimension = _valid_states(rows)
    include_text = bool(getattr(args, "s4_text_mas", True))
    selected_keys, selected_text = _select_balanced(
        rows, valid_keys, args.probe_seed, include_text
    )
    input_state_count = len(by_state)
    text_token_count = sum(row["method"] == "text" for row in rows)
    if include_text and not selected_text:
        raise ValueError("S4 TextMAS is enabled but no balanced Refiner text tokens are available.")
    if not selected_keys:
        return {
            "input_state_count": input_state_count,
            "valid_state_count": 0,
            "invalid_state_count": input_state_count,
            "selected_state_count": 0,
            "text_token_count": text_token_count,
        }

    references = REFERENCES if include_text else ("exact",)
    pca_coordinate_rows = []
    tsne_coordinate_rows = []
    reductions = {}
    pca_summary = {}
    tsne_summary = {}
    geometry_summary = {}
    for alignment in ALIGNMENTS:
        pca_summary[alignment] = {}
        tsne_summary[alignment] = {}
        geometry_summary[alignment] = {}
        for reference in references:
            entries = _joint_entries(
                by_state, selected_keys, selected_text, alignment, reference
            )
            pca_rows, ratios = _pca(entries)
            pca_coordinate_rows.extend(pca_rows)
            pca_summary[alignment][reference] = {
                **ratios,
                "fit": "joint_hidden_reference_aligned",
                "by_space": _space_summary(pca_rows, "pc1", "pc2", args),
            }
            geometry_summary[alignment][reference] = _distribution_geometry(entries, args)
            tsne_rows = None
            if args.s4_tsne:
                tsne_rows, kl_divergence = _tsne(entries)
                tsne_coordinate_rows.extend(tsne_rows)
                tsne_summary[alignment][reference] = {
                    "fit": "joint_hidden_reference_aligned",
                    "kl_divergence": kl_divergence,
                    "by_space": _space_summary(tsne_rows, "tsne1", "tsne2", args),
                }
            reductions[(alignment, reference)] = (pca_rows, ratios, tsne_rows)

    write_rows(pca_coordinate_rows, contextual_stem("s4_joint_pca_coordinates"))
    if tsne_coordinate_rows:
        write_rows(tsne_coordinate_rows, contextual_stem("s4_joint_tsne_coordinates"))
    for (alignment, reference), (pca_rows, ratios, tsne_rows) in reductions.items():
        _plot_joint_reduction(alignment, reference, pca_rows, ratios, tsne_rows)

    summary = {
        "input_state_count": input_state_count,
        "valid_state_count": len(valid_keys),
        "invalid_state_count": input_state_count - len(valid_keys),
        "selected_state_count": len(selected_keys),
        "text_mas_enabled": include_text,
        "text_token_count": text_token_count,
        "selected_text_token_count": len(selected_text),
        "vector_dimension": dimension,
        "joint_point_count_per_reduction": len(selected_keys) * 3,
        "preprocessing": (
            "each alignment/reference triple is fit independently on balanced raw vectors; "
            "global PCA centering only; no per-space scaling or standardization"
        ),
        "spaces": {
            "hidden": "raw Refiner latent hidden state immediately before logits",
            "exact": "exact probability-weighted Judger input embedding",
            "text": "actual TextMAS Refiner tokens in the Judger input embedding table",
            "aligned": "linear- or kernel-aligned Judger-space state",
        },
        "pca": {
            "fit": "independent joint fit for every alignment x reference",
            "by_alignment_and_reference": pca_summary,
        },
        "distribution_geometry": geometry_summary,
        "paired_geometry": _paired_geometry(rows, args, selected_keys),
        "pairing_note": (
            "TextMAS tokens are balanced by question and count but are not treated as "
            "semantic step-wise pairs with latent states."
        ),
    }
    if args.s4_tsne:
        summary["tsne"] = {
            "fit": "independent joint fit for every alignment x reference",
            "parameters": {
                "init": "pca",
                "perplexity": 50,
                "learning_rate": "auto",
                "max_iter": 1500,
                "random_state": 101,
            },
            "by_alignment_and_reference": tsne_summary,
        }
    return summary
