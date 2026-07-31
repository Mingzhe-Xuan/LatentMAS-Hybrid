import math

from .common import *


def _single_kernel_log_estimate(key, query, omega):
    """Evaluate log(phi(key)^T phi(query)) without feature underflow."""
    key = key.float()
    query = query.float()
    log_terms = (
        omega @ (key + query)
        - 0.5 * (key.square().sum() + query.square().sum())
    )
    return torch.logsumexp(log_terms.double(), dim=0) - math.log(omega.shape[0])


def _log_sample_variance(log_values):
    """Return log sample variance for positive values supplied in log space."""
    log_values = torch.stack(log_values).double()
    anchor = log_values.max()
    scaled_variance = (log_values - anchor).exp().var(unbiased=True)
    if not torch.isfinite(scaled_variance) or scaled_variance <= 0:
        return torch.tensor(float("-inf"), dtype=torch.float64)
    return 2 * anchor + scaled_variance.log()


def _tail_rows_for_configuration(
    error_rows,
    feature_count,
    temperature,
    seeds,
    epsilons,
):
    """Aggregate absolute L2 tail events by question and random seed."""
    by_question = {}
    for row in error_rows:
        by_question.setdefault(row["item_id"], []).append(
            row["errors"].detach().double().cpu()
        )

    question_rows = []
    seed_accumulators = {
        (float(epsilon), int(seed)): [] for epsilon in epsilons for seed in seeds
    }
    for item_id, state_errors in sorted(by_question.items()):
        matrix = torch.stack(state_errors)
        if matrix.ndim != 2 or matrix.shape[1] != len(seeds):
            raise ValueError("S3 tail error matrix has an invalid replicate shape")
        finite = torch.isfinite(matrix)
        for epsilon in epsilons:
            epsilon = float(epsilon)
            exceedance = finite & (matrix > epsilon)
            valid_count = int(finite.sum())
            exceedance_count = int(exceedance.sum())
            finite_squared = matrix.square()[finite]
            second_moment = (
                float(finite_squared.mean()) if finite_squared.numel() else None
            )
            empirical = (
                float(exceedance_count / valid_count) if valid_count else None
            )
            bound = (
                min(1.0, second_moment / epsilon**2)
                if second_moment is not None
                else None
            )
            question_rows.append(
                {
                    "item_id": int(item_id),
                    "m": int(feature_count),
                    "tau": float(temperature),
                    "epsilon": epsilon,
                    "state_count": int(matrix.shape[0]),
                    "replicate_count": int(matrix.shape[1]),
                    "observation_count": int(matrix.numel()),
                    "valid_observation_count": valid_count,
                    "invalid_observation_count": int(matrix.numel() - valid_count),
                    "exceedance_count": exceedance_count,
                    "empirical_tail_probability": empirical,
                    "mean_squared_l2_error": second_moment,
                    "markov_mse_upper_bound": bound,
                    "bound_gap": (
                        float(bound - empirical)
                        if bound is not None and empirical is not None
                        else None
                    ),
                }
            )
            for seed_index, seed in enumerate(seeds):
                seed_finite = finite[:, seed_index]
                seed_valid_count = int(seed_finite.sum())
                seed_exceedance_count = int(exceedance[:, seed_index].sum())
                seed_accumulators[(epsilon, int(seed))].append(
                    {
                        "valid_count": seed_valid_count,
                        "invalid_count": int(len(seed_finite) - seed_valid_count),
                        "exceedance_count": seed_exceedance_count,
                        "rate": (
                            float(seed_exceedance_count / seed_valid_count)
                            if seed_valid_count
                            else None
                        ),
                    }
                )

    seed_rows = []
    for (epsilon, seed), question_values in sorted(seed_accumulators.items()):
        rates = [
            value["rate"]
            for value in question_values
            if value["rate"] is not None
        ]
        valid_count = sum(value["valid_count"] for value in question_values)
        invalid_count = sum(value["invalid_count"] for value in question_values)
        exceedance_count = sum(
            value["exceedance_count"] for value in question_values
        )
        seed_rows.append(
            {
                "m": int(feature_count),
                "tau": float(temperature),
                "epsilon": float(epsilon),
                "seed": int(seed),
                "question_count": len(rates),
                "state_count": valid_count + invalid_count,
                "valid_state_count": valid_count,
                "invalid_state_count": invalid_count,
                "exceedance_count": exceedance_count,
                "exceedance_rate": float(np.mean(rates)) if rates else None,
                "pooled_exceedance_rate": (
                    float(exceedance_count / valid_count)
                    if valid_count
                    else None
                ),
            }
        )
    return question_rows, seed_rows


def select_s3(states, max_questions, probe_seed):
    by_question = {}
    for state in states:
        by_question.setdefault(state.item_id, []).append(state)
    item_ids = sorted(by_question)
    random.Random(probe_seed).shuffle(item_ids)
    selected = []
    for item_id in item_ids[:max_questions]:
        candidates = sorted(by_question[item_id], key=lambda state: state.position)
        if len(candidates) <= 16:
            selected.extend(candidates)
        else:
            indices = random.Random(probe_seed + item_id).sample(
                range(len(candidates)), 16
            )
            selected.extend(candidates[index] for index in sorted(indices))
    return selected


def run(states, wo, wi, bias, args, logger):
    selected = select_s3(states, args.s3_max_questions, args.probe_seed)
    rows = []
    tail_question_rows = []
    tail_seed_rows = []
    seeds = list(range(1001, 1001 + args.s3_replicates))
    for feature_count in (512, 1024, 2048):
        for temperature in np.arange(0.5, 2.01, 0.1):
            temperature = float(temperature)
            logger.info("S3 refiner states: m=%d tau=%.1f", feature_count, temperature)
            exact_values = []
            probes = []
            for state in selected:
                exact_value, probabilities = exact(state.vector, wo, wi, bias, temperature)
                query = state.vector.to(wo.device) / temperature
                exact_values.append(exact_value.cpu())
                probes.append(
                    [
                        (
                            rank_band,
                            token_id,
                            float(
                                torch.exp(
                                    (wo[token_id] @ query).double()
                                )
                            ),
                        )
                        for rank_band, token_id in rank_ids(probabilities, state, args)
                    ]
                )

            f_accumulators = [[] for _ in selected]
            kernel_accumulators = [
                [[] for _ in state_probes] for state_probes in probes
            ]
            for seed in seeds:
                kernel = build_kernel_state(
                    wo,
                    wi,
                    bias,
                    feature_count=feature_count,
                    temperature=temperature,
                    seed=seed,
                    chunk_size=args.kernel_chunk_size,
                )
                for index, state in enumerate(selected):
                    approximate, _ = kernel_map(state.vector, kernel)
                    f_accumulators[index].append(approximate.cpu())
                    query = state.vector.to(wo.device) / temperature
                    for probe_index, (_, token_id, _) in enumerate(probes[index]):
                        log_estimate = _single_kernel_log_estimate(
                            wo[token_id], query, kernel.omega
                        )
                        kernel_accumulators[index][probe_index].append(
                            log_estimate.cpu()
                        )

            error_rows = []
            for state_index, state in enumerate(selected):
                exact_value = exact_values[state_index]
                stack = torch.stack(f_accumulators[state_index])
                l2_errors = (stack.double() - exact_value.double()).norm(dim=1)
                error_rows.append(
                    {
                        "item_id": state.item_id,
                        "errors": l2_errors,
                    }
                )
                variance = stack.var(0, unbiased=True).mean()
                mean = stack.mean(0)
                bias_squared = (mean - exact_value).square().mean()
                rows.append(
                    {
                        **base(state),
                        "m": feature_count,
                        "tau": temperature,
                        "kind": "F",
                        "variance": float(variance),
                        "std": float(variance.sqrt()),
                        "relative_std": float(
                            variance.sqrt()
                            / exact_value.cpu().norm().clamp_min(1e-8)
                        ),
                        "bias2": float(bias_squared),
                        "mse": float(bias_squared + variance),
                        "l2_error_mean": float(l2_errors.mean()),
                        "l2_error_rmse": float(l2_errors.square().mean().sqrt()),
                        "mean_squared_l2_error": float(l2_errors.square().mean()),
                    }
                )

                for probe, values in zip(
                    probes[state_index], kernel_accumulators[state_index]
                ):
                    rank_band, token_id, truth = probe
                    log_estimates = torch.stack(values).double()
                    log_variance = _log_sample_variance(values)
                    kernel_variance = log_variance.exp()
                    log_kernel_mean = (
                        torch.logsumexp(log_estimates, dim=0)
                        - math.log(len(log_estimates))
                    )
                    kernel_mean = log_kernel_mean.exp()
                    kernel_bias_squared = (kernel_mean - truth).square()
                    rows.append(
                        {
                            **base(state),
                            "m": feature_count,
                            "tau": temperature,
                            "kind": "kernel",
                            "rank_band": rank_band,
                            "token_id": token_id,
                            "kernel_truth": truth,
                            "kernel_mean": float(kernel_mean),
                            "variance": float(kernel_variance),
                            "log10_variance": float(
                                log_variance / math.log(10)
                            ),
                            "std": float(kernel_variance.sqrt()),
                            "relative_std": float(
                                kernel_variance.sqrt() / max(abs(truth), 1e-8)
                            ),
                            "bias2": float(kernel_bias_squared),
                            "mse": float(kernel_bias_squared + kernel_variance),
                        }
                    )
            question_tail, seed_tail = _tail_rows_for_configuration(
                error_rows,
                feature_count,
                temperature,
                seeds,
                args.s3_tail_epsilons,
            )
            tail_question_rows.extend(question_tail)
            tail_seed_rows.extend(seed_tail)
    return rows, tail_question_rows, tail_seed_rows


def plot_s3(rows):
    figure, axis = plt.subplots(figsize=(7, 4))
    for feature_count in (512, 1024, 2048):
        subset = [
            row
            for row in rows
            if row["m"] == feature_count and row["kind"] == "F"
        ]
        temperatures = sorted({row["tau"] for row in subset})
        medians = [
            np.median(
                [row["variance"] for row in subset if row["tau"] == temperature]
            )
            for temperature in temperatures
        ]
        axis.plot(temperatures, medians, label=f"m={feature_count}")
    axis.set_yscale("log")
    axis.set(xlabel="kernel temperature", ylabel="median F variance")
    axis.legend()
    figure.tight_layout()
    save_figure(figure, "s3_variance_tau")
    plt.close(figure)


def plot_kernel_variance(rows):
    figure, axis = plt.subplots(figsize=(8, 4))
    subset = [
        row
        for row in rows
        if row["kind"] == "kernel" and row["m"] == 2048
    ]
    for rank_band in sorted({row["rank_band"] for row in subset}):
        points = []
        for temperature in sorted({row["tau"] for row in subset}):
            values = [
                row.get("log10_variance", np.nan)
                for row in subset
                if row["rank_band"] == rank_band
                and row["tau"] == temperature
            ]
            finite = [value for value in values if np.isfinite(value)]
            if finite:
                points.append((temperature, float(np.median(finite))))
        if points:
            axis.plot(
                [temperature for temperature, _ in points],
                [median for _, median in points],
                label=rank_band,
            )
    axis.set(
        xlabel="kernel temperature",
        ylabel="median log10 Var_seed[phi(w)^T phi(q)] (m=2048)",
    )
    if axis.lines:
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "No finite log-variance values.\n"
            "Rerun S3 to replace the old underflowed cache.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    figure.tight_layout()
    save_figure(figure, "s3_single_kernel_variance_tau")
    plt.close(figure)


def _question_curve_point(rows, metric, args):
    values = [
        row.get(metric)
        for row in rows
        if row.get(metric) is not None and np.isfinite(row.get(metric))
    ]
    if not values:
        return None
    return describe_values(
        values,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.probe_seed,
    )


def plot_tail_epsilon(question_rows, args):
    """Compare empirical tails and the MSE/epsilon^2 upper bound."""
    figure, axis = plt.subplots(figsize=(8, 5))
    subset = [row for row in question_rows if abs(row["tau"] - 1.0) < 1e-8]
    for feature_count in sorted({row["m"] for row in subset}):
        points = []
        for epsilon in sorted({row["epsilon"] for row in subset}):
            cell = [
                row
                for row in subset
                if row["m"] == feature_count and row["epsilon"] == epsilon
            ]
            empirical = _question_curve_point(
                cell, "empirical_tail_probability", args
            )
            bound = _question_curve_point(cell, "markov_mse_upper_bound", args)
            if empirical and bound:
                points.append((epsilon, empirical, bound))
        if not points:
            continue
        line = axis.plot(
            [point[0] for point in points],
            [point[1]["mean"] for point in points],
            marker="o",
            markersize=3,
            label=f"empirical m={feature_count}",
        )[0]
        axis.fill_between(
            [point[0] for point in points],
            [point[1].get("ci95_low", point[1]["mean"]) for point in points],
            [point[1].get("ci95_high", point[1]["mean"]) for point in points],
            color=line.get_color(),
            alpha=0.15,
        )
        axis.plot(
            [point[0] for point in points],
            [point[2]["mean"] for point in points],
            linestyle="--",
            color=line.get_color(),
            label=f"MSE bound m={feature_count}",
        )
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.set(
        xlabel="epsilon (absolute L2 error threshold)",
        ylabel="P(||F - F_hat||2 > epsilon)",
        title="S3 empirical tail probability vs MSE/epsilon^2 bound (tau=1)",
    )
    axis.grid(True, which="both", alpha=0.25)
    if axis.lines:
        axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    save_figure(figure, "s3_tail_probability_epsilon")
    plt.close(figure)


def plot_tail_temperature(question_rows, args):
    """Show temperature ablation for every configured epsilon at m=2048."""
    subset = [row for row in question_rows if row["m"] == 2048]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    metrics = (
        ("empirical_tail_probability", "Empirical tail probability"),
        ("markov_mse_upper_bound", "MSE/epsilon^2 upper bound"),
    )
    for axis, (metric, title) in zip(axes, metrics):
        for epsilon in sorted({row["epsilon"] for row in subset}):
            points = []
            for temperature in sorted({row["tau"] for row in subset}):
                cell = [
                    row
                    for row in subset
                    if row["epsilon"] == epsilon and row["tau"] == temperature
                ]
                summary = _question_curve_point(cell, metric, args)
                if summary:
                    points.append((temperature, summary))
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1]["mean"] for point in points],
                    marker="o",
                    markersize=2,
                    label=f"epsilon={epsilon:g}",
                )
        axis.set_title(title)
        axis.set_xlabel("kernel temperature")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("probability / upper bound")
    figure.suptitle("S3 tail-probability temperature ablation (m=2048)")
    figure.tight_layout()
    save_figure(figure, "s3_tail_probability_temperature")
    plt.close(figure)


def plot_tail_by_seed(seed_rows):
    """Visualize question-balanced exceedance rates for individual seeds."""
    subset = [
        row
        for row in seed_rows
        if row["m"] == 2048 and abs(row["tau"] - 1.0) < 1e-8
    ]
    seeds = sorted({row["seed"] for row in subset})
    epsilons = sorted({row["epsilon"] for row in subset})
    if not seeds or not epsilons:
        return
    lookup = {
        (row["epsilon"], row["seed"]): (
            row["exceedance_rate"]
            if row.get("exceedance_rate") is not None
            else np.nan
        )
        for row in subset
    }
    matrix = np.asarray(
        [[lookup.get((epsilon, seed), np.nan) for seed in seeds] for epsilon in epsilons],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(max(9, len(seeds) * 0.28), 5))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(seeds)), [str(seed) for seed in seeds], rotation=90)
    axis.set_yticks(
        range(len(epsilons)), [f"{epsilon:g}" for epsilon in epsilons]
    )
    axis.set(
        xlabel="random-feature seed",
        ylabel="epsilon",
        title="S3 exceedance rate by seed (m=2048, tau=1)",
    )
    figure.colorbar(image, ax=axis, label="question-balanced exceedance rate")
    figure.tight_layout()
    save_figure(figure, "s3_tail_probability_by_seed")
    plt.close(figure)

def plot_forest(rows):
    subset = [
        row
        for row in rows
        if row["kind"] == "F"
        and row["m"] == 2048
        and abs(row["tau"] - 1.0) < 1e-6
    ]
    by_question = {}
    for row in subset:
        by_question.setdefault(row["item_id"], []).append(row["variance"])
    points = sorted(
        (item_id, float(np.mean(values)))
        for item_id, values in by_question.items()
    )[:50]
    if not points:
        return
    figure, axis = plt.subplots(figsize=(7, max(4, len(points) * 0.13)))
    axis.scatter(
        [value for _, value in points], range(len(points)), s=14, color="tab:purple"
    )
    axis.set_xscale("log")
    axis.set_yticks(range(len(points)), [str(item_id) for item_id, _ in points])
    axis.set(xlabel="Refiner F variance (m=2048, tau=1)", ylabel="question id")
    figure.tight_layout()
    save_figure(figure, "s3_refiner_state_forest")
    plt.close(figure)
