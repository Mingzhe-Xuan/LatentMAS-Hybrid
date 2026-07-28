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
            for seed in range(1001, 1001 + args.s3_replicates):
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

            for state_index, state in enumerate(selected):
                exact_value = exact_values[state_index]
                stack = torch.stack(f_accumulators[state_index])
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
    return rows


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
