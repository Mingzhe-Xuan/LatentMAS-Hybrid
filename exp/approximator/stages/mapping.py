from .common import *


def mapping_rows(states, wo, wi, bias, kernel, args, include_s2, logger=None):
    mapping = []
    single_kernel = []
    for state_index, state in enumerate(states, start=1):
        if logger and (
            state_index == 1
            or state_index == len(states)
            or state_index % max(1, len(states) // 10) == 0
        ):
            logger.info(
                "%s refiner_to_judger progress: %d/%d states.",
                "S2" if include_s2 else "S1",
                state_index,
                len(states),
            )
        exact_value, probabilities = exact(
            state.vector, wo, wi, bias, args.kernel_temperature
        )
        approximate, valid = kernel_map(state.vector, kernel)
        row = base(state)
        row.update(
            f_rel_l2=float(
                (approximate - exact_value).norm()
                / exact_value.norm().clamp_min(1e-8)
            ),
            f_cosine=float(
                torch.nn.functional.cosine_similarity(
                    approximate[None], exact_value[None]
                )
            ),
            denom_valid=valid,
            nan_inf=bool(not torch.isfinite(approximate).all()),
            entropy=float(
                -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
            ),
            confidence=float(probabilities.max()),
        )
        if include_s2:
            word_features = positive_features(wo, kernel.omega)
            query_features = positive_features(
                (state.vector.to(wo.device) / args.kernel_temperature)[None], kernel.omega
            )[0]
            raw = (word_features @ query_features).clamp_min(0)
            alpha = (
                torch.ones_like(raw)
                if bias is None
                else torch.exp(bias - bias.max())
            )
            approximate_probabilities = raw * alpha
            approximate_probabilities /= approximate_probabilities.sum().clamp_min(
                1e-30
            )
            midpoint = (probabilities + approximate_probabilities) / 2
            row.update(
                kl_p_phat=float(
                    (
                        probabilities
                        * (
                            probabilities.clamp_min(1e-30).log()
                            - approximate_probabilities.clamp_min(1e-30).log()
                        )
                    ).sum()
                ),
                js=float(
                    0.5
                    * (
                        (
                            probabilities
                            * (
                                probabilities.clamp_min(1e-30).log()
                                - midpoint.clamp_min(1e-30).log()
                            )
                        ).sum()
                        + (
                            approximate_probabilities
                            * (
                                approximate_probabilities.clamp_min(1e-30).log()
                                - midpoint.clamp_min(1e-30).log()
                            )
                        ).sum()
                    )
                ),
                tv=float(0.5 * (probabilities - approximate_probabilities).abs().sum()),
                l1=float((probabilities - approximate_probabilities).abs().sum()),
                top1_agree=float(
                    probabilities.argmax() == approximate_probabilities.argmax()
                ),
                top10_overlap=overlap(
                    probabilities, approximate_probabilities, 10
                ),
                top100_overlap=overlap(
                    probabilities, approximate_probabilities, 100
                ),
                exact_top10_mass=float(
                    approximate_probabilities[
                        torch.topk(probabilities, min(10, len(probabilities))).indices
                    ].sum()
                ),
            )
        mapping.append(row)
        for band, index in rank_ids(probabilities, state, args):
            query = state.vector.to(wo.device) / args.kernel_temperature
            truth = torch.exp(wo[index] @ query)
            estimate = (
                positive_features(wo[index : index + 1], kernel.omega)[0]
                @ positive_features(query[None], kernel.omega)[0]
            )
            single_kernel.append(
                {
                    **base(state),
                    "rank_band": band,
                    "kernel_abs_error": float((estimate - truth).abs()),
                    "kernel_relative_error": float(
                        (estimate - truth).abs() / (truth + 1e-8)
                    ),
                    "kernel_log_error": float(
                        (estimate + 1e-8)
                        .log()
                        .sub((truth + 1e-8).log())
                        .abs()
                    ),
                    "kernel_ratio": float(estimate / (truth + 1e-8)),
                }
            )
    return mapping, single_kernel
