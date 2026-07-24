from .common import *

def mapping_rows(states, wo, wi, bias, k, args, include_s2, logger=None):
    fmap = []
    single = []
    for state_index, s in enumerate(states, start=1):
        if logger and (state_index == 1 or state_index == len(states) or state_index % max(1, len(states) // 10) == 0):
            logger.info("%s mapping progress: %d/%d states (%.0f%%).", "S2" if include_s2 else "S1", state_index, len(states), 100 * state_index / max(1, len(states)))
        f, p = exact(s.vector, wo, wi, bias, args.tau)
        h, valid = kernel_map(s.vector, k)
        r = base(s)
        r.update(
            f_rel_l2=float((h - f).norm() / f.norm().clamp_min(1e-8)),
            f_cosine=float(torch.nn.functional.cosine_similarity(h[None], f[None])),
            denom_valid=valid,
            nan_inf=bool(not torch.isfinite(h).all()),
            entropy=float(-(p * p.clamp_min(1e-30).log()).sum()),
            confidence=float(p.max()),
        )
        if include_s2:
            phiw = positive_features(wo, k.omega)
            phiq = positive_features(
                (s.vector.to(wo.device) / args.tau)[None], k.omega
            )[0]
            raw = (phiw @ phiq).clamp_min(0)
            alpha = (
                torch.ones_like(raw) if bias is None else torch.exp(bias - bias.max())
            )
            phat = raw * alpha
            phat /= phat.sum().clamp_min(1e-30)
            mid = (p + phat) / 2
            r.update(
                kl_p_phat=float(
                    (p * (p.clamp_min(1e-30).log() - phat.clamp_min(1e-30).log())).sum()
                ),
                js=float(
                    0.5
                    * (
                        (
                            p * (p.clamp_min(1e-30).log() - mid.clamp_min(1e-30).log())
                        ).sum()
                        + (
                            phat
                            * (phat.clamp_min(1e-30).log() - mid.clamp_min(1e-30).log())
                        ).sum()
                    )
                ),
                tv=float(0.5 * (p - phat).abs().sum()),
                l1=float((p - phat).abs().sum()),
                top1_agree=float(p.argmax() == phat.argmax()),
                top10_overlap=overlap(p, phat, 10),
                top100_overlap=overlap(p, phat, 100),
                exact_top10_mass=float(phat[torch.topk(p, 10).indices].sum()),
            )
        fmap.append(r)
        for band, i in rank_ids(p, s, args):
            x = s.vector.to(wo.device) / args.tau
            true = torch.exp(wo[i] @ x)
            est = (
                positive_features(wo[i : i + 1], k.omega)[0]
                @ positive_features(x[None], k.omega)[0]
            )
            single.append(
                {
                    **base(s),
                    "rank_band": band,
                    "kernel_abs_error": float((est - true).abs()),
                    "kernel_relative_error": float((est - true).abs() / (true + 1e-8)),
                    "kernel_log_error": float(
                        (est + 1e-8).log().sub((true + 1e-8).log()).abs()
                    ),
                    "kernel_ratio": float(est / (true + 1e-8)),
                }
            )
    return fmap, single

