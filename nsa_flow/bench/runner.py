"""Run a suite: availability, cost estimate, protocols, summaries and pre-registered verdicts."""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import methods, protocols, suites  # noqa: F401  (import registers methods/protocols/suites)
from .datasets import DatasetUnavailable, get_dataset, tier_of
from .metrics import long_summary
from .registry import METHODS, PROTOCOLS
from .results import manifest, save_run
from .runtime import get_profile, probe_fit_seconds


def estimate_minutes(entry, ds, profile, jobs) -> float:
    """Minutes for one entry from measured single-fit times (one probe per method), not guessed constants."""
    if ds is None or entry.protocol == "speed":
        return 0.0
    if entry.protocol == "factor_analysis":
        import time as _t
        from .registry import FA_METHODS
        k = int(ds.meta.get("n_factors", 5))
        total = 0.0
        for name in entry.methods:
            m = FA_METHODS[name]
            t0 = _t.time()
            m.fit(ds.X, k, m.default, 0)
            fits = profile.n_outer * (3 + 3 * max(len(m.grid(profile)), 1) * profile.n_inner)
            total += (_t.time() - t0) * fits
        return total / max(jobs, 1) / 60.0
    from .protocols.nested_cv import outer_splits
    if entry.protocol == "nested_cv":
        _, tr, te = next(outer_splits(ds, 1))
    else:
        tr = np.arange(min(ds.n, ds.draw_n or ds.n // 2))
    rep, _ = ds.prep(tr, tr[:1])
    total = 0.0
    for name in entry.methods:
        m = METHODS[name]
        if not m.applies(ds)[0]:
            continue
        if entry.protocol == "nested_cv":
            grid = m.grid(profile, ds.p)
            fits = profile.n_outer * (profile.n_inner * len(grid) + 3)
            hp = grid[-1]
        else:
            from .protocols.module_recovery import variants
            fits = profile.n_outer * 4 * len(variants(m))
            hp = {**m.module_cells[0], "k": 5}
        total += probe_fit_seconds(m, rep, hp, 0, {"spherical": ds.kmeans_spherical}) * fits
    return total / max(jobs, 1) / 60.0


def evaluate_verdicts(suite, summary, only=None, skipped_entries=()):
    out = []
    for e in suite.entries:
        p = e.primary
        if p is None:
            continue
        base = dict(dataset=e.dataset, protocol=e.protocol, metric=p.metric, model=p.model, method=p.method,
                    ref=p.ref, expect=e.expect, tier=tier_of(e.dataset))
        if e.dataset in skipped_entries:
            out.append(dict(base, status="SKIPPED", estimate=np.nan, lo=np.nan, hi=np.nan))
            continue
        if summary is None or summary.empty:
            continue
        s = summary
        sel = s[(s.dataset == e.dataset) & (s.protocol == e.protocol) & (s.metric == p.metric)
                & (s.model.fillna("") == p.model) & (s.method == p.method)]
        if sel.empty:
            continue
        r = sel.iloc[0]
        if p.ref:
            tag = p.ref.split()[0]
            est, lo, hi = r[f"d_vs_{tag}"], r[f"ci_lo_{tag}"], r[f"ci_hi_{tag}"]
            if e.expect == "effect":
                ok = (lo > 0) if p.direction == "gt" else (hi < 0)
                status = "PASS" if ok else ("WORSE" if ((hi < 0) if p.direction == "gt" else (lo > 0)) else "FAIL")
            elif e.expect == "null":
                status = "PASS" if (lo <= 0 <= hi) else "FAIL"
            else:
                status = "INFO"
            guards = []
            for gm, gdir, gtol in p.guards:
                gs = s[(s.dataset == e.dataset) & (s.protocol == e.protocol) & (s.metric == gm)
                       & (s.model.fillna("") == p.model) & (s.method == p.method)]
                if gs.empty or pd.isna(gs.iloc[0][f"ci_lo_{tag}"]):
                    guards.append(dict(metric=gm, direction=gdir, tol=gtol, estimate=np.nan, lo=np.nan, hi=np.nan,
                                       ok=None))          # not evaluable (e.g. CFA skipped on a small test split)
                    continue
                g = gs.iloc[0]
                glo, ghi = float(g[f"ci_lo_{tag}"]), float(g[f"ci_hi_{tag}"])
                guards.append(dict(metric=gm, direction=gdir, tol=gtol, estimate=float(g[f"d_vs_{tag}"]), lo=glo,
                                   hi=ghi, ok=bool(ghi < gtol) if gdir == "lt" else bool(glo > -gtol)))
            if guards:
                base = dict(base, guards=guards)
                if status == "PASS" and any(g["ok"] is False for g in guards):
                    status = "FAIL (guard)"
        else:
            est, lo, hi = r["mean"], r["mean"] - 1.96 * r["se"], r["mean"] + 1.96 * r["se"]
            status = ("PASS" if abs(est) < p.tol else "FAIL") if e.expect == "null" else "INFO"
        out.append(dict(base, estimate=float(est), lo=float(lo), hi=float(hi), status=status))
    return out


def run_suite(suite, profile="standard", jobs=1, out="bench_results", only=None, budget_min=None,
              command="", save=True, log=None, public_only=False):
    import os
    log = log or (lambda *a: print(*a, flush=True))     # flushed: redirected stdout is block-buffered
    os.environ.setdefault("PYTHONWARNINGS", "ignore")           # inherited by loky workers
    profile = get_profile(profile) if isinstance(profile, str) else profile
    frames, infos, skipped = [], [], []
    for e in suite.entries:
        if only and not (e.dataset in only or e.dataset.split("@")[0] in only):
            continue
        info = dict(dataset=e.dataset, protocol=e.protocol, expect=e.expect, tier=tier_of(e.dataset),
                    primary=None if e.primary is None else asdict(e.primary), methods=e.methods)
        if public_only and info["tier"] != "public":
            why = f"{e.dataset} is tier {info['tier']!r}; --public-only runs public datasets only"
            log(f"[skip] {e.dataset} / {e.protocol}: {why}")
            frames.append(pd.DataFrame([dict(dataset=e.dataset, protocol=e.protocol, method="*",
                                             skipped_reason=why)]))
            infos.append(dict(info, status="skipped", reason=why))
            skipped.append(e.dataset)
            continue
        try:
            ds = None if e.protocol == "speed" else get_dataset(e.dataset)
        except DatasetUnavailable as ex:
            log(f"[skip] {e.dataset} / {e.protocol}: {ex}")
            frames.append(pd.DataFrame([dict(dataset=e.dataset, protocol=e.protocol, method="*",
                                             skipped_reason=str(ex))]))
            infos.append(dict(info, status="skipped", reason=str(ex)))
            skipped.append(e.dataset)
            continue
        est = estimate_minutes(e, ds, profile, jobs)
        log(f"[run ] {e.dataset} / {e.protocol}  est {est:.1f} min  "
            f"(n={getattr(ds, 'n', '-')}, p={getattr(ds, 'p', '-')}, profile={profile.name}, jobs={jobs})")
        if budget_min is not None and est > budget_min:
            why = f"estimated {est:.1f} min > budget {budget_min} min"
            log(f"[skip] {e.dataset} / {e.protocol}: {why}")
            frames.append(pd.DataFrame([dict(dataset=e.dataset, protocol=e.protocol, method="*",
                                             skipped_reason=why)]))
            infos.append(dict(info, status="skipped", reason=why))
            skipped.append(e.dataset)
            continue
        t0 = time.time()
        df = PROTOCOLS[e.protocol](ds, e.methods, profile, jobs=jobs, **e.params)
        df["suite"] = suite.name
        frames.append(df)
        infos.append(dict(info, status="ran", seconds=round(time.time() - t0, 1), estimate_min=round(est, 2),
                          fingerprint=None if ds is None else ds.fingerprint(),
                          n=getattr(ds, "n", None), p=getattr(ds, "p", None),
                          dataset_meta={k: v for k, v in (ds.meta if ds else {}).items()
                                        if k in ("annotation", "null", "source", "outcome", "groups")}))
        log(f"       done in {time.time() - t0:.0f}s")
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    refs = sorted({e.primary.ref for e in suite.entries if e.primary and e.primary.ref} | {"PCA", "KMeans features", "NMF"})
    valid = rows[rows.get("value").notna()] if "value" in rows else rows.iloc[0:0]
    summary = long_summary(valid, refs) if len(valid) else pd.DataFrame()
    verdicts = evaluate_verdicts(suite, summary, skipped_entries=set(skipped))
    run_dir = None
    if save:
        man = manifest(suite, profile, jobs, infos, command)
        run_dir = save_run(out, suite.name, rows, man, summary, verdicts)
        log(f"saved {run_dir}")
    return rows, summary, verdicts, run_dir
