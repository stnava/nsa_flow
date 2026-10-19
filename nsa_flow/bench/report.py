"""Reports generated from stored rows only (no hand-copied numbers)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets.base import tier_of
from .results import load_run, resolve_run


def _f(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def verdict_table(verdicts) -> str:
    lines = ["| dataset | tier | protocol | contrast (metric, model) | estimate [95% CI] | expect | status |",
             "|---|---|---|---|---|---|---|"]
    for v in verdicts:
        contrast = f"{v['method']} − {v['ref']}" if v["ref"] else f"{v['method']} (mean vs 0)"
        mm = v["metric"] + (f", {v['model']}" if v["model"] else "")
        ci = "" if v["status"] == "SKIPPED" else f"{_f(v['estimate'])} [{_f(v['lo'])}, {_f(v['hi'])}]"
        lines.append(f"| {v['dataset']} | {v.get('tier') or tier_of(v['dataset'])} | {v['protocol']} | {contrast} ({mm}) | {ci} | "
                     f"{v['expect']} | {v['status']} |")
    return "\n".join(lines)


def health_table(rows: pd.DataFrame) -> str:
    c = rows[rows.get("cert_converged").notna()] if "cert_converged" in rows else rows.iloc[0:0]
    if c.empty:
        return "_no NSA fits recorded_"
    g = c.groupby("method").agg(fits=("cert_converged", "size"),
                                converged=("cert_converged", lambda s: float(np.mean(s.astype(bool)))),
                                max_defect_D=("cert_defect", "max"))
    lines = ["| method | NSA fits | converged | max defect_D |", "|---|---|---|---|"]
    lines += [f"| {m} | {int(r.fits)} | {r.converged:.0%} | {r.max_defect_D:.2g} |" for m, r in g.iterrows()]
    return "\n".join(lines)


def result_tables(summary: pd.DataFrame, max_rows=200) -> str:
    if summary.empty:
        return "_no results_"
    out = []
    for (ds, proto, metric, model), g in summary.groupby(["dataset", "protocol", "metric", "model"], dropna=False):
        diffs = [c for c in g.columns if c.startswith("d_vs_")]
        head = ["method", "mean ± se"] + [f"Δ vs {c[5:]} [95% CI]" for c in diffs]
        out.append(f"**{ds} · {proto} · {metric}{'' if pd.isna(model) or model == '' else ' · ' + str(model)}**\n")
        out.append("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for _, r in g.iterrows():
            cells = [r.method, f"{_f(r['mean'])} ± {_f(r['se'])}"]
            for c in diffs:
                t = c[5:]
                cells.append("" if pd.isna(r[f"ci_lo_{t}"]) else
                             f"{_f(r[c])} [{_f(r[f'ci_lo_{t}'])}, {_f(r[f'ci_hi_{t}'])}]")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out[:max_rows * 4])


def render_markdown(path_or_suite, out_dir="bench_results", brief=False) -> str:
    d = resolve_run(path_or_suite, out_dir)
    rows, man, summ = load_run(d)
    v = man.get("versions", {})
    skipped = rows[rows.get("skipped_reason").notna()] if "skipped_reason" in rows else rows.iloc[0:0]
    parts = [f"# nsa_flow.bench report: {man['suite']['name']}",
             f"profile `{man['profile']['name']}` · nsa_flow {v.get('nsa_flow')} (git {man.get('git')}) · "
             f"torch {v.get('torch')} · sklearn {v.get('sklearn')} · {man.get('started')}",
             "", "## Pre-registered primary contrasts", verdict_table(man.get("verdicts", []))]
    if len(skipped):
        parts += ["", "## Skipped (explicit)"] + [f"- {r.dataset} / {r.protocol} / {r.method}: {r.skipped_reason}"
                                                  for r in skipped.itertuples()]
    if brief:                                    # verdicts + health only (what the README embeds)
        return "\n".join(parts + ["", "Solver health:", "", health_table(rows)]) + "\n"
    parts += ["", "## Results", result_tables(summ), "", "## Solver health", health_table(rows)]
    notes = sorted({e["dataset_meta"]["annotation"] for e in man.get("entries", [])
                    if (e.get("dataset_meta") or {}).get("annotation")})
    if notes:
        parts += ["", "## Annotation notes"] + [f"- {n} (authors' annotation; review before citing)" for n in notes]
    return "\n".join(parts) + "\n"


def inject(path, name, text):
    """Replace the block between ``<!-- bench:NAME:start -->`` and ``<!-- bench:NAME:end -->`` in ``path``."""
    p = Path(path)
    s = p.read_text()
    pat = re.compile(rf"(<!-- bench:{re.escape(name)}:start -->)(.*?)(<!-- bench:{re.escape(name)}:end -->)", re.S)
    if not pat.search(s):
        raise ValueError(f"marker bench:{name} not found in {path}")
    p.write_text(pat.sub(lambda m: f"{m.group(1)}\n{text}\n{m.group(3)}", s))
