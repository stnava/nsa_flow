"""Generate HTML report for the Golub 3-class benchmark (exp23).

Reads paper/results/e23_golub_3class.csv and produces a styled HTML table
with key takeaways.

Usage:
    PYTHONPATH=. python experiments/golub_3class_report.py
"""
import pandas as pd
from pathlib import Path

CSV = Path("paper/results/e23_golub_3class.csv")
df  = pd.read_csv(CSV)

piv = df.pivot_table(index="method", columns="classifier",
                     values=["bal_acc","sd","sparsity","orth_defect"],
                     aggfunc="first")
piv.columns = [f"{c[0]}_{c[1]}" for c in piv.columns]
piv = piv.reset_index()

# Order methods sensibly
ORDER = ["Standard PCA", "SparsePCA (median-thr)", "SparsePCA (sklearn α=1)"]
for w in ["0.25","0.5","0.75","0.9"]:
    ORDER.append(f"NSA-Flow (w={w})")
for wv in ["0.0","0.25","0.5"]:
    for suffix in ["","+consol"]:
        ORDER.append(f"Signed{suffix} (w={wv})")
piv["_order"] = piv["method"].map({m:i for i,m in enumerate(ORDER)}).fillna(99)
piv = piv.sort_values("_order").drop(columns="_order")

def bar(val, lo=0.3, hi=1.0, color="#2166ac", width=80):
    pct = max(0.0, min(1.0, (val - lo)/(hi - lo))) * 100
    return (f'<div style="background:#eee;border-radius:3px;width:{width}px;display:inline-block">'
            f'<div style="background:{color};width:{pct:.1f}%;height:12px;border-radius:3px"></div>'
            f'</div>')

def fmt_acc(r, col):
    v = r[f"bal_acc_{col}"]
    s = r[f"sd_{col}"]
    color = "#2166ac" if col=="linear" else "#d6604d"
    return f'{bar(v,color=color)} {v:.4f} <small>±{s:.4f}</small>'

def fmt_sp(r):
    v = r.get("sparsity_linear", r.get("sparsity_forest", 0.0))
    if pd.isna(v): return "—"
    return f'{v:.3f}'

def fmt_orth(r):
    v = r.get("orth_defect_linear", r.get("orth_defect_forest", 0.0))
    if pd.isna(v): return "—"
    return f'{v:.2e}'

def method_color(m):
    if "PCA" in m and "Sparse" not in m and "median" not in m:
        return "#f0f4ff"
    if "sklearn" in m or "median" in m:
        return "#fff8f0"
    if "NSA-Flow" in m:
        return "#f0fff4"
    if "Signed" in m:
        return "#fff0f4"
    return "white"

rows_html = []
for _, r in piv.iterrows():
    bg = method_color(r["method"])
    rows_html.append(f"""
    <tr style="background:{bg}">
      <td style="padding:6px 10px;font-weight:500">{r['method']}</td>
      <td style="padding:6px 10px">{fmt_acc(r,'linear')}</td>
      <td style="padding:6px 10px">{fmt_acc(r,'forest')}</td>
      <td style="padding:6px 10px;text-align:center">{fmt_sp(r)}</td>
      <td style="padding:6px 10px;text-align:center">{fmt_orth(r)}</td>
    </tr>""")

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NSA-Flow — Golub 3-class benchmark</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:32px;color:#222;}}
  h1 {{font-size:22px;margin-bottom:4px;}}
  h2 {{font-size:16px;color:#555;font-weight:400;margin-top:0;}}
  table {{border-collapse:collapse;width:100%;max-width:900px;}}
  th {{background:#333;color:white;padding:8px 10px;text-align:left;font-size:13px;}}
  td {{border-bottom:1px solid #e8e8e8;font-size:13px;}}
  .section {{margin:24px 0 8px 0;font-size:15px;font-weight:600;color:#444;
             border-left:4px solid #2166ac;padding-left:10px;}}
  .note {{background:#fffbe6;border:1px solid #f0d060;border-radius:6px;
          padding:12px 16px;max-width:860px;margin-top:20px;font-size:13px;
          line-height:1.6;}}
  small {{color:#888;}}
</style>
</head>
<body>
<h1>NSA-Flow — Golub 3-class benchmark (E23)</h1>
<h2>B-ALL / T-ALL / AML · n=72 · p=2000 (top-var log2 genes) · k=3 · 5×50 CV</h2>

<p class="section">Balanced accuracy by method and classifier</p>
<table>
  <tr>
    <th>Method</th>
    <th>Linear classifier</th>
    <th>Random forest</th>
    <th>Sparsity</th>
    <th>Orth defect</th>
  </tr>
  {''.join(rows_html)}
</table>

<div class="note">
  <b>Key findings (v2.10.0, lobe w-scaling fix):</b><br>
  • <b>PCA</b> is the strongest linear arm (0.762 linear, 0.697 forest). On the correct
    top-2000 informative genes, PCA is hard to beat for linear classification.<br>
  • <b>Signed (w=0.0)</b> matches PCA on linear (0.750) and beats it on forest (0.708),
    because the signed lifting captures contrast structure (B-ALL up-genes vs T-ALL up-genes)
    that random forests exploit. At w=0 the solver is essentially signed PCA.<br>
  • <b>Signed+consol (w=0.0)</b> achieves 0.750 linear at 67% sparsity — within 0.012 of PCA
    with fully disjoint gene sets per component (interpretable contrast structure).<br>
  • <b>sklearn SparsePCA α=1.0</b> is over-regularised: 0.669 linear (−0.093 vs PCA).
    The α parameter must be tuned in-fold for fair comparison.<br>
  • <b>NSA-Flow data</b> (non-negative, w=0.25–0.9): linear 0.658–0.679, below PCA.
    The non-negative constraint sacrifices the linear decision boundary on this dataset.<br>
  • <b>Lobe overlap</b> is now monotone non-increasing with w (v2.10.0 fix):
    w=0 → 0.37, w=0.5 → 0.06. The pre-fix spike at w=0.05 (overlap=0.55) is gone.<br>
  • Preprocessing: log2(clip(X,1)) → top-2000 by variance → z-score.
    Bug fixed 2026-09-18: previous code standardised first (all variances→1) then
    filtered, selecting random genes rather than informative ones.
</div>
</body>
</html>"""

out = Path("brain_output_golub3class.html")
out.write_text(html, encoding="utf-8")
print(f"Written: {out.resolve()}")
