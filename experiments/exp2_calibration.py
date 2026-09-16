"""E2 -- does w mean what it says?  The v1 formulation against the corrected one.

Three separate defects are measured:
  (a) term calibration -- is each energy term O(1) at initialisation, for every
      option pair, so that w alone controls the trade-off?
  (b) effective retraction fraction -- v1 blended against a unit-norm polar
      factor, making the realised fraction depend on ||Y|| and hence on p.
  (c) monotone control -- does increasing w actually trade fidelity for
      orthogonality?
"""
import warnings

import numpy as np
import pandas as pd
import torch

from nsa_flow import energy, nsa_flow, polar_factor, project_scaled_stiefel
from nsa_flow.energy import stiefel_defect_normalised

from . import v1

warnings.filterwarnings("ignore")
F64 = torch.float64


def term_calibration():
    """(a) Each term should be O(1) at init.  v1's normalisers were derived for
    ``fidelity_basic`` and applied to whichever ``fidelity_type`` was chosen."""
    rows = []
    torch.manual_seed(0)
    for p, k in [(100, 20), (60, 6), (500, 10)]:
        X0 = torch.rand(p, k, dtype=F64)
        Y = X0 + 0.3 * torch.randn(p, k, dtype=F64)
        for w in [0.1, 0.5, 0.9]:
            # --- v1: reproduce its own calibration exactly
            s = (X0.pow(2).sum() / X0.numel()).sqrt()
            Xn, Yn = X0 / s, Y / s
            g0 = (0.5 * (Yn - Xn).pow(2).sum() / (p * k)).clamp_min(1e-8)
            d0 = v1.defect_fast(Yn).clamp_min(1e-8)
            fid_eta = min(float((1 - w) / (g0 * p * k)), 1e6)
            c_orth = min(float(4 * w / d0), 1e6)
            for ft in ["basic", "scale_invariant", "symmetric"]:
                for ot in ["basic", "scale_invariant"]:
                    E = v1.compute_energy(Yn, Xn, w=w, fidelity_type=ft, orth_type=ot,
                                          fid_eta=fid_eta, c_orth=c_orth,
                                          track_grad=False, return_dict=True)
                    rows.append(dict(version="v1", p=p, k=k, w=w,
                                     fidelity_type=ft, orth_type=ot,
                                     fid_term=float(E["fidelity"]),
                                     orth_term=float(E["orthogonality"]),
                                     fid_target=1 - w, orth_target=4 * w))
            # --- v2
            tot, f, d = energy(Y, X0, w=w, return_parts=True)
            rows.append(dict(version="v2", p=p, k=k, w=w,
                             fidelity_type="scale_invariant", orth_type="normalised_gram",
                             fid_term=float(f), orth_term=float(d),
                             fid_target=1.0, orth_target=1.0))
    df = pd.DataFrame(rows)
    df["fid_ratio"] = df.fid_term / df.fid_target
    df["orth_ratio"] = df.orth_term / df.orth_target
    return df


def effective_fraction():
    """(b) Recover the realised blend fraction from each implementation."""
    rows = []
    torch.manual_seed(0)
    for p in [20, 100, 500, 2000]:
        k = 10
        Y = torch.rand(p, k, dtype=F64)
        Y = Y / (Y.pow(2).mean().sqrt())              # v1 normalises to RMS 1
        for w in [0.25, 0.5, 0.9]:
            # v1: (1-w) Y + w * polar(Y), then rescale to ||Y||
            U = polar_factor(Y)
            A = (1 - w) * Y + w * U
            v1_frac = float((w * U).norm() / A.norm())
            v1_pred = w * (k ** 0.5) / (w * (k ** 0.5) + (1 - w) * float(Y.norm()))
            # v2: blend against the projection, whose scale matches Y
            P = project_scaled_stiefel(Y)
            eff = (1 - w) * Y + w * P
            v2_frac = float(((eff - Y) / (P - Y)).median())
            rows.append(dict(p=p, k=k, w_nominal=w, v1_effective=v1_frac,
                             v1_predicted=v1_pred, v2_effective=v2_frac))
    return pd.DataFrame(rows)


def monotone_control():
    """(c) Sweep w in both implementations and check the trade-off is monotone."""
    rows = []
    torch.manual_seed(0)
    X0 = torch.rand(100, 10, dtype=F64)
    for w in np.linspace(0.0, 1.0, 11):
        r = nsa_flow(X0, w=float(w), max_iter=20000, tol=1e-11)
        rows.append(dict(version="v2", w=float(w), fidelity=r.fidelity,
                         defect=r.raw_defect, eff_rank=r.effective_rank,
                         iters=r.iters, seconds=r.seconds))
        try:
            rv = v1.nsa_flow_orth(X0 + 0.0, X0, w=float(w), max_iter=500,
                                  initial_learning_rate=1e-3, apply_nonneg="hard",
                                  precision="float64", optimizer="adam")
            Yv = rv["Y"]
            rows.append(dict(
                version="v1", w=float(w),
                fidelity=float((Yv - rv["target"]).pow(2).sum() / rv["target"].pow(2).sum()),
                defect=float(v1.defect_fast(Yv)),
                eff_rank=float(10 / (10 * stiefel_defect_normalised(Yv) * (1 - 1 / 10) + 1)),
                iters=int(rv["final_iter"]), seconds=np.nan))
        except Exception as e:                        # record rather than hide
            rows.append(dict(version="v1", w=float(w), fidelity=np.nan,
                             defect=np.nan, eff_rank=np.nan, iters=-1,
                             seconds=np.nan, error=str(e)[:80]))
    return pd.DataFrame(rows)


def run():
    return dict(calibration=term_calibration(),
                effective_fraction=effective_fraction(),
                monotone=monotone_control())
