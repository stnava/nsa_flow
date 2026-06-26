# debug_nsa_energy.py
import torch
import time
import math
import nsa_flow   # your package; must expose nsa_flow_autograd, nsa_flow_retract_auto, defect_fast

torch.set_default_dtype(torch.float64)

def compute_analytic_rgrad(Y, X0, w, simplified=False):
    # replicate compute_ortho_terms and rgrad assembly from your R->Py code
    def defect_fast_local(V):
        norm2 = torch.sum(V ** 2)
        if norm2 <= 1e-12:
            return torch.tensor(0.0, device=V.device)
        S = V.T @ V
        diagS = torch.diag(S)
        off_f2 = torch.sum(S * S) - torch.sum(diagS ** 2)
        return off_f2 / (norm2 ** 2)

    def compute_ortho_terms(Y, c_orth=1.0, simplified=False):
        norm2 = torch.sum(Y ** 2)
        if norm2 <= 1e-12 or c_orth <= 0:
            return torch.zeros_like(Y), torch.tensor(0.0, device=Y.device), norm2
        S = Y.T @ Y
        diagS = torch.diag(S)
        off_f2 = torch.sum(S * S) - torch.sum(diagS ** 2)
        defect = off_f2 / (norm2 ** 2)
        if simplified:
            grad_orth = -2 * c_orth * (Y @ (S - torch.diag(torch.diag(S))))
        else:
            Y_S = Y @ S
            Y_diag_scale = Y * diagS
            term1 = (Y_S - Y_diag_scale) / (norm2 ** 2)
            term2 = (defect / norm2) * Y
            grad_orth = c_orth * (term1 - term2)
        return grad_orth, defect, norm2

    # choose fid_eta and c_orth consistent with your nsa_flow_autograd default heuristic:
    p, k = Y.shape
    g0 = 0.5 * torch.sum((Y - X0) ** 2) / (p * k)
    g0 = torch.clamp(g0, min=1e-8)
    d0 = torch.clamp(defect_fast_local(Y), min=1e-8)
    fid_eta = (1 - w) / (g0 * p * k)
    c_orth = 4 * w / d0

    grad_fid = -fid_eta * (Y - X0)
    grad_orth, defect_val, _ = compute_ortho_terms(Y, c_orth, simplified=simplified)
    symm = lambda A: 0.5 * (A + A.T)
    if c_orth > 0:
        sym_term_orth = symm(Y.T @ grad_orth)
        rgrad_orth = grad_orth - Y @ sym_term_orth
    else:
        rgrad_orth = grad_orth
    rgrad = grad_fid + rgrad_orth
    return rgrad, grad_fid, rgrad_orth, fid_eta, c_orth

def energy_components(Y, X0, w, retraction):
    # returns fidelity, orthogonality, total (as used in your autograd energy)
    Vp = nsa_flow.nsa_flow_retract_auto(Y, w, retraction)      # call signature in your code
    # report whether retraction preserves requires_grad
    retd_req_grad = getattr(Vp, "requires_grad", None)
    if getattr(Vp, "requires_grad", None) is not None:
        req_grad_flag = bool(Vp.requires_grad)
    else:
        req_grad_flag = "no-attr"
    Vp_clamped = torch.clamp(Vp, min=0.0)  # autograd version clamps
    fid = 0.5 * torch.sum((Vp_clamped - X0) ** 2)
    orth = nsa_flow.defect_fast(Vp_clamped)
    total = fid + 0.25 * w * orth
    return {"Vp_requires_grad": req_grad_flag, "fid": fid.item(), "orth": orth.item(), "total": total.item(), "Vp": Vp, "Vp_clamped": Vp_clamped}

def debug_single_run():
    torch.manual_seed(0)
    p, k = 12, 5
    Y0 = torch.randn(p, k)
    X0 = Y0 + 0.05 * torch.randn_like(Y0)
    w = 0.5
    retraction = "soft_polar"

    # prepare autograd variable
    Y = Y0.clone().detach().to(Y0.device).requires_grad_(True)

    # create optimizer (LARS via get_torch_optimizer or replace by Adam for testing)
    try:
        opt = nsa_flow.get_torch_optimizer("lars", [Y], lr=1e-4)
        opt_name = "LARS"
    except Exception as e:
        print("LARS unavailable, falling back to Adam:", e)
        opt = torch.optim.Adam([Y], lr=1e-4)
        opt_name = "Adam"

    print("Optimizer:", opt_name)

    # run a few iterations and print diagnostics
    iters = 10
    w=0.5
    energies = []
    for it in range(1, iters + 1):
        # compute energy and autograd grad
        opt.zero_grad(set_to_none=True)
        # compute energy using your function
        E = nsa_flow.nsa_flow_autograd.__globals__['nsa_flow_energy'](Y) if 'nsa_flow_energy' in nsa_flow.nsa_flow_autograd.__globals__ else None
        # safer: call the same energy path used in your implementation:
        # We'll recompute closure here similar to your code
        def closure_energy():
            Vp = nsa_flow.nsa_flow_retract_auto(Y, 0.5, 'soft_polar')
            Vp = torch.clamp(Vp, min=0.0)
            fid = 0.5 * torch.sum((Vp - X0) ** 2)
            orth = nsa_flow.defect_fast(Vp)
            return fid + 0.25 * w * orth

        E = closure_energy()
        E.backward()
        # grab autograd gradient
        grad_autograd = Y.grad.clone()
        grad_norm = torch.norm(grad_autograd).item()

        # analytic rgrad
        rgrad, grad_fid, rgrad_orth, fid_eta, c_orth = compute_analytic_rgrad(Y.detach(), X0, w)

        # cos similarity between autograd grad and analytic rgrad
        cos = None
        try:
            cos = torch.dot(grad_autograd.flatten(), rgrad.flatten()).item() / (torch.norm(grad_autograd).item() * torch.norm(rgrad).item() + 1e-16)
        except Exception:
            cos = float('nan')

        # print energy components computed from current Y
        comps = energy_components(Y, X0, w, retraction)

        print(f"Iter {it:2d}: total={comps['total']:.6e}, fid={comps['fid']:.6e}, orth={comps['orth']:.6e}, Vp_req_grad={comps['Vp_requires_grad']}")
        print(f"         autograd_grad_norm={grad_norm:.3e}, analytic_rgrad_norm={torch.norm(rgrad).item():.3e}, cos(autograd,analytic)={cos:.4f}")

        # take optimizer step
        if isinstance(opt, torch.optim.LBFGS):
            opt.step(lambda : closure_energy())
        else:
            # clip grad and step
            torch.nn.utils.clip_grad_norm_([Y], max_norm=5.0)
            opt.step()

        # retraction after step (if your code does that)
        with torch.no_grad():
            Y.data = nsa_flow.nsa_flow_retract_auto(Y.data, w, retraction)
            Y.data = torch.clamp(Y.data, min=0.0)

        # zero out grad for next iter
        Y.grad = None

    # Final energies print
    final = energy_components(Y, X0, w, retraction)
    print("Final:", final)

if __name__ == "__main__":
    debug_single_run()
