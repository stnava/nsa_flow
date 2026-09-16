"""Numerical verification of every claim in the NSA-Flow theory note.

Each check prints PASS/FAIL and the measured quantity. Run: python theory/verify.py
"""
import itertools, math
import torch

torch.set_default_dtype(torch.float64)
OK = []

def chk(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

# ---------------------------------------------------------------- definitions
def gram(Y):
    return Y.transpose(-2, -1) @ Y

def D(Y):
    """D(Y) = ||G - I/k||_F^2 with G = Y'Y/tr(Y'Y).  Scale-invariant."""
    S = gram(Y)
    t = S.diagonal(dim1=-2, dim2=-1).sum(-1)
    k = Y.shape[-1]
    return (S * S).sum((-2, -1)) / t**2 - 1.0 / k

def grad_D(Y):
    """(4/t^2) [ Y S - (N/t) Y ]"""
    S = gram(Y)
    t = S.diagonal(dim1=-2, dim2=-1).sum(-1)
    N = (S * S).sum((-2, -1))
    return (4.0 / t**2) * (Y @ S - (N / t) * Y)

def defect_old(Y):                      # the shipped functional, for contrast
    S = gram(Y)
    return ((S * S).sum() - S.diagonal().pow(2).sum()) / Y.pow(2).sum() ** 2

# ------------------------------------------------------------------ P1 bounds
print("\nP1  bounds: 0 <= D <= 1 - 1/k, upper attained iff rank 1")
torch.manual_seed(0)
for p, k in [(40, 6), (100, 20), (12, 12), (200, 3)]:
    lo, hi = math.inf, -math.inf
    for _ in range(300):
        Y = torch.randn(p, k) * torch.rand(1) * 10
        lo, hi = min(lo, D(Y).item()), max(hi, D(Y).item())
    Q = torch.linalg.qr(torch.randn(p, k))[0]
    rank1 = torch.randn(p, 1) @ torch.randn(1, k)
    chk(f"p={p},k={k} range", lo >= -1e-12 and hi <= 1 - 1/k + 1e-12, f"[{lo:.3e},{hi:.6f}] cap={1-1/k:.6f}")
    chk(f"p={p},k={k} rank-1 attains cap", abs(D(rank1).item() - (1 - 1/k)) < 1e-10,
        f"D={D(rank1).item():.12f} vs {1-1/k:.12f}")
    chk(f"p={p},k={k} orthonormal attains 0", abs(D(Q).item()) < 1e-14, f"D={D(Q).item():.3e}")

# ------------------------------------------------------------- P2 zero set
print("\nP2  zero set is exactly R_{>0} . St(p,k)")
p, k = 50, 7
Q = torch.linalg.qr(torch.randn(p, k))[0]
for c in [1e-3, 1.0, 7.3, 1e4]:
    chk(f"D(cQ)=0 for c={c:g}", abs(D(c * Q).item()) < 1e-13, f"D={D(c*Q).item():.3e}")
# orthogonal columns with UNEQUAL norms must NOT be in the zero set
scal = torch.tensor([1., 1, 1, 1, 1, 1, 8.])
chk("orthogonal + unequal norms  => D > 0", D(Q * scal).item() > 0.1,
    f"D={D(Q*scal).item():.4f}   (old defect={defect_old(Q*scal).item():.3e})")
# converse: D=0 forces S proportional to I
Yz = 3.7 * Q
S = gram(Yz); chk("D=0 => Y'Y proportional to I",
                  torch.allclose(S, S.diagonal().mean() * torch.eye(k), atol=1e-11))

# ---------------------------------------------------------- P3 invariances
print("\nP3  invariance: scale, left-O(p), right-O(k)")
Y = torch.rand(60, 8)
d0 = D(Y).item()
chk("scale  D(cY)=D(Y)", abs(D(1e5 * Y).item() - d0) < 1e-12, f"{D(1e5*Y).item():.12f} vs {d0:.12f}")
U = torch.linalg.qr(torch.randn(60, 60))[0]
chk("left   D(UY)=D(Y)", abs(D(U @ Y).item() - d0) < 1e-12)
V = torch.linalg.qr(torch.randn(8, 8))[0]
chk("right  D(YV)=D(Y)  [full O(k), same group as the constraint]",
    abs(D(Y @ V).item() - d0) < 1e-12, f"{D(Y@V).item():.12f} vs {d0:.12f}")
chk("old defect is NOT right-O(k) invariant (breaks the problem's symmetry)",
    abs(defect_old(Y @ V).item() - defect_old(Y).item()) > 1e-6,
    f"{defect_old(Y).item():.6f} -> {defect_old(Y@V).item():.6f}")

# ------------------------------------------------------- P4 spectral identity
print("\nP4  D = k.Var(lambda) = 1/EffectiveRank - 1/k")
for p, k in [(40, 6), (100, 20), (30, 30)]:
    Y = torch.rand(p, k)
    lam = torch.linalg.eigvalsh(gram(Y)); lam = lam / lam.sum()
    er = 1.0 / (lam * lam).sum()
    chk(f"p={p},k={k} identity", abs(D(Y).item() - k * lam.var(unbiased=False).item()) < 1e-12
                                 and abs(D(Y).item() - (1/er - 1/k).item()) < 1e-12,
        f"D={D(Y).item():.12f}  kVar={k*lam.var(unbiased=False).item():.12f}  1/ER-1/k={(1/er-1/k).item():.12f}")

# -------------------------------------------------------- P5 collapse floor
print("\nP5  rank r < k  =>  D >= 1/r - 1/k  (old defect has NO such floor)")
p, k = 60, 6
for r in [1, 2, 3, 4, 5]:
    worst = math.inf
    for _ in range(400):
        Y = torch.randn(p, r) @ torch.randn(r, k)
        worst = min(worst, D(Y).item())
    chk(f"rank {r} floor {1/r-1/k:.6f}", worst >= 1/r - 1/k - 1e-10, f"observed min D={worst:.6f}")
# old defect attains 0 at rank k-1
Qd = torch.linalg.qr(torch.randn(p, k))[0].clone(); Qd[:, -1] = 0
chk("old defect = 0 at rank k-1 (degenerate minimiser)", abs(defect_old(Qd).item()) < 1e-14,
    f"old={defect_old(Qd).item():.3e}   new D={D(Qd).item():.6f} >= {1/(k-1)-1/k:.6f}")

# ------------------------------------------------ P6 gradient + Euler + bound
print("\nP6  gradient: matches autograd, orthogonal to Y, bounded by 8/||Y||")
for p, k in [(40, 6), (100, 20), (9, 9)]:
    Y = torch.rand(p, k).requires_grad_(True)
    D(Y).backward()
    g_auto, g_closed = Y.grad, grad_D(Y.detach())
    rel = (g_auto - g_closed).norm() / g_auto.norm()
    chk(f"p={p},k={k} closed form vs autograd", rel < 1e-11, f"rel err={rel:.3e}")
    Yd = Y.detach()
    chk(f"p={p},k={k} <grad,Y>=0  (D cannot change scale)",
        abs((g_closed * Yd).sum().item()) / (g_closed.norm() * Yd.norm()).item() < 1e-12)
    chk(f"p={p},k={k} ||grad|| <= 8/||Y||",
        g_closed.norm().item() <= 8.0 / Yd.norm().item() + 1e-12,
        f"{g_closed.norm().item():.4e} <= {8/Yd.norm().item():.4e}")

# ------------------------------------------- P7 disjoint supports at optimum
print("\nP7  Y>=0 and Y'Y diagonal  <=>  pairwise disjoint supports")
torch.manual_seed(1)
bad = 0
for _ in range(2000):
    p, k = 8, 3
    Y = (torch.rand(p, k) * (torch.rand(p, k) < 0.4)).clamp_min(0)
    S = gram(Y); offdiag = (S - torch.diag(S.diagonal())).abs().max().item()
    supp = [set((Y[:, i] > 0).nonzero().flatten().tolist()) for i in range(k)]
    disj = all(not (supp[i] & supp[j]) for i, j in itertools.combinations(range(k), 2))
    if (offdiag < 1e-14) != disj:
        bad += 1
chk("equivalence over 2000 random sparse nonneg matrices", bad == 0, f"counterexamples={bad}")

# --------------------------------------------------- P8 k > p handled honestly
print("\nP8  k > p: inf D = 1/p - 1/k > 0 (orthonormal columns impossible, reported not hidden)")
for p, k in [(20, 50), (120, 200)]:
    best = math.inf
    for _ in range(200):
        A = torch.randn(p, k); U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        best = min(best, D(U @ Vh).item())            # row-orthonormal = best possible
    chk(f"p={p},k={k} floor {1/p-1/k:.6e}", abs(best - (1/p - 1/k)) < 1e-10, f"attained {best:.6e}")

# ------------------------------------------------------------------- summary
print(f"\n{'='*66}\n{sum(OK)}/{len(OK)} checks passed\n{'='*66}")
raise SystemExit(0 if all(OK) else 1)
