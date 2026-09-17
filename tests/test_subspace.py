"""Properties of the sign-blind subspace fidelity."""
import pytest
import torch

from nsa_flow import (SubspaceAnchor, grad_subspace_fidelity, negative_mass,
                      nsa_flow, subspace_fidelity)

F64 = torch.float64


def _orth(p, k):
    return torch.linalg.qr(torch.randn(p, k, dtype=F64))[0]


def test_invariant_to_any_invertible_reparametrisation_of_the_target():
    """The point of the construction: P depends only on range(X0).

    Column signs and the rotation of a PCA basis are conventions, not
    information, so a fidelity that charges for them is charging for nothing.
    """
    torch.manual_seed(0)
    X0 = torch.randn(40, 5, dtype=F64)
    Y = torch.rand(40, 5, dtype=F64)
    base = float(subspace_fidelity(Y, SubspaceAnchor(X0)))
    D = torch.diag(torch.tensor([1.0, -1, 1, -1, 1], dtype=F64))
    for T in (D, _orth(5, 5), torch.randn(5, 5, dtype=F64)):
        assert abs(float(subspace_fidelity(Y, SubspaceAnchor(X0 @ T))) - base) < 1e-10


def test_endpoints_are_zero_and_one():
    torch.manual_seed(1)
    X0 = torch.randn(30, 4, dtype=F64)
    a = SubspaceAnchor(X0)
    inside = X0 @ torch.rand(4, 4, dtype=F64)
    assert float(subspace_fidelity(inside, a)) < 1e-18
    U, _, _ = torch.linalg.svd(X0, full_matrices=True)
    outside = U[:, 4:8] @ torch.rand(4, 4, dtype=F64)
    assert abs(float(subspace_fidelity(outside, a)) - 1.0) < 1e-10
    for _ in range(20):
        v = float(subspace_fidelity(torch.randn(30, 4, dtype=F64), a))
        assert -1e-12 <= v <= 1.0 + 1e-12


def test_gradient_matches_autograd_and_cannot_change_the_norm():
    torch.manual_seed(2)
    X0 = torch.randn(25, 4, dtype=F64)
    Y = torch.rand(25, 4, dtype=F64, requires_grad=True)
    a = SubspaceAnchor(X0)
    subspace_fidelity(Y, a).backward()
    g = grad_subspace_fidelity(Y.detach(), a)
    assert (Y.grad - g).abs().max() < 1e-10
    # degree-0 homogeneous, so <grad, Y> = 0
    assert abs(float((g * Y.detach()).sum())) < 1e-10


def test_negative_mass_measures_what_is_unreachable():
    X = torch.rand(20, 5, dtype=F64)
    assert float(negative_mass(X)) == 0.0
    assert float(negative_mass(-X)) == pytest.approx(1.0)
    assert 0.3 < float(negative_mass(torch.randn(200, 5, dtype=F64))) < 0.9


def test_auto_selects_sign_blind_fidelity_for_a_signed_target_and_says_so():
    torch.manual_seed(3)
    signed = torch.randn(60, 5, dtype=F64)
    with pytest.warns(RuntimeWarning, match="negative mass"):
        r = nsa_flow(signed, w=0.5)
    assert r["fidelity_mode"] == "subspace"
    assert r["fidelity_requested"] == "auto"
    assert r["target_negative_mass"] > 0.3
    assert (r.Y >= 0).all()


def test_auto_keeps_the_entrywise_fidelity_for_a_nonnegative_target():
    torch.manual_seed(4)
    r = nsa_flow(torch.rand(60, 5, dtype=F64), w=0.5)
    assert r["fidelity_mode"] == "anchor"
    assert r["target_negative_mass"] == 0.0


def test_clamp_distance_exposes_a_run_that_only_clamped():
    """The anchored fidelity on a signed target degenerates toward max(0, X0).

    Without this diagnostic the caller cannot tell a refinement from a clamp.
    """
    torch.manual_seed(5)
    signed = torch.randn(60, 5, dtype=F64)
    anchored = nsa_flow(signed, w=0.5, fidelity="anchor")
    with pytest.warns(RuntimeWarning):
        blind = nsa_flow(signed, w=0.5)
    assert anchored["clamp_distance"] < blind["clamp_distance"]


def test_explicit_mode_is_respected_and_bad_mode_raises():
    X = torch.randn(30, 4, dtype=F64)
    assert nsa_flow(X, w=0.5, fidelity="anchor")["fidelity_mode"] == "anchor"
    assert nsa_flow(X, w=0.5, fidelity="subspace")["fidelity_mode"] == "subspace"
    with pytest.raises(ValueError):
        nsa_flow(X, w=0.5, fidelity="gram")


def test_subspace_mode_fixes_the_scale_gauge():
    """Both terms are degree-0 homogeneous, so the scale must be set explicitly."""
    torch.manual_seed(6)
    X0 = torch.randn(50, 5, dtype=F64)
    r = nsa_flow(X0, w=0.5, fidelity="subspace")
    assert abs(r["scale_ratio"] - 1.0) < 1e-9


def test_degenerate_target_gives_a_usable_error_not_a_linalg_crash():
    """An all-zero target spans nothing, so the projector is undefined."""
    with pytest.raises(ValueError, match="spans no subspace"):
        SubspaceAnchor(torch.zeros(20, 4, dtype=F64))


def test_rank_deficient_target_is_flagged_rather_than_silently_regularised():
    X0 = torch.randn(20, 3, dtype=F64)
    X0 = torch.cat([X0, X0[:, :1]], dim=1)          # 4 columns, rank 3
    assert SubspaceAnchor(X0).rank_deficient is True
    assert SubspaceAnchor(torch.randn(20, 4, dtype=F64)).rank_deficient is False


def test_auto_does_not_switch_at_w_zero():
    """At w=0 there is no orthogonality term, and the subspace fidelity alone is
    indifferent to rank: any non-negative matrix inside range(X0) is optimal.
    The entrywise fidelity is well posed there, so auto must keep it."""
    torch.manual_seed(7)
    signed = torch.randn(80, 8, dtype=F64)
    r = nsa_flow(signed, w=0.0, max_iter=5000, tol=1e-14)
    assert r["fidelity_mode"] == "anchor"
    assert torch.allclose(r.Y, signed.clamp_min(0), atol=1e-9)


def test_explicit_subspace_at_w_zero_warns_and_can_collapse():
    torch.manual_seed(8)
    signed = torch.randn(60, 5, dtype=F64)
    with pytest.warns(RuntimeWarning, match="degenerate"):
        r = nsa_flow(signed, w=0.0, fidelity="subspace")
    assert r["fidelity_mode"] == "subspace"
