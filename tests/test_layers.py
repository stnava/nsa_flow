"""Layer behaviour, including the three v1 design defects that are now rejected."""
import pytest
import torch
import torch.nn as nn

from nsa_flow import NSAFlowConv2d, NSAFlowLayer, NSAFlowLinear, stiefel_defect_normalised

F64 = torch.float64


# ------------------------------------------------------------------- Linear
def test_linear_shape_and_state_dict_compatible_with_nn_linear():
    layer = NSAFlowLinear(20, 5, w=0.5)
    x = torch.randn(4, 20)
    assert layer(x).shape == (4, 5)
    ref = nn.Linear(20, 5)
    ref.load_state_dict(layer.state_dict())          # interchangeable layout
    assert ref.weight.shape == layer.weight.shape


@pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 1.0])
def test_linear_defect_decreases_with_w(w):
    torch.manual_seed(0)
    layer = NSAFlowLinear(30, 6, w=w)
    with torch.no_grad():
        layer.weight.add_(0.5 * torch.randn_like(layer.weight))
    d = layer.defect().item()
    assert 0.0 <= d <= 1.0 + 1e-12
    if w == 1.0:
        assert d < 1e-10                             # fully projected


def test_linear_w_is_a_true_blend_fraction_independent_of_scale():
    """w must be the actual blend fraction, whatever the scale of W and whatever p.

    v1 blended against a *unit-norm* polar factor and renormalised, giving an
    effective fraction w*sqrt(k) / (w*sqrt(k) + (1-w)*||W||) that drifted with
    ||W|| and with p -- a nominal w=0.5 meant 18% at p=20 and 4% at p=500.
    Blending against the projection P(W), whose scale is matched to W, fixes this.
    """
    from nsa_flow import project_scaled_stiefel

    torch.manual_seed(0)
    for p in [20, 200, 500]:
        for scale in [1e-2, 1.0, 1e2]:
            layer = NSAFlowLinear(p, 10, w=0.5, bias=False)
            with torch.no_grad():                        # move off the manifold
                layer.weight.copy_(scale * torch.rand(10, p))
            W = layer.weight.detach().transpose(0, 1)
            P = project_scaled_stiefel(W)
            eff = layer.effective_weight().detach().transpose(0, 1)
            assert (P - W).norm() > 0, "test precondition: W must be off-manifold"
            # Compare against the predicted blend directly.  An entrywise ratio
            # (eff - W)/(P - W) would divide by near-zero entries and measure
            # nothing but round-off.
            predicted = 0.5 * W + 0.5 * P
            rel = (eff - predicted).norm().item() / predicted.norm().item()
            assert rel < 1e-6, (p, scale, rel)

        # and v1's effective fraction really did drift, for contrast
        W = scale * torch.rand(p, 10, dtype=torch.float64)
        v1_frac = 0.5 * (10 ** 0.5) / (0.5 * (10 ** 0.5) + 0.5 * W.norm().item())
        assert v1_frac < 0.5


def test_linear_gradients_flow():
    layer = NSAFlowLinear(10, 3, w=0.5)
    x = torch.randn(2, 10, requires_grad=True)
    layer(x).sum().backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert x.grad is not None


def test_defect_penalty_route_is_differentiable():
    """The preferred usage: add w * layer.defect() to the task loss."""
    layer = NSAFlowLinear(20, 5, w=0.0)
    x, y = torch.randn(8, 20), torch.randn(8, 5)
    loss = (layer(x) - y).pow(2).mean() + 0.3 * layer.defect()
    loss.backward()
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.abs().sum() > 0


def test_defect_penalty_actually_reduces_defect_when_trained():
    torch.manual_seed(0)
    layer = NSAFlowLinear(30, 6, w=0.0, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.rand(6, 30))        # far from orthogonal
    before = layer.defect().item()
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        layer.defect().backward()
        opt.step()
    assert layer.defect().item() < before * 0.05


# --------------------------------------------------------------------- Conv2d
def test_conv2d_shape_and_defect():
    layer = NSAFlowConv2d(3, 8, 3, w=0.5)
    x = torch.randn(2, 3, 16, 16)
    assert layer(x).shape == (2, 8, 14, 14)
    assert 0.0 <= layer.defect().item() <= 1.0 + 1e-12


def test_conv2d_roundtrip_view_is_exact():
    """[out,in,kh,kw] -> [p,k] -> back must be the identity."""
    layer = NSAFlowConv2d(3, 8, 3, w=0.0)
    W = layer.weight.detach()
    M = NSAFlowConv2d._pk_view(W)
    assert M.shape == (3 * 3 * 3, 8)
    assert torch.equal(NSAFlowConv2d._pk_unview(M, W), W)


def test_conv2d_gradients_flow_and_padding_modes_work():
    for mode in ["zeros", "reflect"]:
        layer = NSAFlowConv2d(3, 5, 3, padding=1, padding_mode=mode, w=0.5)
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        layer(x).sum().backward()
        assert torch.isfinite(layer.weight.grad).all() and x.grad is not None


def test_conv2d_w_one_gives_orthogonal_filters():
    layer = NSAFlowConv2d(4, 9, 3, w=1.0)
    assert layer.defect().item() < 1e-10


# ---------------------------------------------------------------- NSAFlowLayer
def test_layer_is_per_sample_not_batch_coupled():
    """v1 coupled every sample to its batch-mates; outputs must not depend on
    which other samples are present."""
    layer = NSAFlowLayer(w=0.5)
    torch.manual_seed(0)
    Y = torch.rand(6, 20, 4)
    full = layer(Y)
    for i in range(6):
        alone = layer(Y[i : i + 1])
        assert torch.allclose(alone[0], full[i], atol=1e-10)
    # and a different batch composition leaves each sample unchanged
    assert torch.allclose(layer(Y[[3, 1]])[0], full[3], atol=1e-10)


def test_layer_rejects_ambiguous_2d_input():
    with pytest.raises(ValueError, match=r"\[B, p, k\]"):
        NSAFlowLayer(w=0.5)(torch.rand(20, 4))


def test_layer_reduces_defect_monotonically_in_depth():
    torch.manual_seed(0)
    Y = torch.rand(1, 40, 6)
    layers = nn.ModuleList([NSAFlowLayer(w=0.5) for _ in range(5)])
    d = [stiefel_defect_normalised(Y[0]).item()]
    for layer in layers:
        Y = layer(Y)
        d.append(stiefel_defect_normalised(Y[0]).item())
    assert all(d[i + 1] <= d[i] + 1e-12 for i in range(len(d) - 1)), d


# ------------------------------------------------------------- nonneg handling
@pytest.mark.parametrize("cls,args", [(NSAFlowLinear, (10, 4)), (NSAFlowConv2d, (3, 4, 3))])
def test_nonneg_modes(cls, args):
    for mode in [None, "none", "softplus", "hard"]:
        layer = cls(*args, w=0.5, nonneg=mode)
        W = layer.effective_weight()
        if mode in ("softplus", "hard"):
            assert (W >= 0).all()
        assert torch.isfinite(W).all()


@pytest.mark.parametrize("cls,args", [(NSAFlowLinear, (10, 4)), (NSAFlowConv2d, (3, 4, 3))])
def test_unknown_nonneg_mode_raises_instead_of_silently_doing_nothing(cls, args):
    """v1 silently no-opped on any unrecognised string, e.g. the wrong spelling."""
    layer = cls(*args, w=0.5, nonneg="relu_typo")
    with pytest.raises(ValueError, match="nonneg must be"):
        layer.effective_weight()


def test_softplus_is_used_rather_than_a_hard_clamp_by_default_for_nonneg():
    """A hard clamp is not surjective onto the positive orthant, so its Jacobian
    drops rank and reparameterised stationary points need not be constrained
    stationary points.  softplus keeps the map a submersion."""
    layer = NSAFlowLinear(10, 4, w=0.0, nonneg="softplus")
    with torch.no_grad():
        layer.weight.fill_(-5.0)                     # deep in the clamped region
    layer.effective_weight().sum().backward()
    assert layer.weight.grad.abs().min().item() > 0  # gradient survives everywhere


@pytest.mark.parametrize("bad_w", [-0.1, 1.5])
def test_layers_reject_w_outside_unit_interval(bad_w):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        NSAFlowLinear(10, 4, w=bad_w)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        NSAFlowLayer(w=bad_w)
