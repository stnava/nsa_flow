"""Tests for the high-level nsa_flow wrapper and multi-optimizer dispatch."""
import pytest
import torch
import numpy as np

from nsa_flow import nsa_flow

F64 = torch.float64


def test_auto_dispatch_nonneg_data():
    """Non-negative data matrix + k -> dispatches to nsa_flow_data."""
    torch.manual_seed(10)
    X = torch.rand(40, 15, dtype=F64)
    r = nsa_flow(X, k=3, w=0.5)
    assert r.Y.shape == (15, 3)
    assert (r.Y >= -1e-12).all(), "Basis must be non-negative"
    assert "matrix_free" in r, "Expected reconstruct metadata in result"
    assert r.converged, f"Expected convergence, got {r.stop_reason}"


def test_auto_dispatch_signed_data():
    """Data with negative entries + k -> dispatches to nsa_flow_signed."""
    torch.manual_seed(11)
    X = torch.randn(40, 15, dtype=F64)
    r = nsa_flow(X, k=3, w=0.5)
    assert r.Y.shape == (15, 3)
    assert (r.Y < 0).any() and (r.Y > 0).any(), "Expected mixed-sign basis"
    assert "parts" in r, "Expected parts metadata in signed result"
    assert "lobe_overlap" in r
    assert r.converged


def test_auto_dispatch_signed_consolidate():
    """Signed data with consolidate=True guarantees zero lobe overlap."""
    torch.manual_seed(12)
    X = torch.randn(40, 15, dtype=F64)
    r = nsa_flow(X, k=3, w=0.5, consolidate=True)
    assert r["lobe_overlap"] == 0.0, "Consolidated must have exactly zero overlap"
    assert r.converged


def test_auto_dispatch_anchored_target():
    """Passing a target [p, k] without k -> dispatches to anchored solver."""
    target = torch.rand(25, 4, dtype=F64)
    r = nsa_flow(target, w=0.5)
    assert r.Y.shape == (25, 4)
    assert (r.Y >= -1e-12).all()
    assert "fidelity_mode" in r, "Expected anchored metadata"
    assert r.converged


def test_explicit_modes():
    """User can explicitly override mode."""
    torch.manual_seed(13)
    X = torch.rand(30, 10, dtype=F64)
    r_data = nsa_flow(X, k=2, mode="data")
    assert (r_data.Y >= -1e-12).all()

    r_signed = nsa_flow(X, k=2, mode="signed")
    assert "parts" in r_signed


def test_optimizer_lbfgs_data():
    """optimizer='lbfgs' fits non-negative basis with quasi-Newton."""
    torch.manual_seed(14)
    X = torch.rand(35, 12, dtype=F64)
    r = nsa_flow(X, k=3, w=0.5, optimizer="lbfgs")
    assert r.Y.shape == (12, 3)
    assert (r.Y >= -1e-12).all()
    assert r.converged


def test_optimizer_lbfgs_signed():
    """optimizer='lbfgs' fits signed basis with quasi-Newton."""
    torch.manual_seed(15)
    X = torch.randn(35, 12, dtype=F64)
    r = nsa_flow(X, k=3, w=0.5, optimizer="lbfgs")
    assert r.Y.shape == (12, 3)
    assert "parts" in r
    assert r.converged


def test_optimizer_torch_lbfgs_data():
    """Explicit torch_lbfgs fits non-negative basis with pure PyTorch quasi-Newton."""
    torch.manual_seed(16)
    X = torch.rand(35, 12, dtype=F64)
    r = nsa_flow(X, k=3, w=0.5, optimizer="torch_lbfgs")
    assert r.Y.shape == (12, 3)
    assert (r.Y >= -1e-12).all()
    assert r.converged


def test_optimizer_torch_lbfgs_signed():
    """Explicit torch_lbfgs fits signed contrast basis with pure PyTorch quasi-Newton."""
    torch.manual_seed(17)
    X = torch.randn(35, 12, dtype=F64)
    r = nsa_flow(X, k=3, w=0.5, optimizer="torch_lbfgs")
    assert r.Y.shape == (12, 3)
    assert "parts" in r
    assert r.converged


def test_optimizer_torch_lbfgs_anchored():
    """torch_lbfgs fits anchored target matrix."""
    torch.manual_seed(18)
    target = torch.rand(25, 4, dtype=F64)
    r = nsa_flow(target, w=0.5, optimizer="torch_lbfgs")
    assert r.Y.shape == (25, 4)
    assert (r.Y >= -1e-12).all()
    assert r.converged
