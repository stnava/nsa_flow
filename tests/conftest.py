import pytest
import torch
import random
import numpy as np

@pytest.fixture(autouse=True)
def set_seed():
    """Ensure consistent random seed for tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

@pytest.fixture
def device():
    """Device fixture supporting both CPU and GPU testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"
