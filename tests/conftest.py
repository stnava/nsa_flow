import numpy as np
import pytest
import torch

# Every test states its own dtype explicitly.  The default is never mutated at
# module scope, because doing so leaks into whatever pytest collects next and
# makes results depend on collection order.
F64 = torch.float64


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    np.random.seed(42)
