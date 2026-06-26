import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .retraction import nsa_flow_retract_auto
from .energy import fidelity_scaled, invariant_orthogonality_defect

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): 
        return self.net(x)

class NSAFlowLayer(nn.Module):
    """
    Unified NSA Flow layer:
      - Optional residual MLP transform
      - Adaptive retraction blend (learnable w_retract)
      - Optional nonnegativity enforcement ('none', 'soft', 'hard')
      - Can compute fidelity + orthogonality losses if target provided
    """
    def __init__(
        self,
        k,
        hidden=64,
        w_retract=0.5,
        retraction_type="soft_polar",
        apply_nonneg="none",
        residual=True,
        use_transform=True,
    ):
        super().__init__()
        self.k = k
        self.hidden = hidden
        self.retraction_type = retraction_type
        self.apply_nonneg = apply_nonneg
        self.residual = residual
        self.use_transform = use_transform

        if use_transform:
            self.transform_net = nn.Sequential(
                nn.Linear(k, hidden),
                nn.ReLU(),
                nn.Linear(hidden, k)
            )
        else:
            self.transform_net = None

        w_init = float(w_retract)
        w_init = max(1e-6, min(1.0 - 1e-6, w_init))
        logit_w = math.log(w_init / (1.0 - w_init))
        self.w_retract = nn.Parameter(torch.tensor(logit_w))
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _apply_nonneg(self, Y):
        if self.apply_nonneg == "soft":
            return F.softplus(Y)
        elif self.apply_nonneg == "relu":
            return F.relu(Y)
        elif self.apply_nonneg == "hard":
            return torch.clamp(Y, min=0.0)
        return Y

    def forward(self, Y, target=None):
        """
        Forward pass with Convex Combination Flow:
        Y_out = (1-w)*Y + w*retract(Y + transform(Y))
        """
        w = torch.sigmoid(self.w_retract)
        
        if self.use_transform:
            Y_prop = Y + self.transform_net(Y)
        else:
            Y_prop = Y

        Y_orth = nsa_flow_retract_auto(Y_prop, w_retract=1.0, retraction_type=self.retraction_type)

        if self.residual:
            Y_out = (1.0 - w) * Y + w * Y_orth
        else:
            Y_out = Y_orth

        Y_out = self._apply_nonneg(Y_out)

        if target is not None:
            fid = fidelity_scaled(Y_out, target)
            orth = invariant_orthogonality_defect(Y_out)
            w_dyn = torch.sigmoid(self.alpha)
            total_loss = fid * (1.0 - w_dyn) + orth * w_dyn
            return Y_out, total_loss, fid, orth, w_dyn
        else:
            return Y_out

class NSAFlowLinear(nn.Module):
    """
    A linear layer where the parameter matrix W is continuously projected 
    towards the Stiefel manifold during training, creating an orthogonal-like 
    basis without requiring exact hard constraints.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_retract: float = 0.5,
        retraction_type: str = "soft_polar",
        apply_nonneg: str = "none"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.retraction_type = retraction_type
        self.apply_nonneg = apply_nonneg

        self.weight_raw = nn.Parameter(torch.empty(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        w_init = float(w_retract)
        w_init = max(1e-6, min(1.0 - 1e-6, w_init))
        self.w_logit = nn.Parameter(torch.tensor(math.log(w_init / (1.0 - w_init))))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_raw)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _apply_nonneg(self, W):
        if self.apply_nonneg == "soft":
            return F.softplus(W)
        elif self.apply_nonneg == "relu":
            return F.relu(W)
        elif self.apply_nonneg == "hard":
            return torch.clamp(W, min=0.0)
        return W

    def get_manifold_weight(self):
        w = torch.sigmoid(self.w_logit)
        W_orth = nsa_flow_retract_auto(self.weight_raw, w_retract=1.0, retraction_type=self.retraction_type)
        W_eff = (1.0 - w) * self.weight_raw + w * W_orth
        return self._apply_nonneg(W_eff)

    def forward(self, x):
        W_eff = self.get_manifold_weight()
        out = torch.matmul(x, W_eff)
        if self.bias is not None:
            out = out + self.bias
        return out

class NSAFlowConv2d(nn.Conv2d):
    """
    A 2D convolutional layer where the filter weights are continuously projected 
    towards the Stiefel manifold.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        w_retract: float = 0.5,
        retraction_type: str = "soft_polar",
        apply_nonneg: str = "none"
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )
        self.retraction_type = retraction_type
        self.apply_nonneg = apply_nonneg

        w_init = float(w_retract)
        w_init = max(1e-6, min(1.0 - 1e-6, w_init))
        self.w_logit = nn.Parameter(torch.tensor(math.log(w_init / (1.0 - w_init))))

        self._reset_orthogonal_parameters()

    def _reset_orthogonal_parameters(self):
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def _apply_nonneg(self, W):
        if self.apply_nonneg == "soft":
            return F.softplus(W)
        elif self.apply_nonneg == "relu":
            return F.relu(W)
        elif self.apply_nonneg == "hard":
            return torch.clamp(W, min=0.0)
        return W

    def get_manifold_weight(self):
        w = torch.sigmoid(self.w_logit)
        W_flat = self.weight.view(self.weight.size(0), -1).T
        W_orth_flat = nsa_flow_retract_auto(W_flat, w_retract=1.0, retraction_type=self.retraction_type)
        W_eff_flat = (1.0 - w) * W_flat + w * W_orth_flat
        W_eff = W_eff_flat.T.view_as(self.weight)
        return self._apply_nonneg(W_eff)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from torch.nn.modules.utils import _pair
        W_eff = self.get_manifold_weight()
        
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            W_eff, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, W_eff, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
