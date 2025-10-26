import torch
import nsa_flow
Y = torch.nn.Parameter(torch.randn(1, requires_grad=True))
opt_name = "lars"
lr = 0.01
opt = nsa_flow.get_torch_optimizer(opt_name, [Y], lr)
print(opt)
