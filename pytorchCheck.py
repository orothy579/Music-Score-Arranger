import torch

print("MPS 지원 여부:", torch.backends.mps.is_available())
print("MPS 작동 여부:", torch.backends.mps.is_built())
