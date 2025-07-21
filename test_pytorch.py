import torch

print(f"{torch.cuda.is_available()=}")
t = torch.empty(size=(2, 3), dtype=torch.float, device="cuda")
ta = torch.abs(t)
print(ta)

