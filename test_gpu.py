import torch

GPU_AVAIL = torch.cuda.is_available()
print(GPU_AVAIL)