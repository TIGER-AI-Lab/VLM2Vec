import torch
print(torch.backends.cuda.flash_sdp_enabled())   # FlashAttention
print(torch.backends.cuda.mem_efficient_sdp_enabled())  # Memory-efficient
print(torch.backends.cuda.math_sdp_enabled())    # 普通数学计算

