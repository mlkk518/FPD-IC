import torch
import time

# 模拟输入
x_lq = torch.randn(32, 3, 256, 256).cuda()

# 方法 1: 低效写法
start_time = time.time()
for _ in range(100):
    c_txt = torch.zeros(x_lq.size(0), 77, 1024).to(x_lq.device)
print(f"低效写法耗时: {time.time() - start_time:.4f} 秒")

# 方法 2: 高效写法
start_time = time.time()
for _ in range(100):
    c_txt = torch.zeros(x_lq.size(0), 77, 1024, device=x_lq.device)
print(f"高效写法耗时: {time.time() - start_time:.4f} 秒")

# 方法 3: 使用 new_zeros
start_time = time.time()
for _ in range(100):
    c_txt = x_lq.new_zeros(x_lq.size(0), 77, 1024)
print(f"new_zeros 写法耗时: {time.time() - start_time:.4f} 秒")