from thop import profile
import torch
import unet
iterations = 300   # 重复计算的轮次

model = unet.UNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =model.to(device)

random_input = torch.randn(1, 1, 480, 640).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

flops, params = profile(model.to(device), inputs=(random_input,))

print("FLOPs：", flops)
print("参数量：", params)

print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

