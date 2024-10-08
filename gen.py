import torch
import cupy as cp
import matplotlib.pyplot as plt
import math
from PIL import Image
import numpy as np
import re

cu_transfer_sqrt_arr_kernel = cp.RawKernel(r'''
extern "C" __global__
void _cu_transfer_sqrt_arr(float* Plane, int datLen, float wavLen, float pxsize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < datLen && y < datLen) {
        float val_x = ((x - datLen / 2.0) * wavLen / datLen / pxsize);
        float val_y = ((y - datLen / 2.0) * wavLen / datLen / pxsize);
        Plane[y * datLen + x] = 1.0 - val_x * val_x - val_y * val_y;
    }
}
''', '_cu_transfer_sqrt_arr')

def cu_transfer_sqrt_arr(datlen: int, wavlen: cp.float32, pxsize: cp.float32):
    # GPUメモリ上のPlane配列をCuPyで初期化
    Plane = cp.empty((datlen, datlen), dtype=cp.float32)

    # スレッドとブロックの数を指定
    threads_per_block = (32, 32)
    blocks_per_grid = (int(cp.ceil(datlen / threads_per_block[0])),
                       int(cp.ceil(datlen / threads_per_block[1])))

    # CUDAカーネルを呼び出し
    cu_transfer_sqrt_arr_kernel(
        (blocks_per_grid[0], blocks_per_grid[1]),  # ブロック数
        (threads_per_block[0], threads_per_block[1]),  # スレッド数
        (Plane, datlen, cp.float32(wavlen), cp.float32(pxsize))  # カーネル引数
    )

    # CuPyの配列をPyTorchテンソルに変換して返す
    return torch.as_tensor(Plane, device='cuda')

def cu_transfer(z0: float, datlen: int, wavlen: float, d_sqr: torch.Tensor):
    transarr = torch.exp(1j * 2.0 * torch.pi * z0 / wavlen * d_sqr)
    return transarr

def create_particle_mask(size, center, diameter, device="cuda"):
    # 正方形のones配列を作成
    grid = torch.ones((size, size), dtype=torch.float32, device=device)
    
    # 円の半径
    radius = diameter / 2.0
    
    # 座標のグリッドを作成
    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="xy")
    
    # 中心からの距離を計算
    distance_from_center = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # 距離が半径以下の領域を0に設定
    grid[distance_from_center <= radius] = 0.
    
    return grid

cu_super_gaussian_filter_kernel = cp.RawKernel(r'''
extern "C" __global__
void _cu_super_gaussian_filter(float* Plane, float sigma, int datLen, float pxsize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < datLen && y < datLen) {
        float val_x = (x - datLen / 2.0) / datLen / pxsize / sigma;
        float val_y = (y - datLen / 2.0) / datLen / pxsize / sigma;
        float insideexp = (val_x * val_x + val_y * val_y);
        Plane[y * datLen + x] = exp(-0.5 * insideexp * insideexp * insideexp);
    }
}
''', '_cu_super_gaussian_filter')

def cu_super_gaussian_filter(propdist: cp.float32, wavlen: cp.float32, pxsize: cp.float32, datlen: int):
    # GPUメモリ上のPlane配列をCuPyで初期化
    Plane = cp.empty((datlen, datlen), dtype=cp.float32)

    # スレッドとブロックの数を指定
    threads_per_block = (32, 32)
    blocks_per_grid = (int(cp.ceil(datlen / threads_per_block[0])),
                       int(cp.ceil(datlen / threads_per_block[1])))
    
    maxi = 1.0 / wavlen * datlen * datlen * pxsize * pxsize / (4.0 * propdist * propdist + datlen * datlen * pxsize * pxsize)**(1.0 / 2.0)
    sigma = maxi / (datlen * pxsize) / (2.0 * math.log(2.0))**(1.0 / 6.0)

    # CUDAカーネルを呼び出し
    cu_super_gaussian_filter_kernel(
        (blocks_per_grid[0], blocks_per_grid[1]),  # ブロック数
        (threads_per_block[0], threads_per_block[1]),  # スレッド数
        (Plane, cp.float32(sigma), datlen, cp.float32(pxsize))  # カーネル引数
    )

    # CuPyの配列をPyTorchテンソルに変換して返す
    return torch.as_tensor(Plane, device='cuda')

def get_closest_pair(xs, ys, zs, pnum):
    initplanerdist = float("inf")
    planerdist = -100.0
    axialdist = -100.0
    
    for i in range(pnum):
        for j in range(i+1, pnum):
            dist = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
            if dist < initplanerdist:
                initplanerdist = dist
                planerdist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                axialdist = math.fabs(zs[i] - zs[j])
            
    return planerdist, axialdist

def gen(numpx = 256, psizerange = (3., 30.), zrange = (400.,600.), maxpnum = 10):
    device = "cuda"
    assert numpx < 1024, "numpx should be less than 1024."
    
    datlen = 1024
    tempwavlen = 0.6328
    tempdx = 10.0
    
    pnum = torch.randint(0, maxpnum, (1,)).item()
    
    if pnum == 0:
        # return torch.ones((numpx, numpx), dtype=torch.float32, device="cpu"), torch.tensor(-100.0, dtype=torch.float32, device="cpu")
        return torch.ones((numpx, numpx), dtype=torch.float32, device="cpu"), torch.tensor((-100.0, -100.0), dtype=torch.float32, device="cpu")
    
    # 粒径と位置をランダムに生成
    psizes = torch.rand(pnum, device=device) * (psizerange[1] - psizerange[0]) + psizerange[0]
    zs = torch.rand(pnum, device=device) * (zrange[1] - zrange[0]) + zrange[0]
    zs.sort(descending=True)
    zs = zs / tempwavlen * (tempdx**2)
    xs = torch.randint(0, numpx, (pnum,), device=device) - numpx // 2
    ys = torch.randint(0, numpx, (pnum,), device=device) - numpx // 2
    
    planerdist, axialdist = get_closest_pair(xs, ys, zs, pnum)
    
    sqrtensor = cu_transfer_sqrt_arr(datlen, tempwavlen, tempdx)
    holo = torch.zeros((datlen, datlen), dtype=torch.complex64 ,device=device)
    
    for i in range(pnum):
        mask = create_particle_mask(datlen, (xs[i]+datlen/2, ys[i]+datlen/2), psizes[i], device=device)
        
        if i == 0:
            holo.copy_(mask.to(torch.complex64))
                
        else:
            holo.mul_(mask)

        if i == pnum - 1:
            trans = cu_transfer(zs[i], datlen, tempwavlen, sqrtensor)
            holo.copy_(torch.fft.ifft2(torch.fft.ifftshift(trans * torch.fft.fftshift(torch.fft.fft2(holo)))))
                        
        else:
            trans = cu_transfer(zs[i]-zs[i+1], datlen, tempwavlen, sqrtensor)
            holo.copy_(torch.fft.ifft2(torch.fft.ifftshift(trans * torch.fft.fftshift(torch.fft.fft2(holo)))))
        
    filter = cu_super_gaussian_filter(zs[0].item(), tempwavlen, tempdx, datlen)
    holo = holo * filter
    result = holo[datlen//2 - numpx//2:datlen//2 + numpx//2, datlen//2 - numpx//2:datlen//2 + numpx//2].abs().square()
    # return result.cpu(), torch.tensor((planerdist), dtype=torch.float32, device="cpu")
    return result.cpu(), torch.tensor((planerdist, axialdist), dtype=torch.float32, device="cpu")
            
# # 可視化
# # plt.imshow(cu_super_gaussian_filter(80000, 0.6328, 10.0, 1024).cpu(), cmap="gray")
# for _ in range(100):
#     output, tup = gen()
# plt.imshow(output.cpu(), cmap="gray")
# plt.title("Output hologram")
# print(output.size())
# print(tup)
# plt.show()

# # データを生成
# result, (planerdist, axialdist) = gen()

# # 画像として保存するためにデータを0-255にスケール変換
# result_scaled = result * 128
# result_scaled = result_scaled.byte()  # データ型をuint8に変換

# # Torch TensorからNumpy配列に変換
# result_np = result_scaled.cpu().numpy()

# # 変数の値をファイル名に組み込む
# filename = f"planerdist_{planerdist.item():.5f}_axialdist_{axialdist.item():.5f}.png"

# # 画像として保存
# image = Image.fromarray(result_np)
# image.save(filename)

# print(f"Saved as {filename}")

# 例: 保存されたファイル名
# filename = "planerdist_123.45678_axialdist_876.54321.png"

# 正規表現でplanerdistとaxialdistをパースする
# match = re.match(r"planerdist_([0-9.]+)_axialdist_([0-9.]+)\.png", filename)
# match = re.match(r"planerdist_([-+]?\d*\.\d+)_axialdist_([-+]?\d*\.\d+)\.png", filename)

# if match:
#     planerdist_parsed = float(match.group(1))
#     axialdist_parsed = float(match.group(2))
#     print(f"Parsed Planar Distance: {planerdist_parsed}")
#     print(f"Parsed Axial Distance: {axialdist_parsed}")
# else:
#     print("Failed to parse the filename.")