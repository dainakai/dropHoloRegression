import torch
import numpy as np
import math
import matplotlib.pyplot as plt

def transfer_sqrt_arr(datlen: int, wavlen: float, pxsize: float):
    x = np.arange(datlen)
    y = np.arange(datlen)
    xv, yv = np.meshgrid(x, y, indexing='xy')

    val_x = ((xv - datlen / 2.0) * wavlen) / (datlen * pxsize)
    val_y = ((yv - datlen / 2.0) * wavlen) / (datlen * pxsize)
    Plane = 1.0 - val_x**2 - val_y**2

    return torch.from_numpy(Plane.astype(np.float32))

def transfer(z0: float, datlen: int, wavlen: float, d_sqr: torch.Tensor):
    transarr = torch.exp(1j * 2.0 * torch.pi * z0 / wavlen * d_sqr)
    return transarr

def create_particle_mask(size, center, diameter, device="cpu"):
    grid = torch.ones((size, size), dtype=torch.float32, device=device)
    
    radius = diameter / 2.0
    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="xy")
    
    distance_from_center = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
    grid[distance_from_center <= radius] = 0.
    
    return grid

def super_gaussian_filter(propdist: float, wavlen: float, pxsize: float, datlen: int):
    numerator = datlen * datlen * pxsize * pxsize
    denominator = np.sqrt(4.0 * propdist * propdist + datlen * datlen * pxsize * pxsize)
    maxi = numerator / (wavlen * denominator)
    sigma = maxi / (datlen * pxsize) / ((2.0 * np.log(2.0))**(1.0 / 6.0))

    x = np.arange(datlen)
    y = np.arange(datlen)
    xv, yv = np.meshgrid(x, y, indexing='xy')

    val_x = (xv - datlen / 2.0) / (datlen * pxsize * sigma)
    val_y = (yv - datlen / 2.0) / (datlen * pxsize * sigma)
    insideexp = val_x**2 + val_y**2
    Plane = np.exp(-0.5 * insideexp**3)

    return torch.from_numpy(Plane.astype(np.float32))

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
                axialdist = abs(zs[i] - zs[j])

    return planerdist, axialdist

def gen(numpx=256, psizerange=(3., 30.), zrange=(400., 600.), maxpnum=10, device="cpu"):
    assert numpx < 1024, "numpx should be less than 1024."

    datlen = 1024
    tempwavlen = 0.6328
    tempdx = 10.0

    pnum = torch.randint(0, maxpnum, (1,)).item()

    if pnum == 0:
        return torch.ones((numpx, numpx), dtype=torch.float32, device=device), torch.tensor((-100.0, -100.0), dtype=torch.float32, device="cpu")

    psizes = torch.rand(pnum, device=device) * (psizerange[1] - psizerange[0]) + psizerange[0]
    zs = torch.rand(pnum, device=device) * (zrange[1] - zrange[0]) + zrange[0]
    zs, _ = torch.sort(zs, descending=True)
    zs = zs / tempwavlen * (tempdx**2)
    xs = torch.randint(0, numpx, (pnum,), device=device) - numpx // 2
    ys = torch.randint(0, numpx, (pnum,), device=device) - numpx // 2

    planerdist, axialdist = get_closest_pair(xs.tolist(), ys.tolist(), zs.tolist(), pnum)

    sqrtensor = transfer_sqrt_arr(datlen, tempwavlen, tempdx)
    holo = torch.zeros((datlen, datlen), dtype=torch.complex64, device=device)

    for i in range(pnum):
        mask = create_particle_mask(datlen, (xs[i]+datlen/2, ys[i]+datlen/2), psizes[i], device=device)

        if i == 0:
            holo.copy_(mask.to(torch.complex64))
        else:
            holo.mul_(mask)

        if i == pnum - 1:
            trans = transfer(zs[i].item(), datlen, tempwavlen, sqrtensor)
            holo = torch.fft.ifft2(torch.fft.ifftshift(trans * torch.fft.fftshift(torch.fft.fft2(holo))))
        else:
            trans = transfer((zs[i]-zs[i+1]).item(), datlen, tempwavlen, sqrtensor)
            holo = torch.fft.ifft2(torch.fft.ifftshift(trans * torch.fft.fftshift(torch.fft.fft2(holo))))

    filter = super_gaussian_filter(zs[0].item(), tempwavlen, tempdx, datlen)
    holo = holo * filter
    result = holo[datlen//2 - numpx//2:datlen//2 + numpx//2, datlen//2 - numpx//2:datlen//2 + numpx//2].abs().square()
    return result, torch.tensor((planerdist, axialdist), dtype=torch.float32, device="cpu")

            
# # 可視化
# # plt.imshow(cu_super_gaussian_filter(80000, 0.6328, 10.0, 1024).cpu(), cmap="gray")
for _ in range(100):
    output, tup = gen(device="cpu")
# plt.imshow(output, cmap="gray")
# plt.title("Output hologram")
# print(output.size())
# print(tup)
# plt.show()

