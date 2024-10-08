import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.multiprocessing as mp
import gen as g
from fastai.vision.all import *
import torchvision as tv
from torchvision.transforms import GaussianBlur

class StreamingDataset(IterableDataset):
    def __init__(self, data_size=1000, buffer_size=1000):
        self.buffer_size = buffer_size
        self.data_size = data_size

    def __iter__(self):
        self.data_source = self.data_generator(data_size=self.data_size)
        buffer = []
        for item in self.data_source:
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                yield from buffer
                buffer = []
        if buffer:
            yield from buffer

    def __len__(self):
        return self.data_size

    def data_generator(self, data_size=1000):
        for i in range(data_size):
            img, target = g.gen(numpx=224)
            yield img.unsqueeze(0).repeat(3, 1, 1), target


class AddGaussianNoiseorExpBackground(RandTransform):
    def __init__(self, mean=0., std=100., noise_path="/home/dai/Documents/Documents/repos/EffnetV2/noise/", tfmsp=0.5, **kwargs):
        self.mean = mean
        self.std = std
        self.noise_list = glob.glob(noise_path + "*/*.png")
        self.tfmsp = tfmsp
        super().__init__(**kwargs)
        
    def encodes(self, x: TensorImage):
        if random.random() < self.tfmsp:
            noise = torch.normal(self.mean, self.std, size=(x.shape[0], x.shape[2], x.shape[3]))
            noise = noise.unsqueeze(1).repeat(1, 3, 1, 1).to(x.device)
            return x + noise
        else:
            for i in range(x.shape[0]):
                noiseimg = Image.open(self.noise_list[random.randint(0, len(self.noise_list)-1)])
                noise = tv.transforms.functional.to_tensor(noiseimg).to(x.device)
                xs = random.randint(x.shape[2], noise.shape[1] - x.shape[2])
                ys = random.randint(x.shape[3], noise.shape[2] - x.shape[3])
                sub_noise = noise[0,xs:xs+x.shape[2], ys:ys+x.shape[3]]
                sub_noise = sub_noise - sub_noise.mean()
                x[i,0,:,:] = x[i,0,:,:] + sub_noise[:,:]
                x[i,1,:,:] = x[i,1,:,:] + sub_noise[:,:]
                x[i,2,:,:] = x[i,2,:,:] + sub_noise[:,:]
            return x
        
class GaussianFilter(RandTransform):
    def __init__(self, kernel_size=9, sigma=(0.0, 2.2), **kwargs):
        self.trfm = GaussianBlur(kernel_size, sigma)
        super().__init__(**kwargs)
        
    def encodes(self, x: TensorImage):
        return self.trfm(x)

train_trfm = [*aug_transforms(size=224, max_zoom=1.0, max_warp=0.0, mult=1.0 ), AddGaussianNoiseorExpBackground(mean=0., std=7., p=0.75), GaussianFilter(kernel_size=9, sigma=(0.1, 2.2))]
valid_trfm = [Resize(224)]

if __name__ == "__main__":    
    # mp.set_start_method("spawn", force=True)

    # データセットとデータローダーの設定
    dataset = StreamingDataset(data_size=100)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0)

    # データの使用
    for batch in dataloader:
        # ここでバッチ処理を行う
        print(batch[0].size())