import torch.utils
import wandb
import argparse
import torchvision as tv
from torchvision.transforms import GaussianBlur
import torch.multiprocessing as mp
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from fastai.callback.tracker import SaveModelCallback
from fastai.optimizer import Adam
from fastai.torch_core import defaults

defaults.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import dataset

WANDB_PROJECT = 'drop-collision-detection'

config_defaults = SimpleNamespace(
    batch_size=64,
    epochs=5,
    num_experiments=1,
    learning_rate=0.02909,
    model_name="convnext_small.fb_in22k",
    pool="concat",
    seed=42,
    wandb_project=WANDB_PROJECT,
    split_func="default",
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)
    parser.add_argument('--epochs', type=int, default=config_defaults.epochs)
    parser.add_argument('--num_experiments', type=int, default=config_defaults.num_experiments)
    parser.add_argument('--learning_rate', type=float, default=config_defaults.learning_rate)
    parser.add_argument('--model_name', type=str, default=config_defaults.model_name)
    parser.add_argument('--split_func', type=str, default=config_defaults.split_func)
    parser.add_argument('--pool', type=str, default=config_defaults.pool)
    parser.add_argument('--seed', type=int, default=config_defaults.seed)
    parser.add_argument('--wandb_project', type=str, default=WANDB_PROJECT)
    return parser.parse_args()

def get_gpu_mem(device=0):
    "Memory usage in GB"
    gpu_mem = torch.cuda.memory_stats_as_nested_dict(device=device)
    return (gpu_mem["reserved_bytes"]["small_pool"]["peak"] + gpu_mem["reserved_bytes"]["large_pool"]["peak"])*1024**-3

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

def get_dataset(batch_size, seed, *args, **kwargs):
    # カスタムデータセットを使用
    train_ds = dataset.StreamingDataset(32768)  # 任意の大きな数
    valid_ds = dataset.StreamingDataset(64)
    # dls = DataLoaders.from_dsets(
    #     train_ds, valid_ds,
    #     item_tfms=Resize(224),
    #     batch_tfms=Resize(224),
    #     # batch_tfms=[*aug_transforms(size=224, max_zoom=1.0, max_warp=0.0, mult=1.0 ), AddGaussianNoiseorExpBackground(mean=0., std=7., p=0.75), GaussianFilter(kernel_size=9, sigma=(0.1, 2.2))],
    #     bs=batch_size,
    #     val_bs=batch_size,
    #     seed=seed,
    #     num_workers=0,
    #     shuffle = False
    # )
    
    dls = DataLoaders.from_dsets(
        train_ds, valid_ds,
        bs=batch_size,
        val_bs=batch_size,
        seed=seed,
        num_workers=0,
        shuffle = False,
        device=defaults.device
    )
    
    metrics = [rmse]
    return dls, metrics

def train(config=config_defaults):
    with wandb.init(project=config.wandb_project, config=config) as run:
        mp.set_start_method("spawn", force=True)
        run.name = f"20241003_fullunfreeze_regression_{config.learning_rate}"     
        # run.name = f"{config.model_name}_bs_{config.batch_size}_lr_{config.learning_rate}_pool_{config.pool}"     
        config = wandb.config
        dls, metrics = get_dataset(config.batch_size, config.seed)
        learn = vision_learner(
            dls,
            config.model_name,
            metrics=metrics,
            concat_pool=(config.pool=="concat"),
            loss_func=MSELossFlat(),
            opt_func=Adam,          
            n_out=2,     
            cbs=[WandbCallback(log=None, log_preds=False)]
        ).to_fp16()
        ti = time.perf_counter()
        # fine_tuneの代わりにfit_one_cycleを使用してコサインアニーリングを適用
        learn.unfreeze()
        learn.fit_one_cycle(config.epochs, config.learning_rate)
        
        learn.fine_tune(config.epochs, config.learning_rate)
        
        wandb.summary["GPU_mem"] = get_gpu_mem(learn.dls.device)
        wandb.summary["model_family"] = config.model_name.split('_')[0]
        wandb.summary["fit_time"] = time.perf_counter() - ti
        
        # learn.export(fname="model.pkl")

if __name__ == "__main__":
    args = parse_args()
    train(config=args)
