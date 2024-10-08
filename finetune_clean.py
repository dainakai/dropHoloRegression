import torch.utils
import torch.multiprocessing as mp
from fastai.vision.all import *
from fastai.optimizer import Adam
from fastai.torch_core import defaults

defaults.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import dataset

WANDB_PROJECT = 'drop-collision-detection'

config_defaults = SimpleNamespace(
    batch_size=64,
    epochs=3,
    num_experiments=1,
    learning_rate=0.006892961,
    # learning_rate=0.20892961,
    # model_name="resnet34",
    model_name="convnext_small.fb_in22k",
    pool="concat",
    seed=42,
    wandb_project=WANDB_PROJECT,
    split_func="default",
)

def get_dataset(batch_size, seed, *args, **kwargs):
    train_ds = dataset.StreamingDataset(128*8)
    valid_ds = dataset.StreamingDataset(128*8)
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
    mp.set_start_method("spawn", force=True)   
    dls, metrics = get_dataset(config.batch_size, config.seed)
    learn = vision_learner(
        dls,
        config.model_name,
        metrics=metrics,
        concat_pool=(config.pool=="concat"),
        loss_func=MSELossFlat(),
        opt_func=Adam,
        n_out=2
    )
    
    # learn.unfreeze()
    # learn.lr_find()
    
    print(config.learning_rate)
    
    learn.unfreeze()
    learn.fit_one_cycle(config.epochs, config.learning_rate)

    # learn.fit_one_cycle(config.epochs, config.learning_rate)
    # plt.plot(learn.recorder.losses)
    # plt.show()

if __name__ == "__main__":
    train()
