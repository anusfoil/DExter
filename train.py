import os, sys, random
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import hydra
from hydra.utils import to_absolute_path
import model as Model
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils import *
from prepare_data import *
    

@hydra.main(config_path="config", config_name="train")
def main(cfg):
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    os.environ['HYDRA_FULL_ERROR'] = "1"
    os.system("wandb sync --clean-force --clean-old-hours 3")

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    cfg.data_root = to_absolute_path(cfg.data_root)
    if cfg.train_target == "transfer": # load only from paired set. 
        # TODO: modify after the new hdf5 loading
        paired, _ = load_transfer_pair(K=2000000, N=cfg.seg_len) 
        train_set, valid_set = split_train_valid(paired, select_num=0, paired_input=True)
        assert(len(train_set) % 2 == 0)
        assert(len(valid_set) % 2 == 0)   
        cfg.dataloader.train.shuffle = False

    else: 

        hdf5_path = f"{BASE_DIR}/codec_N={cfg.seg_len}_mixup.hdf5"
        train_set, valid_set = load_data_from_hdf5(hdf5_path)

    random.shuffle(train_set)

    # Normalize data
    train_set, valid_set, means, stds = dataset_normalization(train_set, valid_set)
    cfg.task.dataset_means = means
    cfg.task.dataset_stds = stds

    train_loader = DataLoader(train_set, **cfg.dataloader.train)
    val_loader = DataLoader(valid_set, **cfg.dataloader.val) 

    # set the fraction so that around 20 batchs are output for listen.
    cfg.task.valid_fraction = 20 / len(val_loader)      

    # Model
    if cfg.load_trained:
        model = getattr(Model, cfg.model.model.name).load_from_checkpoint(
                                            checkpoint_path=cfg.pretrained_path,\
                                            **cfg.model.model.args, 
                                            **cfg.task)
    else:
        model = getattr(Model, cfg.model.model.name)(**cfg.model.model.args, **cfg.task)
            
    lw = "".join(str(x) for x in cfg.task.loss_weight)
    if cfg.model.model.name == 'DenoiserUnet':
        name = f"target{cfg.train_target}-lw{lw}-len{cfg.seg_len}-beta{round(cfg.task.beta_end, 2)}-steps{cfg.task.timesteps}-{cfg.task.training.mode}-" + \
                f"Transfer{cfg.task.transfer}-ssfrac{cfg.task.sample_steps_frac}-" + \
                f"{cfg.task.sampling.type}-w={cfg.task.sampling.w}-" \
                f"dim={cfg.model.model.args.dim}" 
    else:
        name = f"target{cfg.train_target}-lw{lw}-len{cfg.seg_len}-beta{round(cfg.task.beta_end, 2)}-steps{cfg.task.timesteps}-{cfg.task.training.mode}-" + \
                f"Transfer{cfg.task.transfer}-ssfrac{cfg.task.sample_steps_frac}-" + \
                f"L{cfg.model.model.args.residual_layers}-C{cfg.model.model.args.residual_channels}-" + \
                f"{cfg.task.sampling.type}-w={cfg.task.sampling.w}-" + \
                f"p={cfg.model.model.args.cond_dropout}-k={cfg.model.model.args.kernel_size}-" + \
                f"dia={cfg.model.model.args.dilation_base}-{cfg.model.model.args.dilation_bound}"

    if cfg.test_only:
        name = "TEST-" + name

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint, dirpath=f'artifacts/checkpoint/{name}')    
    wandb_logger = WandbLogger(project="DiffPerformer", name=name, save_code=True)   
    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback,],
                         logger=wandb_logger,
                        #  stochastic_weight_avg=True
                         )
    if not cfg.test_only:
        trainer.fit(model, train_loader, val_loader)
    
    trainer.validate(model, val_loader)
    
if __name__ == "__main__":
    main()