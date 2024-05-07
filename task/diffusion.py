import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from scipy.stats import pearsonr
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
MIN_MIDI = 21
MAX_MIDI = 108
HOP_LENGTH = 160
SAMPLE_RATE = 16000

import partitura as pt
from utils import animate_sampling, get_batch_slice, plot_codec, p_codec_scale, tensor_pair_swap
from renderer import Renderer

# from model.utils import Normalization
def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    x_start: x0 (B, 1, T, F)
    t: timestep information (B,)
    """    
    # sqrt_alphas is mean of the Gaussian N() - for the forward process distribution
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t] # extract the value of \bar{\alpha} at time=t
    # one minus alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    
    # boardcasting into correct shape
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None].to(x_start.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None].to(x_start.device)

    # scale down the input, and scale up the noise as time increases?
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract_x0(x_t, epsilon, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    x_t: The output from q_sample
    epsilon: The noise predicted from the model
    t: timestep information
    """
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t] # extract the value of \bar{\alpha} at time=t
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None].to(x_t.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None].to(x_t.device)    

    # obtaining x0 like the one in audioLDM?
    pred_x0 = 1.0 / sqrt_alphas_cumprod_t * x_t - epsilon / sqrt_one_minus_alphas_cumprod_t
    # pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * epsilon) / sqrt_alphas_cumprod_t   # original - should be wrong - but it's the same as the labml??

    return torch.clamp(pred_x0, -1.0, 1.0)


class RollDiffusion(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_type,
                 beta_start,
                ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # define beta schedule
        # beta is variance
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas - reparameterization
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # alpha bar
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def training_step(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        batch = batch["frame"].unsqueeze(1)  
        device = batch.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        loss = self.p_losses(batch, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.hparams.loss_type)
        self.log("Train/loss", loss)            
      
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        batch = batch["frame"].unsqueeze(1)  
        device = batch.device
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        loss = self.p_losses(batch, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.hparams.loss_type)
        self.log("Val/loss", loss)
        
    def test_step(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        batch = batch["frame"].unsqueeze(1)
        device = batch.device
        
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        loss = self.p_losses(batch, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.hparams.loss_type)
        self.log("Test/loss", loss)        
        
    def predict_step(self, batch, batch_idx):
        # inference code
        # Unwrapping TensorDataset (list)
        # It is a pure noise
        img = batch[0]
        b = img.shape[0] # extracting batchsize
        device=img.device
        imgs = []
        for i in tqdm(reversed(range(0, self.hparams.timesteps)), desc='sampling loop time step', total=self.hparams.timesteps):
            img = self.p_sample(
                           img,
                           i)
            img_npy = img.cpu().numpy()
            
            if (i+1)%10==0:
                for idx, j in enumerate(img_npy):
                    # j (1, T, F)
                    fig, ax = plt.subplots(1,1)
                    ax.imshow(j[0].T, aspect='auto', origin='lower')
                    self.logger.experiment.add_figure(
                        f"sample_{idx}",
                        fig,
                        global_step=self.hparams.timesteps-i)
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            imgs.append(img_npy)
        torch.save(imgs, 'imgs.pt')
    
    def p_losses(self, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = q_sample(x_start=x_start, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
        predicted_noise = self(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
    
    def p_sample(self, x, t_index):
        # x is Guassian noise
        
        # extracting coefficients at time t
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self(x, t_tensor) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise             
 
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.hparams.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]


class CodecDiffusion(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_type,
                 loss_keys,
                 beta_start,
                 beta_end,                 
                 training,
                 sampling,
                 samples_root,
                 debug=False,
                 generation_filter=0.0,
                 **kwargs
                ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # define beta schedule (beta is variance)
        self.betas = linear_beta_schedule(beta_start, beta_end, timesteps=timesteps)

        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1- alphas_cumprod)
        self.inner_loop = tqdm(range(self.hparams.timesteps), desc='sampling loop time step')
        
        self.reverse_diffusion = getattr(self, sampling['type'])
        # self.reverse_diffusion = getattr(self, sampling.type)
        self.alphas = alphas

    def training_step(self, batch, batch_idx):
        losses, tensors = self.step(batch, batch_idx)
        
        # calculating total loss based on keys give
        total_loss = 0
        for k in self.hparams.loss_keys:
            total_loss += losses[k]
            self.log(f"Train/{k}", losses[k])            
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        losses, tensors = self.step(batch, batch_idx)
        total_loss = 0
        for k in self.hparams.loss_keys:
            total_loss += losses[k]
            self.log(f"Val/{k}", losses[k])           

        # sample a fraction (1 percent) of the testing set
        randgen = torch.rand(1)[0]
        if randgen <= self.hparams.valid_fraction:
            sampled_loss, tc_fig, tempo_vel_loss, tempo_vel_cor = self.predict(batch, batch_idx)
            
            self.logger.log_image(key=f"Val/tempo_curves", images=[tc_fig])
            self.log(f"Val/sampled_loss", sampled_loss)
            self.log(f"Val/tempo_vel_loss", tempo_vel_loss)
            self.log(f"Val/tempo_vel_cor", tempo_vel_cor)

    def test_step(self, batch, batch_idx):
        # evaluation
        if batch_idx < self.hparams.eval_starting_epoch:
            return
        _, _, _, _ = self.predict(batch, batch_idx, evaluate=True)
        

    def predict(self, batch, batch_idx, save_animation=False, evaluate=False):

        batch_label = batch
        batch_source = dict([(k, tensor_pair_swap(v)) for k, v in batch.items()])

        # rescale
        batch_source['p_codec'] = p_codec_scale(batch_source['p_codec'], self.hparams.dataset_means, self.hparams.dataset_stds)
        batch_label['p_codec'] = p_codec_scale(batch_label['p_codec'], self.hparams.dataset_means, self.hparams.dataset_stds)

        save_root = f'{self.hparams.samples_root}/{self.logger._name}/epoch={self.current_epoch}/batch={batch_idx}'

        c_configs = self.set_eval_c_config()
        
        for c_config in c_configs:
            save_root_ = save_root + f"/{c_config}"
            start_noise, sample_steps, c_codec = self.set_predict_start(batch_source, batch_label, c_config=c_config)

            pred_list = self.p_sample(batch_label, start_noise=start_noise, 
                                    sample_steps=sample_steps,
                                    c_codec=c_codec)  

            # noise_list: [(pred_t, t), ..., (pred_0, 0)]
            pcodec_pred, _ = pred_list[-1] # (B, 1, T, F)        
            batch_label_codec = batch_label['p_codec'].unsqueeze(1).cpu()
            N = len(batch_label['score_path']) 
            
            os.makedirs(save_root_, exist_ok=True)

            # save the prediction samples (of one batch) and render. 
            np.save(f'{save_root_}/test_sample.npy', pcodec_pred)

            # rescale the predictions and labels back to normal
            pcodec_pred = p_codec_scale(pcodec_pred, self.hparams.dataset_means, self.hparams.dataset_stds)

            tempo_vel_loss, tempo_vel_cor = 0, 0
            if len(pcodec_pred) % 2 != 1:
                tc_fig, tempo_vel_loss, tempo_vel_cor = self.render_batch(
                    pcodec_pred, batch_source, batch_label, save_root_, evaluate=evaluate)

        if save_animation:
            t_list = torch.arange(1, self.hparams.timesteps, 20).flip(0)
            if t_list[-1] != self.hparams.timesteps:
                t_list = torch.cat((t_list, torch.tensor([self.hparams.timesteps])), 0)
            fig, axes = plt.subplots(2 * N, 1, figsize=(24, 4 * N))

            ax_flat = axes.flatten()
            caxs = []
            for ax in axes.flatten():
                div = make_axes_locatable(ax)
                caxs.append(div.append_axes('right', '5%', '5%'))

            ani = animation.FuncAnimation(fig,
                                        animate_sampling,
                                        frames=tqdm(t_list, desc='Animating'),
                                        fargs=(fig, ax_flat, caxs, noise_list, self.hparams.timesteps),                                          
                                        interval=500,                                          
                                        blit=False,
                                        repeat_delay=1000)
            ani.save(f'{save_root}/animation.gif', dpi=80, writer='imagemagick')

        # plot the comparison between prediction and label 
        fig, ax = plt.subplots(2 * N, 1, figsize=(24, 4 * N))
        for idx, (pred, label) in enumerate(zip(pcodec_pred, batch_label_codec)):
            plot_codec(pred, label, ax[idx*2], ax[idx*2+1], fig)
        self.logger.log_image(key=f"Val/pred_label", images=[fig])
        plt.savefig(f"{save_root}/pred_label.png")


        sampled_loss = self.p_losses(batch_label_codec, torch.tensor(pcodec_pred), loss_type='l2')

        return sampled_loss, tc_fig, tempo_vel_loss, tempo_vel_cor
        

    def inference_one(self, batch):

        sample_steps = self.hparams.timesteps - 1 

        pred_list = self.p_sample(batch, sample_steps=sample_steps)  
        pcodec_pred, _ = pred_list[-1]  # Extract the final prediction

        pcodec_pred_processed = p_codec_scale(pcodec_pred, self.hparams.dataset_means, self.hparams.dataset_stds)

        return pcodec_pred_processed


    def step(self, batch, batch_idx):
        p_codec, s_codec, c_codec = batch['p_codec'], batch['s_codec'], batch['c_codec']

        if self.hparams.drop_c_con:
            c_codec = torch.zeros_like(c_codec)

        batch_size = p_codec.shape[0]
        device = p_codec.device

        p_codec = p_codec.unsqueeze(1)  # (B, 1, T, F)

        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long().cpu() # more diverse sampling
        
        noise = torch.randn_like(p_codec) 
        if self.hparams.training['target'] == "transfer": # invert each pair 
            noise = tensor_pair_swap(p_codec)
            # in transfer context, c_codec is the difference between the two that's being transfered. 
            c_codec = tensor_pair_swap(c_codec) - c_codec # (tgt - src)
        
        x_t = q_sample( # sampling noise at time t
            x_start=p_codec,
            t=t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            noise=noise)
        
        # train to predict the noise, loss is computed for the noise
        if self.hparams.training['mode'] == 'epsilon':
            epsilon_pred, _ = self(x_t, s_codec, c_codec, t) # predict the noise N(0, 1)
            diffusion_loss = self.p_losses(noise, epsilon_pred, loss_type=self.hparams.loss_type)
            pred_p_codec = extract_x0(
                x_t,
                epsilon_pred,
                t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)
                        
        # train to predict p_codec
        elif self.hparams.training['mode'] == 'x_0':
            pred_p_codec, _ = self(x_t, s_codec, c_codec, t) 
            # pred_p_codec = p_codec_scale(pred_p_codec, self.hparams.dataset_means, self.hparams.dataset_stds)
            diffusion_loss = self.p_losses(p_codec, pred_p_codec, loss_type=self.hparams.loss_type)
            
        elif self.hparams.training['mode'] == 'ex_0':
            epsilon_pred, _ = self(x_t, s_codec, c_codec, t) # predict the noise N(0, 1)
            pred_p_codec = extract_x0(
                x_t,
                epsilon_pred,
                t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod) 
            diffusion_loss = self.p_losses(p_codec, pred_p_codec, loss_type=self.hparams.loss_type)   
        else:
            raise ValueError(f"training mode {self.training.mode} is not supported. Please either use 'x_0' or 'epsilon'.")

        losses = {}       
        # compute tempo_vel_loss for the prediced pcodec, for bp and velocity rows. 
        # tempo_vel_loss = self.p_losses(p_codec[:, :, :, :2], pred_p_codec[:, :, :, :2], loss_type='l2') # only tempo and velocity
        weight_sum = sum(self.hparams.loss_weight)
        for idx, name in enumerate(['tempo', 'velocity', 'timing', 'duration', 'pedal']):
            losses[f"recon_{name}_loss"] = (self.p_losses(p_codec[..., idx], pred_p_codec[..., idx], loss_type='l2')
                                            * self.hparams.loss_weight[idx] / weight_sum * self.hparams.recon_loss_weight)

        losses["diffusion_loss"] = diffusion_loss

        tensors = {
            "pred_pcodec": pred_p_codec,
            "label_pcodec": p_codec
        }               
        
        return losses, tensors
    
    def p_sample(self, batch, start_noise=None, sample_steps=None, c_codec=None):

        p_codec = batch['p_codec'].unsqueeze(1) 
        s_codec = batch['s_codec']

        if type(c_codec) == type(None):
            c_codec = batch['c_codec']

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        
        self.inner_loop.refresh()
        self.inner_loop.reset()
        
        if type(start_noise) != type(None):
            noise = start_noise
        else:
            noise = torch.randn_like(p_codec.float()).to(self.device)

        noise_list = []
        noise_list.append((noise, self.hparams.timesteps))

        if type(sample_steps) == type(None):
            sample_steps = self.hparams.timesteps
        for t_index in reversed(range(0, sample_steps)):
            noise, _ = self.reverse_diffusion(noise, s_codec, c_codec, t_index) # cfdg_ddpm_x0 or cfdg_ddpm
            noise_npy = noise.detach().cpu().numpy()
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            noise_list.append((noise_npy, t_index))                       
            self.inner_loop.update()
        
        return noise_list
        
    def p_losses(self, label, prediction, loss_type="l1"):
        
        if loss_type == 'l1':
            loss = F.l1_loss(label, prediction)
        elif loss_type == 'l2':
            loss = F.mse_loss(label, prediction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(label, prediction)
        else:
            raise NotImplementedError()

        return loss
    
    def ddpm(self, x, waveform, t_index):
        # x is Guassian noise
        
        # extracting coefficients at time t
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        epsilon, spec = self(x, waveform, t_tensor)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean, spec
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return (model_mean + torch.sqrt(posterior_variance_t) * noise), spec
        
    def ddpm_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec
    
    def ddim_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)

        if t_index == 0:
            sigma = 0
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = 0                 
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec               
        
    def ddim(self, x, waveform, t_index):
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        epsilon, spec = self(x, waveform, t_tensor)
        
        if t_index == 0:
            model_mean = (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index] 
        else:
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * (
                (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index]) + (
                self.sqrt_one_minus_alphas_cumprod[t_index-1] * epsilon)
            
        return model_mean, spec

    def ddim2ddpm(self, x, waveform, t_index):
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        epsilon, spec = self(x, waveform, t_tensor)

        if t_index == 0:   
            model_mean = (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * (
                (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index]) + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * epsilon) + sigma * torch.randn_like(x)

        return model_mean, spec           
        
    def cfdg_ddpm(self, x, s_codec, c_codec, t_index):
        # x is Guassian noise
        
        # extracting coefficients at time t
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Use our model (noise predictor) to predict the mean 
        epsilon_c, cond = self(x, s_codec, c_codec, t_tensor)
        epsilon_0, _ = self(x, torch.zeros_like(s_codec), torch.zeros_like(c_codec), t_tensor)
        epsilon = self.hparams.sampling['w'] * epsilon_c + (1 - self.hparams.sampling['w']) * epsilon_0
        
        # Equation 11 in the paper
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
        ) 

        if t_index == 0:
            return model_mean, cond
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise

            # From the diffusion-inspired training strategy paper, remove the deviation parameter since noise is deterministic
            if self.hparams.training['target'] == "transfer": 
                return model_mean, cond
            return (model_mean + variance), cond     
        
        
    def cfdg_ddpm_x0(self, x, condition, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Use model (noise & x0 predictor) to predict the mean and weight the conditioned & unconditioned
        if self.hparams.training.mode == "ex_0" or self.hparams.training.mode == "epsilon":
            epsilon_c, cond = self(x, condition, t_tensor)
            epsilon_0, _ = self(x, torch.zeros_like(condition), t_tensor)
            epsilon = self.hparams.sampling['w'] * epsilon_c + (1 - self.hparams.sampling['w']) * epsilon_0
            x0_pred = extract_x0(x, 
                                 epsilon, 
                                 t_tensor, 
                                 sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)

        elif self.hparams.training.mode == "x0": 
            x0_pred_c, cond = self(x, condition, t_tensor)
            x0_pred_0, _ = self(x, torch.zeros_like(condition), t_tensor, sampling=True) # if sampling = True, the input condition will be overwritten
            # wait... is this really weighting???
            # x0_pred =  (1 + self.hparams.sampling.w) * x0_pred_c - self.hparams.sampling.w * x0_pred_0
            x0_pred =  self.hparams.sampling['w'] * x0_pred_c + (1 - self.hparams.sampling['w']) * x0_pred_0

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            # sigma is the noise magnitude (variance to scale noise)
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index - 1] / self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1 - self.alphas[t_index]))  
            if self.hparams.training.mode == "x0": 
                # I modified the division to multiplication according to the inverse of DDPM paper eq.11
                epsilon = (x - self.sqrt_alphas_cumprod[t_index] * x0_pred) * self.sqrt_one_minus_alphas_cumprod[t_index]
            model_mean = (self.sqrt_alphas_cumprod[t_index - 1]) * x0_pred + (
                            torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index - 1] ** 2 - sigma ** 2) * 
                            epsilon) + (  # epsilon
                            sigma * torch.randn_like(x))

        return model_mean, cond

    def cfdg_ddim_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_c, spec = self(x, waveform, t_tensor)
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor)
        x0_pred = (1 + self.hparams.sampling.w) * x0_pred_c - self.hparams.sampling.w * x0_pred_0
        # x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = 0 
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = 0          
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index - 1] ** 2 - sigma ** 2) * (
                    x - self.sqrt_alphas_cumprod[t_index] * x0_pred) / self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec            
    
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = TriStageLRSchedule(optimizer,
        #                                [1e-8, self.hparams.lr, 1e-8],
        #                                [0.2,0.6,0.2],
        #                                max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
        # scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

        # return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]

    def animate_sampling(self, t_idx, fig, ax_flat, caxs, noise_list):
        # noise_list: Tuple of (x_t, t), (x_t-1, t-1), ... (x_0, 0)
        # x_t (B, 1, T, F)
        # clearing figures to prevent slow down in each iteration.d
        fig.canvas.draw()
        for idx in range(len(noise_list[0][0])): # visualize only 4 piano rolls in a batch
            ax_flat[idx].cla()
            ax_flat[4+idx].cla()
            caxs[idx].cla()
            caxs[4+idx].cla()     

            # roll_pred (1, T, F)
            im1 = ax_flat[idx].imshow(noise_list[0][0][idx][0].detach().T.cpu(), aspect='auto', origin='lower')
            im2 = ax_flat[4+idx].imshow(noise_list[1 + self.hparams.timesteps - t_idx][0][idx][0].T, aspect='auto', origin='lower')
            fig.colorbar(im1, cax=caxs[idx])
            fig.colorbar(im2, cax=caxs[4+idx])

        fig.suptitle(f't={t_idx}')
        row1_txt = ax_flat[0].text(-400,45,f'Gaussian N(0,1)')
        row2_txt = ax_flat[4].text(-300,45,'x_{t-1}')       

    def render_batch(self, pcodec_pred, batch_source, batch_label, save_root, evaluate=False):
        B = len(pcodec_pred) 

        fig, ax = plt.subplots(int(B/2), 4, figsize=(24, 3*B))

        tempo_vel_loss, tempo_vel_cor = 0, 0
        for idx in range(B): 
            renderer = Renderer(save_root, 
                                pcodec_pred[idx],  
                                source_data=get_batch_slice(batch_source, idx), 
                                label_data=get_batch_slice(batch_label, idx), 
                                with_source=self.hparams.transfer,
                                idx=idx, B=B)
            
            tvl, tvc = renderer.render_sample(save_sourcelabel=True)
            tempo_vel_loss += tvl 
            tempo_vel_cor += tvc
            if renderer.success:
                renderer.plot_curves(ax)
                if evaluate:
                    renderer.save_performance_features(save_source=False, save_label=True)
                    renderer.save_pf_distribution()
                
        plt.savefig(f"{save_root}/tempo_curves.png") 
        return fig, tempo_vel_loss / B, tempo_vel_cor / B


    def set_predict_start(self, batch_source, batch_label, c_config=""):
        """set up the starting point of inference given the configuration. 
    
        - transfer in training: start from source codec, conditioned on label c_codec - source c_codec
        - transfer in inference: starting from source codec + noise (75%), conditioned on label c_codec
        - no transfer: starting from noise, conditioned on label c_codec [or other adjustable c_codec]

        c_config: e.g. "melodiousness_more", changes the parameter by half or double
        """
        batch_source_codec = batch_source['p_codec']
        batch_source_codec = batch_source_codec.unsqueeze(1)  # (B, 1, T, F)

        sample_steps = self.hparams.timesteps - 1 
        c_codec = batch_label['c_codec'] 

        if "transfer" not in self.hparams.training.target: # pure noise 
            noise = torch.randn_like(batch_source_codec)

            if self.hparams.transfer: # only transfer in inference
                sample_steps = int((self.hparams.timesteps - 1) * self.hparams.sample_steps_frac) # steps for noisify the source
                start_noise = q_sample(x_start=batch_source_codec,
                                    t=torch.tensor([sample_steps] * int(batch_source['p_codec'].shape[0])),
                                    sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                                    sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                                    noise=noise)
            else:
                start_noise = None

        else: # transfer in training
            start_noise = batch_source_codec
            c_codec = batch_label['c_codec'] - batch_source['c_codec']

        if c_config != "":
            c, control = c_config.split("/")
            c_idx = self.c_type.index(c)
            if control == "more":
                c_codec[c_idx] *= 2
            if control == "less":
                c_codec[c_idx] *= 0.5 # normal is just itself
        return start_noise, sample_steps, c_codec

    def set_eval_c_config(self):
        # set the c_config for oracle evaluation
        if self.hparams.condition_eval:
            self.c_type = ["melodiousness", "articulation", "rhythm complexity", "rhythm stability", "dissonance", "tonal stability", "minorness"]
            c_configs = [f"{c}_{control}" for c in self.c_type for control in ["less", "normal", "more"]]
        else:
            c_configs = [""]
        return c_configs
