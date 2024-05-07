# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from task.diffusion import CodecDiffusion
import torchaudio
from model.utils import Normalization
from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d(*args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer



@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table



class ResidualBlock(nn.Module):
    def __init__(self,
                 residual_channels,
                 dilation,
                 kernel_size=3,
                 uncond=False,
                 condition_rows=4):
        '''
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable s_codec conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels,
                                   2 * residual_channels,
                                   kernel_size,
                                   padding=((kernel_size-1)*(dilation-1)+kernel_size-1)//2,
                                   dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond: # conditional model
            self.conditioner_projection = Conv1d(condition_rows, 2 * residual_channels, 1)
        else: # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None: # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip
    
class ResidualBlockwithFilm(nn.Module):
    def __init__(self,
                 residual_channels,
                 dilation,
                 kernel_size=3,
                 uncond=False,
                 condition_rows=4):
        '''
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable s_codec conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels,
                                   2 * residual_channels,
                                   kernel_size,
                                   padding=((kernel_size-1)*(dilation-1)+kernel_size-1)//2,
                                   dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond: # conditional model
            self.conditioner_projection = Conv1d(condition_rows, 2 * residual_channels, 1)
        else: # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, gamma, beta, conditioner=None):

        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None: # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

            # Insert FiLM conditions; beta and gammas for each residual layer. 
            y = y*gamma[:, None, None]  + beta[:, None, None] 

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip



class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        if self.params.unconditional: # use unconditional model
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.n_mels, params.residual_channels, 2**(i % params.dilation_base), uncond=params.unconditional)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram=None):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
               (spectrogram is not None and self.spectrogram_upsampler is not None)
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        
        if self.spectrogram_upsampler: # use conditional model
            spectrogram = self.spectrogram_upsampler(spectrogram)
            
        skip = None
        
        index = 0
        for layer in self.residual_layers:
            index += 1
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x

    
   
    
class ClassifierFreeDenoiser(CodecDiffusion):
    def __init__(self,
                 residual_channels,
                 unconditional,
                 condition,
                 p_codec_rows,
                 s_codec_rows,
                 c_codec_rows,
                 norm_args,
                 seg_len,
                 residual_layers = 30,
                 kernel_size = 3,
                 dilation_base = 1,
                 dilation_bound = 4,
                 cond_dropout = 0.5,
                 **kwargs):
        
        self.cond_dropout = cond_dropout
        super().__init__(**kwargs)

        self.condition_normalize = Normalization(norm_args[0], norm_args[1], norm_args[2])

        self.input_projection = Conv1d(p_codec_rows, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(self.betas))
        
        if condition == 'trainable_score':
            trainable_parameters = torch.full((s_codec_rows, self.hparams.seg_len), -1).float() 
            
            trainable_parameters = nn.Parameter(trainable_parameters, requires_grad=True)
            self.register_parameter("trainable_parameters", trainable_parameters)
            self.uncon_dropout = self.trainable_dropout        
            
        elif condition == 'fixed':
            self.uncon_dropout = self.fixed_dropout
        else:
            raise ValueError("unrecognized condition '{condition}'")
        
        # Original dilation was 2**(i % dilation_cycle_length)           
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, dilation_base**(i % dilation_bound), kernel_size, 
                                  uncond=unconditional, condition_rows=s_codec_rows)
            for i in range(residual_layers)
        ])

        #FiLM condition parameter (beta and gamma) generator for score information (MIDI frame roll)
        #The socre length is fixed for 20 sec and 32 resolution/second for the MIDI frame roll. 
        # self.film_layer = nn.Linear(in_features= 200*c_codec_rows, out_features= 2*residual_layers, bias= True)

        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, p_codec_rows, 1)
        nn.init.zeros_(self.output_projection.weight)
        

    def forward(self, x_t, s_codec, c_codec, diffusion_step, sampling=False):
        """
        x_t : (B, 1, T, N)
        s_codec : (B, 4, N)
        c_codec : (B, 7, N)
        """
        
        x_t = x_t.squeeze(1).transpose(1, 2) # (B, 5, LEN)

        s_codec = s_codec.transpose(1, 2).float()
        s_codec = self.condition_normalize(s_codec)
        # c_codec = self.condition_normalize(c_codec)
        if self.training: # only use dropout during training
            s_codec = self.uncon_dropout(s_codec, self.hparams.cond_dropout) # making some score 0 to be unconditional
            # c_codec = self.uncon_dropout(c_codec, self.hparams.cond_dropout) 

        # Generate FiLM conditions (beta and gamma) by FiLM generator (Linear layer)
        # c_codec_flat = torch.flatten(c_codec, start_dim = 1).float()
        # film_feat = self.film_layer(c_codec_flat)

        x = self.input_projection(x_t) # (B, 512, LEN)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step) # (B, 512)
            
        skip = None
        index = 0
        for layer in self.residual_layers:
            index += 1

            # gamma = film_feat[:, 2*(index-1)]
            # beta = film_feat[:, 2*index-1]
            
            # all shapes: (B, 512, LEN)
            x, skip_connection = layer(x, diffusion_step, s_codec)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers)) 
        x = self.skip_projection(x) # (B, 512, LEN)
        x = F.relu(x)
        x = self.output_projection(x) # (B, 4, LEN)

        return x.transpose(1,2).unsqueeze(1), s_codec # (B, 1, LEN, 4)
    
    
    def fixed_dropout(self, x, p, masked_value=-1):
        mask = torch.distributions.Bernoulli(probs=(p)).sample((x.shape[0],)).long()
        mask_idx = mask.nonzero()
        x[mask_idx] = masked_value
        return x
    
    def trainable_dropout(self, x, p):
        mask = torch.distributions.Bernoulli(probs=(p)).sample((x.shape[0],)).long()
        mask_idx = mask.nonzero()
        x[mask_idx] = self.trainable_parameters
        return x        