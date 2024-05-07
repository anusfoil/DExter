import os, sys, glob, copy
import torch
from collections import defaultdict
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, gaussian_kde, entropy
from scipy.special import kl_div
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from prepare_data import *
import hook


def tensor_pair_swap(x):
    if type(x) == list:
        x = np.array(x)
    # given batched x, swap the pairs from the first dimension
    permute_index = torch.arange(x.shape[0]).view(-1, 2)[:, [1, 0]].contiguous().view(-1)
    return x[permute_index]


def get_batch_slice(batch, idx):
    """given a dictionary batch, get the sliced ditionary given idx"""

    return {k: v[idx] for k, v in batch.items()}


TESTING_GROUP = [
    '07935_seg0',  # bethoven appasionata beginning
    '10713_seg0',  # Jeux D'eau beginning
    '11579_seg3',  # Chopin Ballade 2 transitioning passage
    '00129_seg0',  # Rachmaninov etude tableaux no.5 eb minor
    'ASAP_Schumann_Arabeske_seg0',         # Schumann Arabeske beginning
    'ASAP_Bach_Fugue_bwv_867_seg0',        # Bach fugue 867 beginning 
    'ASAP_Mozart_Piano_Sonatas_12-1_seg0',  # Mozart a minor beginning
    'VIENNA422_Schubert_D783_no15_seg0',   # Vienna422 schubert piece beginning
    #### second group of testing data: paired another performance of the test
    '07925_seg0',  
    '10717_seg0', 
    '11587_seg3',  
    '00130_seg0',  
    'ASAP_Schumann_Arabeske_seg0.',         # select another data example with the same snote_id_path
    'ASAP_Bach_Fugue_bwv_867_seg0.',        
    'ASAP_Mozart_Piano_Sonatas_12-1_seg0.',  
    'VIENNA422_Schubert_D783_no15_seg0.',    
]

def split_train_valid(codec_data, select_num=58008, paired_input=False):
    """select the train and valid set according to the following criteria: 
        - ASAP and VIENNA422 goes to testing set as they are better ground truths
        - split regarding to pieces.(same score path)
    """
    codec_data_ = codec_data
    # codec_data_ = codec_data[list(map(lambda x: not (("ATEPP" in x['score_path']) and ("musicxml_cleaned.musicxml" in x['score_path'])), codec_data))]
    train_idx = int(len(codec_data_) * 0.85 - 1) 
    assert (train_idx % 2 == 0)
    if not select_num:
        return codec_data_[:train_idx], codec_data_[train_idx:]

    selected_cd, unselected_cd = [], []

    if paired_input:
        for idx in range(0, len(codec_data), 2):
            if len(selected_cd) > select_num:  
                break
            cd, cd_ = codec_data[idx], codec_data[idx+1]
            if "ATEPP" not in cd['snote_id_path']:
                selected_cd.extend([cd, cd_])
            else:
                unselected_cd.extend([cd, cd_])
    else:
        for idx in range(0, len(codec_data_)):
            if len(selected_cd) > select_num:  
                break
            cd = codec_data_[idx]
            if "ATEPP" not in cd['snote_id_path']:
                selected_cd.extend([cd])
            else:
                unselected_cd.extend([cd])        
    
    # selected_cd, unselected_cd = defaultdict(list), []
    # for cd in codec_data:
    #     for name in TESTING_GROUP:
    #         if (not selected_cd[name]) and name in cd['snote_id_path']:
    #             selected_cd[name] = cd
    #             break
    #     else:
    #         unselected_cd.append(cd)
            
    
    # np.random.shuffle(unselected_cd)
    train_set = unselected_cd[:train_idx]
    valid_set = selected_cd

    valid_set = np.array(valid_set)[list(map(lambda x: "mu" not in x['piece_name'], valid_set))]     
    np.save(f"{BASE_DIR}/codec_N=200_mixup_train.npy", train_set)
    np.save(f"{BASE_DIR}/codec_N=200_mixup_test.npy", valid_set)

    return train_set, valid_set

def load_transfer_pair(K=50000, N=200):
    """
        returns:
        - transfer_pairs: list of tuple of codec
        - unpaired: list of single codec
    """
    transfer_pairs = np.load(f"{BASE_DIR}/codec_N={N}_mixup_paired_K={K}.npy", allow_pickle=True)
    unpaired = np.load(f"{BASE_DIR}/codec_N={N}_mixup_unpaired_K={K}.npy", allow_pickle=True)
    return transfer_pairs, unpaired


def group_same_seg(valid_set):
    """Group all valid_set by segment, return a dictionary"""
    valid_df = pd.DataFrame(list(valid_set))
    # Group by 'grouptype' and collect indices
    grouped_indices = valid_df.groupby("snote_id_path").apply(lambda group: list(group.index))
    indices_dict = grouped_indices.to_dict()

    indices_dict = {k.split("/")[-1][:-4]: v for k, v in indices_dict.items()}

    # {"ASAP_Bach_Fugue_bwv_854_seg2": [23, 766], ...}
    avg_repetition = sum([len(v) for k, v in indices_dict.items()]) / len(indices_dict) # 5.39
    return indices_dict


def plot_codec(codec1, codec2, ax0, ax1, fig):
    # plot the pred p_codec and label p_codec, on the given two axes

    p_im = ax0.imshow(codec1.T, aspect='auto', origin='lower')
    ax0.set_yticks([0, 1, 2, 3, 4])
    ax0.set_yticklabels(["beat_period", "velocity", "timing", "articulation_log", "pedal"])
    fig.colorbar(p_im, orientation='vertical', ax=ax0)

    s_im = ax1.imshow(codec2.T, aspect='auto', origin='lower')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["beat_period", "velocity", "timing", "articulation_log"])
    fig.colorbar(s_im, orientation='vertical', ax=ax1)
    
    return 


def animate_sampling(t_idx, fig, ax_flat, caxs, noise_list, total_timesteps):
    # noise_list: Tuple of (x_t, t), (x_t-1, t-1), ... (x_0, 0)
    # x_t (B, 1, T, F)
    # clearing figures to prevent slow down in each iteration.d

    fig.canvas.draw()
    for idx in range(len(noise_list[0][0])): # visualize 8 samples in the batch
        ax_flat[2*idx].cla()
        ax_flat[2*idx+1].cla()
        caxs[2*idx].cla()
        caxs[2*idx+1].cla()     

        # roll_pred (1, T, F)
        im1 = ax_flat[2*idx].imshow(noise_list[0][0][idx][0].detach().T.cpu(), aspect='auto', origin='lower')
        im2 = ax_flat[2*idx+1].imshow(noise_list[1 + total_timesteps - t_idx][0][idx][0].T, aspect='auto', origin='lower')
        fig.colorbar(im1, cax=caxs[2*idx])
        fig.colorbar(im2, cax=caxs[2*idx+1])

    fig.suptitle(f't={t_idx}')
    row1_txt = ax_flat[0].text(-400,45,f'Gaussian N(0,1)')
    row2_txt = ax_flat[4].text(-300,45,'x_{t-1}')


def compile_condition(s_codec, c_codec):
    """compile s_codec and c_codec into a joint condition 

    Args:
        s_codec : (B, N, 4)
        c_codec : (B, N, 7)
    """
    return torch.cat((s_codec, c_codec), dim=2)

def apply_normalization(cd, mean, std, i, idx):
    # apply normalization for p codec in codec data
    return (cd['p_codec'][:, i] - mean) / std

def dataset_normalization(train_set, valid_set):
    """ normalize the p_codec, across the dataset range. 
        return mean and std for each column. 
    """
    codec_data = np.hstack([train_set, valid_set])
    dataset_pc = np.vstack([cd['p_codec'] for cd in codec_data])
    means, stds = [], []
    codec_data_ = copy.deepcopy(codec_data)
    for i in range(5):
        mean = dataset_pc[:, i].mean() 
        std = dataset_pc[:, i].std() 
        for idx, cd in enumerate(codec_data):
            codec_data_[idx]['p_codec'][:, i] = apply_normalization(cd, mean, std, i, idx)
        means.append(float(mean))
        stds.append(float(std))  # conversion for save into OmegaConf

    return codec_data_[:len(train_set)], codec_data_[len(train_set):], means, stds

def p_codec_scale(p_codec, means, stds):
    # inverse of normalization applied on p_codec 
    # p_codec: (B, N, 5) or (B, 1, N, 5)
    for i in range(5):
        p_codec[..., i] = p_codec[..., i] * stds[i] + means[i]

    return p_codec


class Normalization():
    """
    This class is for normalizing the input batch by batch.
    The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected.
    In this paper, we found that 'imagewise' normalization works better than 'framewise'
    
    If framewise is used, then X must follow the shape of (B, F, T)
    """
    def __init__(self, min, max, mode='imagewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                x_max = x.max(1, keepdim=True)[0] # Finding max values for each frame
                x_min = x.min(1, keepdim=True)[0]  
                x_std = (x-x_min)/(x_max-x_min) # If there is a column with all zero, nan will occur
                x_std[torch.isnan(x_std)]=0 # Making nan to 0
                x_scaled = x_std * (max - min) + min
                return x_scaled
        elif mode == 'imagewise':
            def normalize(x):
                # x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                # x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                x_max = x.flatten(1).max(1, keepdim=True)[0]
                x_min = x.flatten(1).min(1, keepdim=True)[0]
                x_max = x_max.unsqueeze(1) # Make it broadcastable
                x_min = x_min.unsqueeze(1) # Make it broadcastable 
                x_std = (x-x_min)/(x_max-x_min)
                x_scaled = x_std * (max - min) + min
                x_scaled[torch.isnan(x_scaled)]=min # if piano roll is empty, turn them to min
                return x_scaled
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def __call__(self, x):
        return self.normalize(x)


def render_midi_to_audio(midi_path, output_path=None):
    """The soundfont we used is the Essential Keys-sofrzando-v9.6 from https://sites.google.com/site/soundfonts4u/ """

    if output_path == None:
        output_path = midi_path[:-4] + ".wav"

    os.system(f"fluidsynth -ni ../artifacts/Essential-Keys-sforzando-v9.6.sf2 {midi_path} -F {output_path} ")
    return



if __name__ == "__main__":

    import random
    asap_mid = glob.glob("../Datasets/asap-dataset-alignment/**/*.mid", recursive=True)
    for am in random.choices(asap_mid, k=2):
        render_midi_to_audio(am, output_path=f"{am[35:-4]}.wav")

