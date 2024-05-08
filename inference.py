import os, sys, glob, argparse, yaml
import warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt

import hydra
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
torch.set_printoptions(sci_mode=False)

import model as Model
from utils import *
from renderer import Renderer


@hydra.main(config_path="config", config_name="inference")
def main(cfg):
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    os.environ['HYDRA_FULL_ERROR'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpus[0])
    os.system("wandb sync --clean-force --clean-old-hours 3")

    score = pt.load_musicxml(cfg.score_path)

    sna = score.note_array()
    s_codec = torch.tensor(rfn.structured_to_unstructured(
        sna[['onset_div', 'duration_div', 'pitch', 'voice']]))

    s_codec_tensor = []
    for idx in range(0, len(s_codec) - cfg.overlap, cfg.seg_len - cfg.overlap):  
        end_idx = idx + cfg.seg_len

        if end_idx > len(s_codec):
            end_idx = len(s_codec)
        seg_s_codec = s_codec[idx:end_idx]

        if len(seg_s_codec) < cfg.seg_len:
            seg_s_codec = np.pad(seg_s_codec, ((0, cfg.seg_len - len(seg_s_codec)), (0, 0)), mode='constant', constant_values=0)

        s_codec_tensor.append(torch.tensor(seg_s_codec))

    s_codec_tensor = torch.stack(s_codec_tensor)

    model = getattr(Model, cfg.model.model.name).load_from_checkpoint(
                                        checkpoint_path=cfg.pretrained_path,\
                                        **cfg.model.model.args, 
                                        **cfg.task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        model.to(device)

        c_codec = torch.zeros(s_codec_tensor.size(0), s_codec_tensor.size(1), 7)
        batch = {
            'p_codec': torch.zeros(s_codec_tensor.shape[0], s_codec_tensor.shape[1], 5), # (B, T, F)
            's_codec': s_codec_tensor.to(device),  
            'c_codec': c_codec.to(device)   
        }
        
        p_codec_pred = model.inference_one(batch)

    # connect the codecs back into one for inferencing
    accumulated_preds = np.zeros((len(s_codec)+cfg.seg_len, 5), dtype=np.float32)

    start_idx = 0
    for pred in p_codec_pred:
        accumulated_preds[start_idx:start_idx+cfg.seg_len] += pred[0]
        if start_idx != 0:
            accumulated_preds[start_idx:start_idx+cfg.overlap] /= 2
        start_idx += cfg.seg_len - cfg.overlap

    batch['p_codec'] = accumulated_preds[:len(s_codec)]
    # batch['p_codec'] = np.concatenate(p_codec_pred[:, 0, ...], axis=0)[:len(s_codec)]
    batch['s_codec'] = s_codec

    save_root = 'inference_out'
    renderer = Renderer(save_root, batch['p_codec'])

    renderer.render_inference_sample(score, output_path=cfg.output_path)


if __name__ == "__main__":
    main()
