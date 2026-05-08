import os
import warnings
warnings.simplefilter("ignore")
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

    score = pt.load_musicxml(cfg.score_path, force_note_ids='keep')

    # Build s_codec at the width the loaded checkpoint expects.
    # If cfg.include_xml_features is set, extract the 15 xml-derived columns
    # (articulation / slur / dynamic / wedge progress / expected_bpm / pedal /
    # ...) so v2 inference matches the training feature layout. cfg.target_bpm,
    # if set, overrides the score-derived expected_bpm column with the user's
    # requested anchor — model treats the new anchor as score conditioning, so
    # articulation and timing decisions are made *consistent with* that tempo
    # (not just a post-hoc rescale of the predicted curve).
    sna = score.parts[0].note_array(
        include_pitch_spelling=True,
        include_key_signature=True,
        include_time_signature=True,
        include_metrical_position=True,
        include_grace_notes=True,
        include_staff=True,
        include_divs_per_quarter=True,
    )
    base = rfn.structured_to_unstructured(
        sna[['onset_div', 'duration_div', 'pitch', 'voice']]
    ).astype(np.float32)

    if cfg.get("include_xml_features", False):
        from features.xml_features import extract_xml_features, FEATURE_SPEC
        feats = extract_xml_features(score, sna)
        xml_cols = np.stack(
            [feats[name].astype(np.float32) for name, _, _ in FEATURE_SPEC],
            axis=-1,
        )
        if cfg.get("target_bpm"):
            ebpm_idx = next(i for i, (n, _, _) in enumerate(FEATURE_SPEC) if n == "expected_bpm")
            print(f"[inference] override expected_bpm column → {cfg.target_bpm} BPM")
            xml_cols[:, ebpm_idx] = float(cfg.target_bpm)
        s_codec = torch.tensor(np.concatenate([base, xml_cols], axis=-1))
    else:
        s_codec = torch.tensor(base)

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

        # AR-context aware chunking. When the loaded checkpoint was trained
        # with enable_prior_context=True, the model expects an extra
        # (F_p + 1)-channel "prior_features" input that holds the previous
        # chunk's last `overlap` predicted notes (plus a 0/1 mask). For
        # chunk 0 this is all zeros (cold start). Each subsequent chunk
        # depends on the previous one, so we sample sequentially rather
        # than batch all chunks together as the v1 path did.
        ar_context = bool(cfg.get("enable_prior_context", False))
        F_p = 5
        F_c = 7
        n_chunks, T, _ = s_codec_tensor.shape
        chunk_preds: list[np.ndarray] = []

        for i in tqdm(range(n_chunks), desc="AR chunks" if ar_context else "chunks"):
            chunk_s = s_codec_tensor[i:i+1].to(device)            # (1, T, F_s)
            batch = {
                'p_codec': torch.zeros(1, T, F_p, device=device),
                's_codec': chunk_s,
                'c_codec': torch.zeros(1, T, F_c, device=device),
            }
            if ar_context:
                prior = torch.zeros(1, T, F_p + 1, device=device)
                if i > 0 and chunk_preds:
                    overlap = cfg.overlap
                    prev = torch.tensor(chunk_preds[-1][0], device=device, dtype=torch.float32)
                    prior[0, :overlap, :F_p] = prev[-overlap:]
                    prior[0, :overlap, F_p:] = 1.0
                batch['prior_features'] = prior

            pred = model.inference_one(batch)                     # (1, 1, T, F_p)
            chunk_preds.append(pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred)

    # Stitch chunks back together with overlap-averaging.
    accumulated_preds = np.zeros((len(s_codec)+cfg.seg_len, 5), dtype=np.float32)
    start_idx = 0
    for pred in chunk_preds:
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
