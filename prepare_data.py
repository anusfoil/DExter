import os, sys, glob
import argparse
from collections import defaultdict
import warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt

import h5py
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm


VIENNA_MATCH_DIR = "../Datasets/vienna4x22/match/"
VIENNA_MUSICXML_DIR = "../Datasets/vienna4x22/musicxml/"
VIENNA_PERFORMANCE_DIR = "../Datasets/vienna4x22/midi/"

ASAP_DIR = "../Datasets/asap-dataset-alignment/"

BMZ_MATCH_DIR = "../Datasets/pianodata-master/match"
BMZ_MUSICXML_DIR = "../Datasets/pianodata-master/xml/"

ATEPP_DIR = "../Datasets/ATEPP-1.1"
ATEPP_META_DIR = "../Datasets/ATEPP-1.1/ATEPP-metadata-1.3.csv"

BASE_DIR = "/import/c4dm-scratch-02/DiffPerformer"



def process_dataset_codec(max_note_len, mix_up=False):
    """process the performance features for the given dataset. Save the 
    computed features in the form of hdf5 arrays in the same directory as 
    performance data.
    """

    with open('skip_data.txt', 'r') as file:
        skip_paths = file.read().splitlines()
    skip_paths = [sp.strip() for sp in skip_paths]

    splits_file = open("score_splits.csv", "a")
    splits_file.write("score_path,split\n")

    os.makedirs(f"{BASE_DIR}/snote_ids/N={max_note_len}", exist_ok=True)
    hdf5_name = f"{BASE_DIR}/codec_N={max_note_len}.hdf5"
    if mix_up:
        hdf5_name = f"{BASE_DIR}/codec_N={max_note_len}_mixup.hdf5"
    
    with h5py.File(hdf5_name, 'a') as hdf5_file:
        for dataset in [
                        'ATEPP', 
                        'ASAP', 
                        'VIENNA422'
                        ]:
            # Paths setup here...
            if dataset == "VIENNA422":
                alignment_paths = glob.glob(os.path.join(VIENNA_MATCH_DIR, "*[!e].match"))
                alignment_paths = sorted(alignment_paths)
                score_paths = [(VIENNA_MUSICXML_DIR + pp.split("/")[-1][:-10] + ".musicxml") for pp in alignment_paths]
                performance_paths = [None] * len(alignment_paths) # don't use the given performance, use the aligned.
                cep_feat_paths = [("../Datasets/vienna4x22/cep_features/" + pp.split("/")[-1][:-6] + ".csv") for pp in alignment_paths]
            if dataset == "ASAP":
                performance_paths = glob.glob(os.path.join(ASAP_DIR, "**/*[!e].mid"), recursive=True)
                alignment_paths = [(pp[:-4] + "_note_alignments/note_alignment.tsv") for pp in performance_paths]
                score_paths = [os.path.join("/".join(pp.split("/")[:-1]), "xml_score.musicxml") for pp in performance_paths]
                cep_feat_paths = [pp[:-4] + "_cep_features.csv" for pp in performance_paths]
            if dataset == "ATEPP":
                alignment_paths = glob.glob(os.path.join(ATEPP_DIR, "**/[!z]*n.csv"), recursive=True)
                alignment_paths = sorted(alignment_paths)
                performance_paths = [(aa[:-10] + ".mid") for aa in alignment_paths]
                score_paths = [glob.glob(os.path.join("/".join(pp.split("/")[:-1]), "*.*l"))[0] for idx, pp in enumerate(performance_paths)]
                cep_feat_paths = ["/".join(aa.split("/")[:-1]) + "_cep_features.csv" for aa in alignment_paths]
                atepp_overlap_dirs = get_atepp_overlap()
                atepp_overlap_dirs = [f"{ATEPP_DIR}/{ao_dir}" for ao_dir in atepp_overlap_dirs]

            # storing codecs for existing score
            same_score_p_codec, mixuped_p_codec = [], []
            same_score_c_codec, mixuped_c_codec = [], []
            
            prev_s_path = None
            for j, (s_path, p_path, a_path, c_path) in tqdm(enumerate(zip(score_paths, performance_paths, alignment_paths, cep_feat_paths))):
                
                if dataset == 'ATEPP': # ATEPP: skip the bad ones
                    alignment = pt.io.importparangonada.load_parangonada_alignment(a_path)
                    match_aligns = [a for a in alignment if a['label'] == 'match']
                    insertion_aligns = [a for a in alignment if a['label'] == 'insertion']
                    deletion_aligns = [a for a in alignment if a['label'] == 'deletion']
                    if (len(match_aligns) / (len(insertion_aligns) + len(deletion_aligns) + len(match_aligns))) < 0.5:
                        continue
                # path to save the reproducing artifacts
                if dataset == "VIENNA422":
                    piece_name = s_path.split("/")[-1].split(".")[0]
                if dataset == "ASAP":
                    piece_name = "_".join(s_path.split("alignment/")[-1].split("/")[:-1])
                if dataset == 'ATEPP':
                    piece_name = p_path.split("/")[-1][:-4] 
                    perf_name = p_path.split("/")[-1][:-4] 
                save_snote_id_path = f"{BASE_DIR}/snote_ids/N={max_note_len}/{dataset}_{piece_name}"


                # Check if any of the provided paths should be skipped (file can't load)
                if (s_path in skip_paths) or (a_path in skip_paths) or (p_path in skip_paths) or not ((os.path.exists(s_path) and os.path.exists(a_path))):
                    continue
                if f"{dataset}_{j}" in hdf5_file: # or already computed
                    print(f"Data for {dataset}_{j} already computed. Skipping...")
                    continue
                
                data = defaultdict(list)

                # depending on whether we are still processing the same composition, do mixup
                reuse_score = (prev_s_path == s_path)

                p_codec, c_codec, snote_ids, score = get_codecs(s_path, a_path, c_path, 
                                                                performance_path=p_path, score=(score if reuse_score else None))
                
                if 60 / p_codec[:, 0].mean() > 200: # Tempo filter check.
                    grp = hdf5_file.create_group(f"{dataset}_{j}")
                    grp.create_dataset('error', data='error')
                    continue

                if not reuse_score:
                    same_score_p_codec, mixuped_p_codec = [], []
                    same_score_c_codec, mixuped_c_codec = [], []
                    split = np.random.choice(['train', 'test'], p=[0.85, 0.15])
                    splits_file.write(f"{s_path},{split}\n")
                else:
                    if mix_up and ((dataset == "VIENNA422") or (dataset == 'ATEPP' and "/".join(s_path.split("/")[:-1]) in atepp_overlap_dirs)):
                        mixuped_p_codec = [np.mean(np.array([p_codec, ss_p_codec]), axis=0) for ss_p_codec in same_score_p_codec]
                        same_score_p_codec.append(p_codec)
                        mixuped_c_codec = [np.mean(np.array([c_codec, ss_c_codec]), axis=0) for ss_c_codec in same_score_c_codec]
                        same_score_c_codec.append(c_codec)

                sna = score.note_array()
                sna = sna[np.in1d(sna['id'], snote_ids)]
                s_codec = rfn.structured_to_unstructured(
                    sna[['onset_div', 'duration_div', 'pitch', 'voice']])

                if not ((len(p_codec) == len(s_codec)) and (len(p_codec) == len(snote_ids))):
                    print(f"{a_path} has length issue: p: {len(p_codec)}; s: {len(s_codec)}") 
                    grp = hdf5_file.create_group(f"{dataset}_{j}")
                    grp.create_dataset('error', data='error')
                    continue

                for i, (p_codec, c_codec) in enumerate(zip(([p_codec] + mixuped_p_codec), ([c_codec] + mixuped_c_codec))): # segmentation
                    if i == 1:
                        piece_name = piece_name + "_mixup"  # mixuped codec name

                    for idx in range(0, len(p_codec), max_note_len): # segment the piece 
                        seg_p_codec = p_codec[idx : idx + max_note_len]
                        seg_s_codec = s_codec[idx : idx + max_note_len]
                        seg_c_codec = c_codec[idx : idx + max_note_len]
                        seg_snote_ids = snote_ids[idx : idx + max_note_len]

                        if len(seg_p_codec) < max_note_len:
                            seg_p_codec = np.pad(seg_p_codec, ((0, max_note_len - len(seg_p_codec)), (0, 0)), mode='constant', constant_values=0)
                            seg_s_codec = np.pad(seg_s_codec, ((0, max_note_len - len(seg_s_codec)), (0, 0)), mode='constant', constant_values=0)
                            seg_c_codec = np.pad(seg_c_codec, ((0, max_note_len - len(seg_c_codec)), (0, 0)), mode='constant', constant_values=0)

                        if len(seg_snote_ids) == 0:
                            hook()

                        seg_id_path = f"{save_snote_id_path}_seg{int(idx/max_note_len)}.npy"
                        # save snote_id
                        np.save(seg_id_path, seg_snote_ids) 

                        data['p_codec'].append(seg_p_codec)
                        data['s_codec'].append(seg_s_codec)
                        data['c_codec'].append(seg_c_codec)
                        data['snote_id_path'].append(seg_id_path)
                        data['score_path'].append(s_path)
                        data['piece_name'].append(piece_name)  # piece name for shortcut and identifying the generated sample. unique for performance not composition
                        data['split'].append(split)
                
                # Save the data for this piece
                grp = hdf5_file.create_group(f"{dataset}_{j}")
                grp.create_dataset('p_codec', data=data["p_codec"])
                grp.create_dataset('s_codec', data=data["s_codec"])
                grp.create_dataset('c_codec', data=data["c_codec"])
                grp.create_dataset('snote_id_path', data=data["snote_id_path"])
                grp.create_dataset('score_path', data=data["score_path"])
                grp.create_dataset('piece_name', data=data["piece_name"])
                grp.create_dataset('split', data=data["split"], dtype=h5py.string_dtype(encoding='utf-8'))

                # Keep track of the last processed score path
                prev_s_path = s_path
                
    print("Processing and data saving complete.")




def load_data_from_hdf5(hdf5_path):
    train_data, test_data = [], []
    print('Loading data from HDF5 file...')

    # Open the HDF5 file for reading
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # Iterate over groups in HDF5 file
        for group_name in tqdm(hdf5_file):
            group = hdf5_file[group_name]
            
            # Skip groups with 'error'
            if 'error' in group:
                continue
            
            # Pre-load datasets into memory once
            p_codec = np.array(group['p_codec'])
            s_codec = np.array(group['s_codec'])
            # c_codec = np.ones(group['c_codec'].shape)
            c_codec = np.array(group['c_codec'])
            snote_id_path = np.array(group['snote_id_path'])
            score_path = np.array(group['score_path'])
            piece_name = np.array(group['piece_name'])
            split = np.array(group['split'])

            # Collect all the pieces' data in a list
            piece_data = [{
                'p_codec': p_codec[i],
                's_codec': s_codec[i],
                'c_codec': c_codec[i],
                'snote_id_path': snote_id_path[i],
                'score_path': score_path[i],
                'piece_name': piece_name[i].decode(),
                'split': split[i].decode()
            } for i in range(len(p_codec))]
                        
            # Check split and extend data
            if (piece_data[0]['split'] == 'train'):   
                train_data.extend(piece_data)
            else:
                test_data.extend([pd for pd in piece_data if "mixup" not in pd['piece_name']])
                

    return train_data, test_data

def get_codecs(score_path, alignment_path, c_path, performance_path=None, score=None):
    """compute the performance feature given score, alignment and performance path.
    """

    if alignment_path[-5:] == "match":
        performance, alignment = pt.load_match(alignment_path)
    elif alignment_path[-3:] == "tsv":
        alignment = pt.io.importparangonada.load_alignment_from_ASAP(alignment_path)
    elif alignment_path[-3:] == "csv": # case for ATEPP
        alignment = pt.io.importparangonada.load_parangonada_alignment(alignment_path)
        if isinstance(score, type(None)):
            score = pt.load_musicxml(score_path, force_note_ids='keep')

    if isinstance(score, type(None)):
        score = pt.load_musicxml(score_path)

    # if doesn't match the note id in alignment, unfold the score.
    if (('score_id' in alignment[0]) 
        and ("-" in alignment[0]['score_id'])
        and ("-" not in score.note_array(include_divs_per_quarter=False)['id'][0])): 
        score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 

    if not isinstance(performance_path, type(None)): # use the performance if it's given
        performance = pt.load_performance(performance_path)

    # get the performance encodings
    parameters, snote_ids, pad_mask, m_score = pt.musicanalysis.encode_performance(score, performance, alignment, 
                                                                        #   tempo_smooth='derivative'
                                                                          )

    p_codec = rfn.structured_to_unstructured(parameters)
    
    # Check if c_path exists and compute c_codec, otherwise return None.
    if os.path.exists(c_path):
        cep_feats = pd.read_csv(c_path)
        c_codec = rfn.structured_to_unstructured(get_cep_codec(cep_feats, m_score))
    else:
        print(f"No cep_features for {alignment_path}")
        c_codec = np.zeros((len(p_codec), 7))
        # return None, None, None, None  # Early exit if c_codec cannot be computed.

    return p_codec, c_codec, snote_ids, score


def get_cep_codec(cep_feats, m_score):
    """ align the perceptual features with the performance, return
        an array in the same size of p_codec (structured array)
    """
    c_codec = pd.DataFrame()
    for row in m_score:
        next_window = cep_feats[cep_feats['frame_start_time'] <= row['p_onset']]
        if not len(next_window):
            c_codec = c_codec.append(cep_feats.iloc[-1])
        else:
            c_codec = c_codec.append(next_window.iloc[-1])

    records = c_codec.drop(columns=['frame_start_time']).to_records(index=False)
    c_codec = np.array(records, dtype = records.dtype.descr)
    return c_codec


def get_atepp_overlap(): 
    """get the atepp subset with pieces of more than 8+ performances
    Returns: 
    """
    atepp_meta = pd.read_csv(ATEPP_META_DIR)
    score_groups = atepp_meta.groupby(['score_path']).count().sort_values(['midi_path'], ascending=False)
    selected_scores = score_groups.iloc[:]
    selected_score_folders = ["/".join(score_entry.name.split("/")[:-1]) for _, score_entry in selected_scores.iterrows()]

    return selected_score_folders


def render_sample(score_part, sample_path, snote_ids_path):
    """render """
    snote_ids = np.load(snote_ids_path)
    for idx in range(32):
        performance_array = reverse_quantized_codec(np.load(sample_path)[idx])
        performed_part = pt.musicanalysis.decode_performance(score_part, performance_array, snote_ids=snote_ids)

        pt.save_performance_midi(performed_part, f"samples/sample_{idx}.mid")

    return performed_part


def codec_data_analysis():
    # look at the distribution of the bp, velocity...
    codec_data = np.load(f"{BASE_DIR}/codec_N=200.npy", allow_pickle=True) # (N_data, 1000, 4)
    p_codecs = [cd['p_codec'] for cd in codec_data]

    avg_tempo = [60 / pc[:, 0].mean() for pc in p_codecs]
    avg_tempo = [a for a in avg_tempo if a <= 500]
    plt.hist(avg_tempo, bins=100)
    plt.xlim((0, 500))
    plt.savefig("tmp.png")

    """
        - N=200 101,947 pieces of data. 
        - there are 673 with an avg_tempo > 1000 !! 2k with avg_tempo > 500. 75% of the data are under 200bpm 

    """

def plot_codec(data, ax0, ax1, ax2, fig):
    # plot the p_codec and s_codec, on the given two axes

    p_im = ax0.imshow(data["p_codec"].T, aspect='auto', origin='lower')
    ax0.set_yticks([0, 1, 2, 3, 4])
    ax0.set_yticklabels(["beat_period", "velocity", "timing", "articulation_log", 'pedal'])
    fig.colorbar(p_im, orientation='vertical', ax=ax0)
    ax0.set_title(data['snote_id_path'])

    s_im = ax1.imshow(data['s_codec'].T, aspect='auto', origin='lower')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['onset_div', 'duration_div', 'pitch', 'voice'])
    fig.colorbar(s_im, orientation='vertical', ax=ax1)
    
    c_im = ax2.imshow(data['c_codec'].T, aspect='auto', origin='lower')
    ax2.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax2.set_yticklabels(['melodiousness', 'articulation', 'rhythm_complexity', 
                         'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness'])
    fig.colorbar(c_im, orientation='vertical', ax=ax2)

    return 


def plot_codec_list(codec_list):

    n_data = len(codec_list)
    fig, ax = plt.subplots(3 * n_data, 1, figsize=(24, 4 * n_data))
    for idx, data in enumerate(codec_list):
        plot_codec(data, ax[idx * 3], ax[idx * 3 + 1], ax[idx * 3 + 2], fig)

    plt.savefig("tmp.png")

    return fig

def match_midlevels():
    mid_paths = glob.glob("../Datasets/midlevel_2bf74_2/**/*.csv", recursive=True)
    for mp in mid_paths:
        newdir = "/".join(mp.split("/")[3:])[:-4]
        os.system(f"mv {mp} ../Datasets/asap-dataset-alignment/{newdir}_cep_features.csv")
    return 


def make_transfer_pair(codec_data, K=50000, N=200):
    """make transfer pair from the codec data for the testing set. 
        Transfer data come from real performance (not mix-up combinations), and the pieces used in testing set doesn't go into training. 

        returns:
        - transfer_pairs: list of tuple of codec
        - unpaired: list of single codec

        stats:
        - in total 2M pairs can be found. In testing we can use ~1000 pairs (K). rest can go into unpaired for training. For full transfer 
            training, we use full pairs. ()
    """
    if os.path.exists(f"{BASE_DIR}/codec_N={N}_mixup_paired_K={K}.npy"):
        transfer_pairs = np.load(f"{BASE_DIR}/codec_N={N}_mixup_paired_K={K}.npy", allow_pickle=True)
        unpaired = np.load(f"{BASE_DIR}/codec_N={N}_mixup_unpaired_K={K}.npy", allow_pickle=True)
        return transfer_pairs, unpaired

    transfer_pairs = []

    codec_data = np.array(codec_data)
    codec_data_ = codec_data[list(map(lambda x: "mu" not in x['piece_name'], codec_data))] # only consider those not in mixup
    np.random.shuffle(codec_data_)
    unpaired = codec_data

    for cd in tqdm(codec_data_):
        if len(transfer_pairs) > K:
            break
        seg_id = cd['snote_id_path'].split("seg")[-1].split(".")[0]
        # find the one that belongs to the same piece, same segment number, but not itself
        mask = list(map(lambda x: ((
                                    x['score_path'] == cd['score_path']) 
                                   and (f"seg{seg_id}." in x['snote_id_path']) 
                                   and (x['p_codec'] != cd['p_codec']).any() 
                                   ), codec_data_))
        same_piece_cd = codec_data_[mask]
        if len(same_piece_cd):
            for scd in same_piece_cd:
                transfer_pairs.extend([cd, scd])
            # remove all this piece's segments from unpaired list (mixup as well, to prevent leakage)
            unpaired = unpaired[list(map(lambda x: x['score_path'] != cd['score_path'], unpaired))]


    # shuffle by each pair and recover
    transfer_pairs = np.array(transfer_pairs).reshape(2, -1, order='F')
    np.random.shuffle(transfer_pairs.T) 
    transfer_pairs = transfer_pairs.ravel(order='F')
    
    np.save(f"{BASE_DIR}/codec_N={N}_mixup_test_paired_K={K}.npy", transfer_pairs)
    np.save(f"{BASE_DIR}/codec_N={N}_mixup_test_unpaired_K={K}.npy", unpaired)
    
    return transfer_pairs, unpaired



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_codec', action='store_true')
    parser.add_argument('--pairing', action='store_true')
    parser.add_argument('--MAX_NOTE_LEN', type=int, default=200, required=False)
    parser.add_argument('--K', type=int, default=2000, required=False)
    args = parser.parse_args()

    if args.compute_codec:
        process_dataset_codec(args.MAX_NOTE_LEN, mix_up=True)
        hook()
    elif args.pairing:
        codec_data = np.load(f"{BASE_DIR}/codec_N={args.MAX_NOTE_LEN}_mixup_test.npy", allow_pickle=True) 
        transfer_pairs, unpaired = make_transfer_pair(codec_data, K=args.K, N=args.MAX_NOTE_LEN)
    # codec_data_analysis()
    
    # plot_codec_list(codec_data[:1])

    # for data in codec_data:
    #     if '11579_seg2' in data['snote_id_path']:
    #         score = pt.load_musicxml(data['score_path'], force_note_ids='keep')
    #         # score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
    #         snote_ids = np.load(data['snote_id_path'])
    #         performed_part = pt.musicanalysis.decode_performance(score, parameters_to_performance_array(data['p_codec']), snote_ids=snote_ids)
    #         pt.save_performance_midi(performed_part, "tmp0.mid")
    #         hook()

    # score = pt.load_musicxml("../Datasets/vienna4x22/musicxml/Schubert_D783_no15.musicxml")
    # performance = pt.load_performance("../Datasets/vienna4x22/midi/Schubert_D783_no15_p01.mid")
    # _, alignment = pt.load_match("../Datasets/vienna4x22/match/Schubert_D783_no15_p01.match")
    # parameters, snote_ids, _ = pt.musicanalysis.encode_performance(score, performance, alignment)
    # performed_part = pt.musicanalysis.decode_performance(score, parameters, snote_ids=snote_ids)
    # pt.save_performance_midi(performed_part, "tmp0.mid")
    # score_part = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
    # performed_part = render_sample(score_part, "logs/log_conv_transformer_melody_156/samples/samples_4000.npz.npy", "tmp.npy")
