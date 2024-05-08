import os, sys, glob, copy
import torch
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt
import parangonar as pa
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
from utils import *



class Renderer():
    """Renderer class that takes in one codec sample and convert to MIDI, and analyze.

    - sampled_pcodec:  (B, L, 5)
    - source_data: dictionary contains {score, snote_id, piece_name...}
    - label_data: same with source data

    """
    def __init__(self, save_root,
                 sampled_pcodec=None,  
                 source_data=None, label_data=None,
                 with_source=False,
                 means=None, stds=None,
                 idx=0, B=16, piece_name=""):
        
        self.save_root = save_root
        self.sampled_pcodec = sampled_pcodec
        self.source_data = source_data
        self.label_data = label_data
        self.with_source = with_source
        self.idx = idx
        self.B = B
        self.piece_name = piece_name
        self.success = True
        self.gt_id = 0

    def load_external_performances(self, performance_path, score_path, snote_ids, 
                                   label_performance_path=None, piece_name=None, save_seg=False, 
                                   merge_tracks=False, external_align=False):
        """load the performance that's already generated (for evaluating other models)"""

        self.performed_part = pt.load_performance(performance_path, merge_tracks=merge_tracks).performedparts[0]
        self.score = pt.load_score(score_path)
        self.snote_ids = snote_ids
        self.piece_name = piece_name
        # unfold the score if necessary (mostly for ASAP)
        if ("-" in self.snote_ids[0] and 
            "-" not in self.score.note_array()['id'][0]):
            self.score = pt.score.unfold_part_maximal(pt.score.merge_parts(self.score.parts)) 
        
        if label_performance_path:
            # labels are saved from training targets so they are already perfectly aligned.
            self.performed_part_label = pt.load_performance(label_performance_path).performedparts[0]
            self.gt_id = label_performance_path.split("/")[-1].split("_")[0]

        self.pnote_ids = [f"n{i}" for i in range(len(self.snote_ids))]
        self.alignment = [{'label': "match", "score_id": sid, "performance_id": pid} for sid, pid in zip(self.snote_ids, self.pnote_ids)]
        self.label_alignment = [{'label': "match", "score_id": sid, "performance_id": pid} for sid, pid in zip(self.snote_ids, self.pnote_ids)]
        if external_align:
            self.alignment = self.align_external_performances(self.score, self.performed_part)
            self.alignment = [al for al in self.alignment if ('score_id' in al and  al['score_id'] in self.snote_ids)]
        self.pcodec_pred, _, _, _ = pt.musicanalysis.encode_performance(self.score, self.performed_part, self.alignment)
        self.pcodec_label, _, _, _ = pt.musicanalysis.encode_performance(self.score, self.performed_part_label, self.label_alignment)
        self.N = min(len(self.pcodec_pred), len(self.pcodec_label))
        self.pcodec_pred, self.pcodec_label = self.pcodec_pred[:self.N], self.pcodec_label[:self.N]
        self.compare_performance_curve()
        if save_seg:
            # in the case of scoreperformer renderer, need to hear the segment since the input is the entire piece.
            performed_part = pt.musicanalysis.decode_performance(self.score, self.pcodec_pred, snote_ids=self.snote_ids)     
            os.makedirs(self.save_root, exist_ok=True)                                                                                                                                                                                             
            pt.save_performance_midi(performed_part, f"{self.save_root}/{self.idx}_{self.piece_name}.mid")


    def align_external_performances(self, score, performed_part):
        """The external performances needs to be aligned"""

        # compute note arrays from the loaded score and performance
        sna = score.note_array()
        pna = performed_part.note_array()

        # match the notes in the note arrays --------------------- DualDTWNoteMatcher
        sdm = pa.AutomaticNoteMatcher()
        pred_alignment = sdm(sna, 
                            pna,
                            verbose_time=False)

        return pred_alignment

    def render_sample(self, save_sourcelabel=False):
        """render the sample to midi file and save 
        """

        self.pcodec_pred = self.parameters_to_performance_array(self.sampled_pcodec)
        self.pcodec_label = self.parameters_to_performance_array(self.label_data['p_codec'])
        if self.with_source:
            self.pcodec_source = self.parameters_to_performance_array(self.source_data['p_codec'])

        tempo_vel_loss, tempo_vel_cor = np.inf, -1
        try:
            # load the batch information and decode into performed parts
            self.load_and_decode()

            # compute the tempo curve of sampled parameters (avg for joint-onsets)
            self.compare_performance_curve()

            # get loss and correlation metrics
            tempo_vel_loss = F.l1_loss(torch.tensor(self.performed_tempo), torch.tensor(self.label_tempo)) + \
                                    F.l1_loss(torch.tensor(self.performed_vel), torch.tensor(self.label_vel))
            tempo_vel_cor = pearsonr(self.performed_tempo, self.label_tempo)[0] + pearsonr(self.performed_vel, self.label_vel)[0]

            pt.save_performance_midi(self.performed_part, f"{self.save_root}/{self.idx}_{self.piece_name}.mid")
            if save_sourcelabel:
                pt.save_performance_midi(self.performed_part_label, f"{self.save_root}/{self.idx}_{self.piece_name}_label.mid")
                if self.with_source:
                    pt.save_performance_midi(self.performed_part_source, f"{self.save_root}/{self.idx}_{self.piece_name}_source.mid")
        except Exception as e:
            self.success = False
            print(e)

        return tempo_vel_loss, tempo_vel_cor


    def render_inference_sample(self, score, output_path):
        """render the sample to midi file and save 
        """

        self.pcodec_pred = self.parameters_to_performance_array(self.sampled_pcodec)


        self.performed_part = pt.musicanalysis.decode_performance(score, self.pcodec_pred)
        pt.save_performance_midi(self.performed_part, output_path)
        return 



    def load_and_decode(self):
        """load the meta information (scores, snote_ids and piece name) then decode the p_codecs into performed parts  
        load into module:
            - self.performed_part
            - self.performed_part_label
            - self.performed_part_source
            - snote_ids
            - score
            - piece_name
         """
        # update the snote_id_path to the new one
        snote_id_path = self.label_data['snote_id_path']
        self.snote_ids = np.load(snote_id_path)
        self.pnote_ids = self.snote_ids

        if len(self.snote_ids) < 10: # when there is too few notes, the rendering would have problems.
            raise RuntimeError("snote_ids too short")
        self.score = pt.load_musicxml(self.label_data['score_path'], force_note_ids='keep')
        # unfold the score if necessary (mostly for ASAP)
        if ("-" in self.snote_ids[0] and 
            "-" not in self.score.note_array()['id'][0]):
            self.score = pt.score.unfold_part_maximal(pt.score.merge_parts(self.score.parts)) 
        self.piece_name = self.label_data['piece_name']
        self.N = len(self.snote_ids)
        
        pad_mask = np.full(self.snote_ids.shape, False)
        self.performed_part = pt.musicanalysis.decode_performance(self.score, self.pcodec_pred[:self.N], snote_ids=self.snote_ids, pad_mask=pad_mask)
        self.performed_part_label = pt.musicanalysis.decode_performance(self.score, self.pcodec_label[:self.N], snote_ids=self.snote_ids, pad_mask=pad_mask)
        assert(len(self.performed_part_label.note_array()) == 200)
        if self.with_source:
            self.performed_part_source = pt.musicanalysis.decode_performance(self.score, self.pcodec_source[:self.N], snote_ids=self.snote_ids, pad_mask=pad_mask)
            assert(len(self.performed_part_source.note_array()) == 200)


    def compare_performance_curve(self):
        """compute the performance curve (tempo curve \ velocity curve) from given performance array
            pcodec_original: another parameter curve, coming from the optional starting point of transfer

        Returns: (load into module)
            - onset_beats
            - performed_tempo
            - label_tempo
            - source_tempo
            - performed_vel
            - label_vel
            - source_vel 
        """
        na = self.score.note_array()
        na = na[np.in1d(na['id'], self.snote_ids)]

        pcodecs = [self.pcodec_pred, self.pcodec_label]
        if self.with_source:
            pcodecs.append(self.pcodec_source)

        self.onset_beats = np.unique(na['onset_beat'])
        res = [self.onset_beats]
        for pcodec in pcodecs:
            N = min(len(na), len(pcodec))
            joint_pcodec = rfn.merge_arrays([na[:N], pcodec[:N]], flatten = True, usemask = False)
            bp = [joint_pcodec[joint_pcodec['onset_beat'] == ob]['beat_period'].mean() for ob in self.onset_beats]
            vel = [joint_pcodec[joint_pcodec['onset_beat'] == ob]['velocity'].mean() for ob in self.onset_beats]
            tempo_curve_pred = interp1d(self.onset_beats, 60 / np.array(bp))
            tempo_curve, velocity_curve = 60 / np.array(bp), np.array(vel)
            if np.isinf(tempo_curve).any():
                raise RuntimeError("inf in tempo")
            res.extend([tempo_curve, velocity_curve])

        if self.with_source:
            _, self.performed_tempo, self.performed_vel, self.label_tempo, self.label_vel, self.source_tempo, self.source_vel = res
            self.tv_source_feats = pd.DataFrame({"onset_beats": self.onset_beats, "performed_tempo": self.source_tempo, "performed_vel": self.source_vel})
        else:
            _, self.performed_tempo, self.performed_vel, self.label_tempo, self.label_vel = res

        self.tv_feats = pd.DataFrame({"onset_beats": self.onset_beats, "performed_tempo": self.performed_tempo, "performed_vel": self.performed_vel})
        self.tv_label_feats = pd.DataFrame({"onset_beats": self.onset_beats, "performed_tempo": self.label_tempo, "performed_vel": self.label_vel})


    def plot_curves(self, ax):
        ax.flatten()[self.idx].plot(self.onset_beats, self.performed_tempo, label="performed_tempo")
        ax.flatten()[self.idx].plot(self.onset_beats, self.label_tempo, label="label_tempo")
        ax.flatten()[self.idx].set_ylim(0, 300)

        ax.flatten()[self.idx+self.B].plot(self.onset_beats, self.performed_vel, label="performed_vel")
        ax.flatten()[self.idx+self.B].plot(self.onset_beats, self.label_vel, label="label_vel")

        if self.with_source:
            ax.flatten()[self.idx].plot(self.onset_beats, self.source_tempo, label="source_tempo")
            ax.flatten()[self.idx+self.B].plot(self.onset_beats, self.source_vel, label="source_vel")

        ax.flatten()[self.idx].legend()
        ax.flatten()[self.idx].set_title(f"tempo: {self.piece_name}")   
        ax.flatten()[self.idx+self.B].legend()
        ax.flatten()[self.idx+self.B].set_title(f"vel: {self.piece_name}")


    def parameters_to_performance_array(self, parameters):
        """
            parameters (np.ndarray) : shape (B, N, 5)
        """
        parameters = list(zip(*parameters.T))
        performance_array = np.array(parameters, 
                            dtype=[("beat_period", "f4"), ("velocity", "f4"), ("timing", "f4"), ("articulation_log", "f4"), ("pedal", "f4")])

        # performance_array['timing'] = 0

        # performance_array['beat_period'] = np.convolve(performance_array['beat_period'], np.ones(10)/10, mode='same')

        return performance_array



    def save_performance_features(self, save_source=False, save_label=False):
        '''
        For codec attributes:
            - Tempo and velocity deviation
            - Tempo and velocity correlation
        
        For performance features:
            - Deviation of each parameter on note level (for articulation: drop the ones with mask?)
            - Distribution difference of each parameters using KL estimation.
        '''
        alignment = [{'label': "match", "score_id": sid, "performance_id": pid} for sid, pid in zip(self.snote_ids, self.pnote_ids)]

        self.feats_pred, self.res = pt.musicanalysis.compute_performance_features(self.score, self.performed_part, alignment, feature_functions='all')        
        pd.DataFrame(self.feats_pred).to_csv(f"{self.save_root}/{self.idx}_{self.piece_name}_feats_pred.csv", index=False)
        self.tv_feats.to_csv(f"{self.save_root}/{self.idx}_{self.piece_name}_tv_feats.csv", index=False)
        self.feats_label, res = pt.musicanalysis.compute_performance_features(self.score, self.performed_part_label, alignment, feature_functions='all')
        self.feats_pred, self.feats_label = self.feats_pred[:self.N], self.feats_label[:self.N]

        if save_source:
            self.feats_source, res = pt.musicanalysis.compute_performance_features(self.score, self.performed_part_source, alignment, feature_functions='all')
            pd.DataFrame(self.feats_source).to_csv(f"{self.save_root}/{self.idx}_{self.piece_name}_feats_source.csv", index=False)
            self.tv_source_feats.to_csv(f"{self.save_root}/{self.idx}_{self.piece_name}_label_tv_feats.csv", index=False)

        if save_label:
            pd.DataFrame(self.feats_label).to_csv(f"{self.save_root}/{self.idx}_{self.piece_name}_feats_label.csv", index=False)
            self.tv_label_feats.to_csv(f"{self.save_root}/{self.idx}_{self.piece_name}_source_tv_feats.csv", index=False)



    def save_pf_distribution(self, pred_label=True, pred_source=False, label_source=False, gt_space=None):
        """compare distributions of the performance features, default is comparing the prediction & label

        pre-requisite:
        - features of pred and label
        - tempo curve of pred and label
        - velocity curve of pred and label
        """
        features_distribution = {}

        tempo_3, vel_3, tempo_4, vel_4 = None, None, None, None
        if pred_label:
            feat_1, feat_2, tempo_1, tempo_2, vel_1, vel_2 = self.feats_pred, self.feats_label, self.performed_tempo, self.label_tempo, self.performed_vel, self.label_vel
            name = "pred_label"
        if pred_source:
            feat_1, feat_2, tempo_1, tempo_2, vel_1, vel_2 = self.feats_pred, self.feats_source, self.performed_tempo, self.source_tempo, self.performed_vel, self.source_vel
            name = "pred_source"
        if label_source:
            feat_1, feat_2, tempo_1, tempo_2, vel_1, vel_2 = self.feats_label, self.feats_source, self.label_tempo, self.source_tempo, self.label_vel, self.source_vel
            name = "label_source"
        if type(gt_space) != type(None):
            feat_2, feat_3, label_tv, label_tv_ = pd.read_csv(f"{gt_space}/feats_pred_mean.csv"), pd.read_csv(f"{gt_space}/feats_pred_std.csv"), pd.read_csv(f"{gt_space}/tv_feats_mean.csv"), pd.read_csv(f"{gt_space}/tv_feats_std.csv")
            tempo_2, vel_2, tempo_3, vel_3 = label_tv['performed_tempo'], label_tv['performed_vel'], label_tv_['performed_tempo'], label_tv_['performed_vel']
            feat_4, label_tv = pd.read_csv(f"{gt_space}/feats_pred_all.csv"), pd.read_csv(f"{gt_space}/tv_feats_all.csv")
            tempo_4, vel_4 = label_tv['performed_tempo'], label_tv['performed_vel']
            name = "pred_GTs"     
            self.gt_id = "all"       

        for feat_name in ['articulation_feature.kor',
                        'asynchrony_feature.pitch_cor',
                        # 'asynchrony_feature.vel_cor',
                        'asynchrony_feature.delta',
                        'dynamics_feature.agreement',
                        'dynamics_feature.consistency_std',
                        'dynamics_feature.ramp_cor',
                        'dynamics_feature.tempo_cor',
                        'pedal_feature.onset_value'
                        ]:
            
            mask = np.full(self.res['no_kor_mask'].shape, False)
            if 'kor' in feat_name:
                mask = self.res['no_kor_mask']
            if type(gt_space) != type(None):
                features_distribution[feat_name] = self.dev_kl_cor_estimate(feat_1[feat_name], feat_2[feat_name], feat_3=feat_3[feat_name], feat_4=feat_4[feat_name], mask=mask)
            else:
                features_distribution[feat_name] = self.dev_kl_cor_estimate(feat_1[feat_name], feat_2[feat_name], mask=mask)

        # clip the tempo curve in range before computing deviation. 
        tempo_1, tempo_2 = np.clip(tempo_1, 15, 480), np.clip(tempo_2, 15, 480)
        features_distribution["tempo_curve"] = self.dev_kl_cor_estimate(tempo_1, tempo_2, feat_3=tempo_3, feat_4=tempo_4, mask=np.full(self.performed_tempo.shape, False))    
        features_distribution["vel_curve"] = self.dev_kl_cor_estimate(vel_1, vel_2, feat_3=vel_3, feat_4=vel_4, mask=np.full(self.performed_vel.shape, False))    
        self.features_distribution = pd.DataFrame(features_distribution)
        self.features_distribution.to_csv(f"{self.save_root}/{self.idx}_{self.piece_name}_{name}_distribution_{self.gt_id}.csv", index=False)
        


    def dev_kl_cor_estimate(self, feat_1, feat_2, feat_3=None, feat_4=None,
                        N=300, mask=None):
        """
            dev: deviation between the prediction and the target / source we want to compare with.
            KL: MC estimate KL divergence by convering the features into kde distribution, and sample their pdf
            cor: correlation between the two compared series

            mask: for the values that we don't want to look at. 
            feat_3: ground truth std in the case of looking at GT space
        """
        feat_1, feat_2 = feat_1[~mask], feat_2[~mask]
        # also mask the nan value
        nan_mask = np.isnan(feat_1)| np.isnan(feat_2)
        feat_1, feat_2 = feat_1[~nan_mask], feat_2[~nan_mask]

        try:
            kde_1 = gaussian_kde(feat_1)
            if type(feat_4) != type(None):
                kde_label = gaussian_kde(feat_4)
            else:
                kde_label = gaussian_kde(feat_2)

            pred_points = kde_1.resample(N) 
            KL = entropy(kde_1.pdf(pred_points), kde_label.pdf(pred_points))
        except Exception as e:
            KL = -1   # null value since KL can't be negative                                                                                                                                                                                                                                                      
        
        # if np.isinf(KL):
        #     hook()

        if type(feat_3) != type(None):
            # cap the deviation by -5 and 5, as the ratio is very much dependent on the value of label.
            deviation = max(min(np.ma.masked_invalid((feat_1 - feat_2) / feat_3).mean(), 5), -5)
        else:
            deviation = max(min(np.ma.masked_invalid((feat_1 - feat_2) / feat_2).mean(), 5), -5)

        return {
            "Deviation": deviation, # filter out the inf and nan
            "KL divergence": KL,
            "Correlation": pearsonr(feat_1, feat_2)[0],
        }
    
