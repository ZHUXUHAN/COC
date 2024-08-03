import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
import math
from math import pi
from .model_motifs import LSTMContext, FrequencyBias
from .model_vctree import VCTreeLSTMContext
from .model_runet import RUNetContext, Boxes_Encode
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from .utils_motifs import obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info

from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling.utils import cat


@registry.ROI_RELATION_PREDICTOR.register("COC_Predictor")
class COC_Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(COC_Predictor, self).__init__()

        self.config = config
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        self.context_type = config.MODEL.ROI_RELATION_HEAD.COC.CONTEXT_TYPE
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.context_type == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        self.embed_dim = 300
        self.mlp_dim = 2048

        dropout_p = 0.2
        self.rel_compress = nn.Parameter(torch.FloatTensor(self.num_rel_cls, self.pooling_dim + 64))
        nn.init.xavier_uniform_(self.rel_compress)
        self.linear_rel_rep = nn.Linear(self.pooling_dim + 64, self.pooling_dim + 64)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        self.norm_rel_rep = nn.LayerNorm(self.pooling_dim + 64)
        self.project_head_1 = MLP(self.pooling_dim + 64, self.mlp_dim, self.pooling_dim + 64, 2)
        self.project_head_2 = MLP(self.pooling_dim + 64, self.mlp_dim, self.pooling_dim + 64, 2)
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.post_emb_s = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_s, xavier=True)
        self.post_emb_o = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_o, xavier=True)
        self.merge_obj_low = nn.Linear(self.pooling_dim + 5 + 200, self.pooling_dim)
        layer_init(self.merge_obj_low, xavier=True)
        self.merge_obj_high = nn.Linear(self.hidden_dim, self.pooling_dim)
        layer_init(self.merge_obj_high, xavier=True)

        self.post_emb_s_1 = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_s_1, xavier=True)
        self.post_emb_o_1 = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_o_1, xavier=True)
        self.merge_obj_low_1 = nn.Linear(self.pooling_dim + 5 + 200, self.pooling_dim)
        layer_init(self.merge_obj_low_1, xavier=True)
        self.merge_obj_high_1 = nn.Linear(self.hidden_dim, self.pooling_dim)
        layer_init(self.merge_obj_high_1, xavier=True)

        self.get_boxes_encode = Boxes_Encode()
        self.ort_embedding = nn.Parameter(self.get_ort_embeds(self.num_obj_cls, 200), requires_grad=False)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.freq_bias = FrequencyBias(config, statistics)

    def get_ort_embeds(self, k, dims):
        ind = torch.arange(1, k + 1).float().unsqueeze(1).repeat(1, dims)
        lin_space = torch.linspace(-pi, pi, dims).unsqueeze(0).repeat(k, 1)
        t = ind * lin_space
        return torch.sin(t) + torch.cos(t)

    def get_relation_embedding(self, rel_pair_idxs, obj_preds, roi_features, union_features,
                               obj_preds_embeds, proposals, edge_ctxs):

        prod_reps = []
        pair_preds = []
        spt_feats = []

        for rel_pair_idx, obj_pred, roi_feat, union_feat, obj_embed, bboxes, edge_ctx_i in zip(
                rel_pair_idxs, obj_preds, roi_features, union_features,
                obj_preds_embeds, proposals, edge_ctxs):
            if torch.numel(rel_pair_idx) == 0:
                continue
            w, h = bboxes.size
            bboxes_tensor = bboxes.bbox
            transfered_boxes = torch.stack(
                (
                    bboxes_tensor[:, 0] / w,
                    bboxes_tensor[:, 3] / h,
                    bboxes_tensor[:, 2] / w,
                    bboxes_tensor[:, 1] / h,
                    (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]) * \
                    (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) / w / h,
                ), dim=-1
            )
            obj_features_low = cat(
                (
                    roi_feat, obj_embed, transfered_boxes
                ), dim=-1
            )

            obj_features = self.merge_obj_low(obj_features_low) + self.merge_obj_high(edge_ctx_i)

            subj_rep, obj_rep = self.post_emb_s(obj_features), self.post_emb_o(obj_features)
            assert torch.numel(rel_pair_idx) > 0

            spt_feats.append(self.get_boxes_encode(bboxes_tensor, rel_pair_idx, w, h))
            prod_reps.append(subj_rep[rel_pair_idx[:, 0]] * obj_rep[rel_pair_idx[:, 1]] * union_feat)
            pair_preds.append(torch.stack((obj_pred[rel_pair_idx[:, 0]], obj_pred[rel_pair_idx[:, 1]]), dim=1))

        prod_reps = cat(prod_reps, dim=0)
        pair_preds = cat(pair_preds, dim=0)
        spt_feats = cat(spt_feats, dim=0)

        prod_reps = cat((prod_reps, spt_feats), dim=-1)

        # mlp layer to extract the triplet feature
        # x1 = LN(Dropout(Relu(LN(Linear(x))))) + x
        # MLP(Dropout(Relu(LN(x1))))
        triplet_feats = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(prod_reps))) + prod_reps)
        triplet_feats = self.project_head_1(self.dropout_rel(torch.relu(triplet_feats)))

        return triplet_feats, pair_preds

    def get_rel_dist_norm_fc(self, x):
        # W = MLP(initial random parameters)
        w = self.project_head_2(self.dropout_pred(torch.relu(self.rel_compress)))

        x_norm = F.normalize(x)
        w_norm = F.normalize(w)

        rel_dists = self.logit_scale.exp() * x_norm @ w_norm.T
        return x, w, x_norm, w_norm, rel_dists

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        add_losses = {}
        if self.context_type == 'Motifs':
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        obj_preds_embeds = self.ort_embedding.index_select(0, obj_preds.long())

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]

        union_features = union_features.split(num_rels, dim=0)
        roi_features = roi_features.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_preds_embeds = obj_preds_embeds.split(num_objs, dim=0)
        edge_ctxs = edge_ctx.split(num_objs, dim=0)

        triplet_feats, pair_preds = self.get_relation_embedding(rel_pair_idxs, obj_preds, roi_features, union_features, obj_preds_embeds, proposals, edge_ctxs)

        x, w, x_norm, w_norm, rel_dists = self.get_rel_dist_norm_fc(triplet_feats)

        if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_preds)

        rel_dists = rel_dists.split(num_rels, dim=0)
        obj_dists = obj_dists.split(num_objs, dim=0)

        return obj_dists, rel_dists, add_losses


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

