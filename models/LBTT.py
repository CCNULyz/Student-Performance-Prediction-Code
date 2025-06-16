import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import LBTTEncoderLayer, LBTTEncoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PatchEmbedding
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, configs, patch_len=8, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Non-overlap Patches
        padding = stride

        # 学习过程 Embedding
        self.enc_pro_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,
                                                    configs.dropout)
        # 学习行为 Embedding
        self.enc_beh_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,
                                                    configs.dropout)

        # Patch Embedding  Non-overlap Patch
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = LBTTEncoder(
            [
                LBTTEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2 + 1)

        # Decoder
        # self.act = F.gelu
        self.flatten = nn.Flatten(start_dim=2)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        # 构造学习过程 Input
        # Embedding
        enc_process_out = self.enc_pro_embedding(x_enc, None)
        # 与 enc_patch_out 统一维度
        enc_process_out_reshape = torch.reshape(
            enc_process_out, (enc_process_out.shape[0] * enc_process_out.shape[1], 1, enc_process_out.shape[-1]))

        # do patching and embedding
        x_enc_patch = x_enc.permute(0, 2, 1)

        # u: [bs * nvars x patch_num x d_model]
        enc_patch_out, n_vars = self.patch_embedding(x_enc_patch)

        process_input = torch.concatenate([enc_patch_out, enc_process_out_reshape], dim=-2)

        # 构造 学习行为 Input
        enc_behavior_out = self.enc_beh_embedding(x_enc, None)
        behavior_input = enc_behavior_out

        # Encoder
        # Patch-wise Self-Attention
        # z: [bs * nvars x patch_num x d_model]
        cross_out_merge, cross_out, attn1, attn2 = self.encoder(process_input, behavior_input, n_vars)
        cross_out_merge = torch.reshape(
            cross_out_merge, (-1, n_vars, cross_out_merge.shape[-2], cross_out_merge.shape[-1]))
        cross_out_merge = cross_out_merge.permute(0, 1, 3, 2)

        # Output
        # output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.flatten(cross_out_merge) # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc):
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, N]
