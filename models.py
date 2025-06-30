import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torchvision
import math
import torch.nn.functional as F
from utils.masking import TriangularCausalMask
from config import Config

class VariatesEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_time):
        # x:      [Batch Time Variate] == [128, 120, 1]
        # x_time: [Batch Time Variate] == [128, 120, 5]
        
        # B: batch size  L:seq_len  C:num of variaties
        # x: B, L, C -> B, C, L
        x = x.permute(0, 2, 1)
        if x_time is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_time.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]

        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        # print(x.shape)
        # torch.Size([128, 1, 96])
        x = self.padding_patch_layer(x)
        # print(x.shape)
        # torch.Size([128, 1, 104])
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 将向量分patch  
        # patch_len = 16, stride = 16, patch无重叠
        # torch.Size([128, 1, 7, 16])
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # torch.Size([128, 7, 16])    [batch_size, patch_num, patch_len]
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        # print(x.shape)
        # torch.Size([128, 7, 512])
        return self.dropout(x), n_vars

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TimeXerEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(TimeXerEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x1, x2, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        # torch.Size([128, 16, 512]) torch.Size([128, 11, 512])
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x1, attn = attn_layer(x1, x2, attn_mask=attn_mask, tau=tau, delta=delta)
                x1 = conv_layer(x1)
                attns.append(attn)
            x1, attn = self.attn_layers[-1](x1, x2, tau=tau, delta=None)
            attns.append(attn)
        else:  # iTransformer默认走这边
            for attn_layer in self.attn_layers:
                x1, attn = attn_layer(x1, x2, attn_mask=attn_mask, tau=tau, delta=delta)  # 这里x1会输入到下一层

        if self.norm is not None:
            x1 = self.norm(x1)

        return x1, attns

class TimeXerEncoderLayer(nn.Module):
    def __init__(self, attention, cross_attention, configs, d_model, d_ff=None, dropout=0.1, activation="relu", cross2self_attention=False):
        super(TimeXerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.cross_attention = cross_attention
        self.cross2self_attention = cross2self_attention
        self.configs = configs
        # print(type(attention))
        # print(type(cross_attention))
        # <class 'layers.SelfAttention_Family.TimeXerAttentionLayer'>
        # <class 'int'>
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x1, x2,  attn_mask=None, tau=None, delta=None):
        if not self.configs.no_attention:
            if not self.configs.no_self_attention:
                new_x1, attn = self.attention(
                    x1, x1, x1,
                    attn_mask=attn_mask,
                    tau=tau, delta=delta
                )
                x1 = x1 + self.dropout(new_x1)

                x1 = self.norm1(x1)
            
            if not self.configs.no_cross_attention:
                if not self.cross2self_attention:
                    x1_cross_attention_input = x1[:, -6:, :]
                    new_x2, attn = self.cross_attention(
                        x1_cross_attention_input, x2, x2,
                        attn_mask=attn_mask,
                        tau=tau, delta=delta
                    )
                    new_x2 = torch.concat([x1[:, :-6, :], new_x2], dim=1)
                else:
                    new_x2, attn = self.cross_attention(
                        x1, x1, x1,
                        attn_mask=attn_mask,
                        tau=tau, delta=delta
                    )

                x = x1 + self.dropout(new_x2)


                y = x = self.norm2(x)
            else:
                y = x = x1
                attn = None
        else:
            y = x = x1
            attn = None

        if not self.configs.no_FF:
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 这里就是FFN，就是两个卷积层变换一下维度合同到，不用管
            y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), attn

class TimeXerAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(TimeXerAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape # batch_size, query_length, num_heads, embed_dim
        _, S, _, D = values.shape
        # print(queries.shape)
        # print(keys.shape)
        # print(values.shape)
        # torch.Size([128, 16, 8, 64])
        # torch.Size([128, 16, 8, 64])
        # torch.Size([128, 16, 8, 64])
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # print(scores.shape)
        # torch.Size([128, 8, 16, 16])

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # print(V.shape)
        # torch.Size([128, 16, 8, 64])

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class TimeXer(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # parameters
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        if not configs.no_variate_embedding:
            self.exo_i_embedding = VariatesEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                        configs.dropout)
            self.end_i_embedding = VariatesEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                        configs.dropout)
        padding = configs.stride
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        if configs.no_embedding:
            configs.d_model = configs.seq_len
        self.net = TimeXerEncoder(
            [
                TimeXerEncoderLayer(  # 计算embedding
                    TimeXerAttentionLayer(  # 计算self attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    TimeXerAttentionLayer(  # 计算cross attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    cross2self_attention=configs.cross2self_attention
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # configs.factor = 1
        # configs.dropout = 0.1
        # configs.d_model = 512
        # configs.n_heads = 8
        # configs.d_ff = 512
        # configs.activation = gelu

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
    def forecast(self, seq_exogenous_x, seq_endogenous_x, seq_x_mark):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            # exogenous_means = seq_exogenous_x.mean(1, keepdim=True).detach()
            # seq_exogenous_x = seq_exogenous_x - exogenous_means
            # exogenous_st = torch.sqrt(torch.var(seq_exogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # seq_exogenous_x /= exogenous_st

            endogenous_means = seq_endogenous_x.mean(1, keepdim=True).detach()
            seq_endogenous_x = seq_endogenous_x - endogenous_means
            endogenous_st = torch.sqrt(torch.var(seq_endogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seq_endogenous_x /= endogenous_st
        # print(seq_exogenous_x.shape)
        # print(seq_endogenous_x.shape)
        # print(seq_x_mark.shape)
        # torch.Size([128, 96, 8])
        # torch.Size([128, 96, 1])
        # torch.Size([128, 96, 3])


        _, _, N = seq_endogenous_x.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        if not self.configs.cross2self_attention and not self.configs.no_variate_embedding and not self.configs.no_attention:
            exogenous_i_embedding = self.exo_i_embedding(seq_exogenous_x, seq_x_mark)  # covariates (e.g timestamp) can be also embedded as tokens
        else:
            exogenous_i_embedding = torch.tensor([])
        if not self.configs.no_variate_embedding:
            endogenous_i_embedding = self.end_i_embedding(seq_endogenous_x, seq_x_mark)
        seq_endogenous_x_for_patch = seq_endogenous_x.permute(0, 2, 1)
    
        if not self.configs.no_embedding:
            endogenous_patch_embedding, n_vars = self.patch_embedding(seq_endogenous_x_for_patch)
        else:
            endogenous_patch_embedding = seq_endogenous_x_for_patch
            self.d_model = seq_endogenous_x_for_patch.shape[2]
        # torch.Size([128, 1, 96])
        # torch.Size([128, 12, 512])

        # print(endogenous_i_embedding.shape)
        # torch.Size([128, 4, 512])
        if not self.configs.no_variate_embedding:
            self_attention_input = torch.concat([endogenous_patch_embedding, endogenous_i_embedding], dim=1)
        else:
            self_attention_input = endogenous_patch_embedding
        # print(self_attention_input.shape)
        # torch.Size([128, 16, 512])

        # print(exogenous_i_embedding.shape)
        # torch.Size([128, 11, 512])

        # B N E -> B N E
        encode_output, attns = self.net(self_attention_input, exogenous_i_embedding)
        # print(encode_output.shape)
        # torch.Size([128, 16, 512])

        # B N E -> B N S -> B S N
        # torch.Size([128, 16, 512]) -> torch.Size([128, 16, 96]) -> torch.Size([128, 96, 1])
        dec_out = self.projector(encode_output).permute(0, 2, 1)[:, :, :N]  # filter the covariates
        # print(dec_out.shape)
        # torch.Size([128, 96, 1])

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (endogenous_st[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (endogenous_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # print(dec_out.shape)
        # torch.Size([128, 96, 1])
        return dec_out

    def forward(self, seq_exogenous_x, seq_endogenous_x, seq_x_mark):
        dec_out = self.forecast(seq_exogenous_x, seq_endogenous_x, seq_x_mark)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
class MyNet(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # parameters
        self.use_glaff = configs.use_glaff
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        if self.configs.train_alpha:
            # 将 alpha 改为可训练参数
            self.alpha = nn.Parameter(torch.tensor(configs.alpha, dtype=torch.float))
        else:
            self.alpha = configs.alpha

        self.exo_per_i_embedding = VariatesEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.exo_non_per_i_embedding = VariatesEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.end_periodic_i_embedding = VariatesEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.end_non_periodic_i_embedding = VariatesEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.end_i_embedding = VariatesEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        padding = configs.stride
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)
        self.periodic_patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)
        self.non_periodic_patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        if configs.no_embedding:
            configs.d_model = configs.seq_len
        self.net1 = TimeXerEncoder(
            [
                TimeXerEncoderLayer(  # 计算embedding
                    TimeXerAttentionLayer(  # 计算self attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    TimeXerAttentionLayer(  # 计算cross attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    cross2self_attention=configs.cross2self_attention
                ) for l in range(configs.L1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.net2 = TimeXerEncoder(
            [
                TimeXerEncoderLayer(  # 计算embedding
                    TimeXerAttentionLayer(  # 计算self attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    TimeXerAttentionLayer(  # 计算cross attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    cross2self_attention=configs.cross2self_attention
                ) for l in range(configs.L2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # configs.factor = 1
        # configs.dropout = 0.1
        # configs.d_model = 512
        # configs.n_heads = 8
        # configs.d_ff = 512
        # configs.activation = gelu

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        if self.use_glaff:
            self.plugin = Plugin(configs)

        
    def forecast(self, seq_exogenous_periodic_x, seq_exogenous_non_periodic_x, seq_endogenous_periodic_x, 
                 seq_endogenous_non_periodic_x, seq_x_mark, x_time, y_time):
        seq_endogenous_x = seq_endogenous_periodic_x + seq_endogenous_non_periodic_x

        x_enc_copy = seq_endogenous_x.clone()
        x_mark_enc_copy = x_time.clone()
        x_mark_dec_copy = y_time.clone()

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            # exogenous_means = seq_exogenous_x.mean(1, keepdim=True).detach()
            # seq_exogenous_x = seq_exogenous_x - exogenous_means
            # exogenous_st = torch.sqrt(torch.var(seq_exogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # seq_exogenous_x /= exogenous_st
            endogenous_means = seq_endogenous_x.mean(1, keepdim=True).detach()
            seq_endogenous_x = seq_endogenous_x - endogenous_means
            endogenous_st = torch.sqrt(torch.var(seq_endogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seq_endogenous_x /= endogenous_st
        # print(seq_exogenous_x.shape)
        # print(seq_endogenous_x.shape)
        # print(seq_x_mark.shape)
        # torch.Size([128, 96, 8])
        # torch.Size([128, 96, 1])
        # torch.Size([128, 96, 3])


        _, _, N = seq_endogenous_x.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        if not self.configs.cross2self_attention and not self.configs.no_variate_embedding and not self.configs.no_attention:
            exogenous_periodic_i_embedding = self.exo_per_i_embedding(seq_exogenous_periodic_x, seq_x_mark)  # covariates (e.g timestamp) can be also embedded as tokens
            exogenous_non_periodic_i_embedding = self.exo_non_per_i_embedding(seq_exogenous_non_periodic_x, seq_x_mark)
        else:
            exogenous_periodic_i_embedding = torch.tensor([])
            exogenous_non_periodic_i_embedding = torch.tensor([])
        if not self.configs.no_variate_embedding:
            endogenous_periodic_i_embedding = self.end_periodic_i_embedding(seq_endogenous_periodic_x, seq_x_mark)
            endogenous_non_periodic_i_embedding = self.end_non_periodic_i_embedding(seq_endogenous_non_periodic_x, seq_x_mark)
        seq_endogenous_periodic_x_for_patch = seq_endogenous_periodic_x.permute(0, 2, 1)
        seq_endogenous_non_periodic_x_for_patch = seq_endogenous_non_periodic_x.permute(0, 2, 1)
    
        if not self.configs.no_embedding:
            endogenous_periodic_patch_embedding, n_vars = self.periodic_patch_embedding(seq_endogenous_periodic_x_for_patch)
            endogenous_non_periodic_patch_embedding, n_vars = self.non_periodic_patch_embedding(seq_endogenous_non_periodic_x_for_patch)
        else:
            endogenous_periodic_patch_embedding = seq_endogenous_periodic_x_for_patch
            endogenous_non_periodic_patch_embedding = seq_endogenous_non_periodic_x_for_patch
            self.d_model = seq_endogenous_periodic_x_for_patch.shape[2]
        # torch.Size([128, 1, 96])
        # torch.Size([128, 12, 512])

        # print(endogenous_i_embedding.shape)
        # torch.Size([128, 4, 512])
        endogenous_patch_embedding = endogenous_periodic_patch_embedding + endogenous_non_periodic_patch_embedding
        endogenous_i_embedding = endogenous_periodic_i_embedding + endogenous_non_periodic_i_embedding

        # seq_endogenous_x_for_patch = seq_endogenous_periodic_x_for_patch + seq_endogenous_non_periodic_x_for_patch
        # endogenous_patch_embedding, n_vars = self.patch_embedding(seq_endogenous_x_for_patch)
        # endogenous_i_embedding = self.end_i_embedding(seq_endogenous_x, seq_x_mark)


        if self.configs.endogenous_separate:
            if not self.configs.no_variate_embedding:
                endogenous_periodic_input = torch.concat([endogenous_periodic_patch_embedding, endogenous_periodic_i_embedding], dim=1)
                endogenous_non_periodic_input = torch.concat([endogenous_non_periodic_patch_embedding, endogenous_non_periodic_i_embedding], dim=1)
            else:
                endogenous_periodic_input = endogenous_periodic_patch_embedding
                endogenous_non_periodic_input = endogenous_non_periodic_patch_embedding
        else:
            if not self.configs.no_variate_embedding:
                endogenous_input = torch.concat([endogenous_patch_embedding, endogenous_i_embedding], dim=1)
            else:
                endogenous_input = endogenous_patch_embedding

        # print(endogenous_input.shape)
        # print(endogenous_input[:5])
        # sys.exit(0)
        # print(endogenous_input.shape)
        # torch.Size([128, 16, 512])

        # print(exogenous_i_embedding.shape)
        # torch.Size([128, 11, 512])

        # B N E -> B N E
        if self.configs.endogenous_separate:
            periodic_output, attns = self.net1(endogenous_periodic_input, exogenous_periodic_i_embedding)
            non_periodic_output, attns = self.net2(endogenous_non_periodic_input, exogenous_non_periodic_i_embedding)
        else:
            periodic_output, attns = self.net1(endogenous_input, exogenous_periodic_i_embedding)
            non_periodic_output, attns = self.net2(endogenous_input, exogenous_non_periodic_i_embedding)
        
        encode_output = self.alpha * periodic_output + (1 - self.alpha) * non_periodic_output

        # encode_output, attns = self.net(endogenous_input, exogenous_i_embedding)
        # print(encode_output.shape)
        # torch.Size([128, 16, 512])

        # B N E -> B N S -> B S N
        # torch.Size([128, 16, 512]) -> torch.Size([128, 16, 96]) -> torch.Size([128, 96, 1])
        dec_out = self.projector(encode_output).permute(0, 2, 1)[:, :, :N]  # filter the covariates
        # print(dec_out.shape)
        # torch.Size([128, 96, 1])

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (endogenous_st[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (endogenous_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # print(dec_out.shape)
        # torch.Size([128, 96, 1])

        if self.use_glaff:
            dec_out = self.plugin(x_enc_copy, x_mark_enc_copy, dec_out, x_mark_dec_copy[:, -self.pred_len:, :])

        return dec_out

    def forward(self, seq_exogenous_periodic_x, seq_exogenous_non_periodic_x, seq_endogenous_periodic_x, 
                seq_endogenous_non_periodic_x, seq_x_mark, x_time, y_time):
        
        dec_out = self.forecast(seq_exogenous_periodic_x, seq_exogenous_non_periodic_x, seq_endogenous_periodic_x, 
                                seq_endogenous_non_periodic_x, seq_x_mark, x_time, y_time)
        
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class MyNet_SD(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # parameters
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        if self.configs.train_alpha:
            # 将 alpha 改为可训练参数
            self.alpha = nn.Parameter(torch.tensor(configs.alpha, dtype=torch.float))
        else:
            self.alpha = configs.alpha

        # self.exo_per_i_embedding = VariatesEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)
        # self.exo_non_per_i_embedding = VariatesEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)
        self.end_periodic_i_embedding = VariatesEmbedding(configs.seq_len * configs.num_of_similar_days, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.end_non_periodic_i_embedding = VariatesEmbedding(configs.seq_len * configs.num_of_similar_days, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.end_i_embedding = VariatesEmbedding(configs.seq_len * configs.num_of_similar_days, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        padding = configs.stride
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)
        self.periodic_patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)
        self.non_periodic_patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        if configs.no_embedding:
            configs.d_model = configs.seq_len

        self.net1 = TimeXerEncoder(
            [
                TimeXerEncoderLayer(  # 计算embedding
                    TimeXerAttentionLayer(  # 计算self attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    TimeXerAttentionLayer(  # 计算cross attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    cross2self_attention=configs.cross2self_attention
                ) for l in range(configs.L1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.net2 = TimeXerEncoder(
            [
                TimeXerEncoderLayer(  # 计算embedding
                    TimeXerAttentionLayer(  # 计算self attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    TimeXerAttentionLayer(  # 计算cross attention
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    cross2self_attention=configs.cross2self_attention
                ) for l in range(configs.L2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # configs.factor = 1
        # configs.dropout = 0.1
        # configs.d_model = 512
        # configs.n_heads = 8
        # configs.d_ff = 512
        # configs.activation = gelu

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        
    def forecast(self, seq_endogenous_periodic_x, seq_endogenous_non_periodic_x, seq_x_mark):
        # print(seq_endogenous_periodic_x.shape)
        # print(seq_endogenous_non_periodic_x.shape)
        # print(seq_x_mark.shape)
        # sys.exit(0)
        # torch.Size([128, 5, 24, 1])
        # torch.Size([128, 5, 24, 1])
        # torch.Size([128, 5, 24, 5])
        shape = seq_endogenous_periodic_x.shape
        seq_endogenous_periodic_x = seq_endogenous_periodic_x.reshape(shape[0] , shape[1] * shape[2], shape[3])
        seq_endogenous_non_periodic_x = seq_endogenous_non_periodic_x.reshape(shape[0], shape[1] * shape[2], shape[3])
        shape = seq_x_mark.shape
        seq_x_mark = seq_x_mark.reshape(shape[0], shape[1] * shape[2], shape[3])

        seq_endogenous_x = seq_endogenous_periodic_x + seq_endogenous_non_periodic_x

        # print(seq_endogenous_x.shape)
        # print(seq_endogenous_periodic_x.shape)
        # print(seq_endogenous_non_periodic_x.shape)
        # print(seq_x_mark.shape)
        # sys.exit(0)
        # torch.Size([128, 120, 1])
        # torch.Size([128, 120, 1])
        # torch.Size([128, 120, 1])
        # torch.Size([128, 120, 5])


        if self.use_norm:
            # Normalization from Non-stationary Transformer
            # exogenous_means = seq_exogenous_x.mean(1, keepdim=True).detach()
            # seq_exogenous_x = seq_exogenous_x - exogenous_means
            # exogenous_st = torch.sqrt(torch.var(seq_exogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # seq_exogenous_x /= exogenous_st
            endogenous_means = seq_endogenous_x.mean(1, keepdim=True).detach()
            seq_endogenous_x = seq_endogenous_x - endogenous_means
            endogenous_st = torch.sqrt(torch.var(seq_endogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seq_endogenous_x /= endogenous_st

        _, _, N = seq_endogenous_x.shape  # B L N

        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        if not self.configs.no_variate_embedding:
            endogenous_periodic_i_embedding = self.end_periodic_i_embedding(seq_endogenous_periodic_x, seq_x_mark)
            endogenous_non_periodic_i_embedding = self.end_non_periodic_i_embedding(seq_endogenous_non_periodic_x, seq_x_mark)
        seq_endogenous_periodic_x_for_patch = seq_endogenous_periodic_x.permute(0, 2, 1)
        seq_endogenous_non_periodic_x_for_patch = seq_endogenous_non_periodic_x.permute(0, 2, 1)
    
        if not self.configs.no_embedding:
            endogenous_periodic_patch_embedding, n_vars = self.periodic_patch_embedding(seq_endogenous_periodic_x_for_patch)
            endogenous_non_periodic_patch_embedding, n_vars = self.non_periodic_patch_embedding(seq_endogenous_non_periodic_x_for_patch)
        else:
            endogenous_periodic_patch_embedding = seq_endogenous_periodic_x_for_patch
            endogenous_non_periodic_patch_embedding = seq_endogenous_non_periodic_x_for_patch
            self.d_model = seq_endogenous_periodic_x_for_patch.shape[2]
        # torch.Size([128, 1, 96])
        # torch.Size([128, 12, 512])

        # print(endogenous_i_embedding.shape)
        # torch.Size([128, 4, 512])
        endogenous_patch_embedding = endogenous_periodic_patch_embedding + endogenous_non_periodic_patch_embedding
        endogenous_i_embedding = endogenous_periodic_i_embedding + endogenous_non_periodic_i_embedding

        # seq_endogenous_x_for_patch = seq_endogenous_periodic_x_for_patch + seq_endogenous_non_periodic_x_for_patch
        # endogenous_patch_embedding, n_vars = self.patch_embedding(seq_endogenous_x_for_patch)
        # endogenous_i_embedding = self.end_i_embedding(seq_endogenous_x, seq_x_mark)


        if self.configs.endogenous_separate:
            if not self.configs.no_variate_embedding:
                endogenous_periodic_input = torch.concat([endogenous_periodic_patch_embedding, endogenous_periodic_i_embedding], dim=1)
                endogenous_non_periodic_input = torch.concat([endogenous_non_periodic_patch_embedding, endogenous_non_periodic_i_embedding], dim=1)
            else:
                endogenous_periodic_input = endogenous_periodic_patch_embedding
                endogenous_non_periodic_input = endogenous_non_periodic_patch_embedding
        else:
            if not self.configs.no_variate_embedding:
                endogenous_input = torch.concat([endogenous_patch_embedding, endogenous_i_embedding], dim=1)
            else:
                endogenous_input = endogenous_patch_embedding

        # print(endogenous_input.shape)
        # print(endogenous_input[:5])
        # sys.exit(0)
        # print(endogenous_input.shape)
        # torch.Size([128, 16, 512])

        # print(exogenous_i_embedding.shape)
        # torch.Size([128, 11, 512])

        # B N E -> B N E
        if self.configs.endogenous_separate:
            periodic_output, attns = self.net1(endogenous_periodic_input, None)
            non_periodic_output, attns = self.net2(endogenous_non_periodic_input, None)
        else:
            periodic_output, attns = self.net1(endogenous_input, None)
            non_periodic_output, attns = self.net2(endogenous_input, None)
        
        encode_output = self.alpha * periodic_output + (1 - self.alpha) * non_periodic_output

        # encode_output, attns = self.net(endogenous_input, exogenous_i_embedding)
        # print(encode_output.shape)
        # torch.Size([128, 16, 512])

        # B N E -> B N S -> B S N
        # torch.Size([128, 16, 512]) -> torch.Size([128, 16, 96]) -> torch.Size([128, 96, 1])
        dec_out = self.projector(encode_output).permute(0, 2, 1)[:, :, :N]  # filter the covariates
        # print(dec_out.shape)
        # torch.Size([128, 96, 1])

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (endogenous_st[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (endogenous_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # print(dec_out.shape)
        # torch.Size([128, 96, 1])
        return dec_out

    def forward(self, seq_endogenous_periodic_x, seq_endogenous_non_periodic_x, seq_x_mark):
        
        dec_out = self.forecast(seq_endogenous_periodic_x, seq_endogenous_non_periodic_x, seq_x_mark)
        
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

# class iTransformer(nn.Module):

#     def __init__(self, configs):
#         super().__init__()

#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         # Embedding
#         self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=False), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#         # Decoder
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
#         if self.task_name == 'imputation':
#             self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
#         if self.task_name == 'anomaly_detection':
#             self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
#         if self.task_name == 'classification':
#             self.act = F.gelu
#             self.dropout = nn.Dropout(configs.dropout)
#             self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         # Normalization from Non-stationary Transformer
#         means = x_enc.mean(1, keepdim=True).detach()
#         x_enc = x_enc - means
#         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_enc /= stdev

#         _, _, N = x_enc.shape

#         # Embedding
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
#         # De-Normalization from Non-stationary Transformer
#         dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#         dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#         return dec_out

#     def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
#         # Normalization from Non-stationary Transformer
#         means = x_enc.mean(1, keepdim=True).detach()
#         x_enc = x_enc - means
#         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_enc /= stdev

#         _, L, N = x_enc.shape

#         # Embedding
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
#         # De-Normalization from Non-stationary Transformer
#         dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#         dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#         return dec_out

#     def anomaly_detection(self, x_enc):
#         # Normalization from Non-stationary Transformer
#         means = x_enc.mean(1, keepdim=True).detach()
#         x_enc = x_enc - means
#         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_enc /= stdev

#         _, L, N = x_enc.shape

#         # Embedding
#         enc_out = self.enc_embedding(x_enc, None)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
#         # De-Normalization from Non-stationary Transformer
#         dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#         dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#         return dec_out

#     def classification(self, x_enc, x_mark_enc):
#         # Embedding
#         enc_out = self.enc_embedding(x_enc, None)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         # Output
#         output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
#         output = self.dropout(output)
#         output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
#         output = self.projection(output)  # (batch_size, num_classes)
#         return output

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
#         if self.task_name == 'imputation':
#             dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
#             return dec_out  # [B, L, D]
#         if self.task_name == 'anomaly_detection':
#             dec_out = self.anomaly_detection(x_enc)
#             return dec_out  # [B, L, D]
#         if self.task_name == 'classification':
#             dec_out = self.classification(x_enc, x_mark_enc)
#             return dec_out  # [B, N]
#         return None


import torch.nn.functional as F
from Plugin.model import Plugin


class MLP(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # parameters
        self.use_glaff = configs.use_glaff
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        padding = configs.stride
        # self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        self.net = nn.Sequential(
            # nn.Linear(configs.seq_len, configs.pred_len),
            nn.Linear(45, 32),
            nn.GELU(),
            nn.Linear(32, 45),
            nn.GELU()
            # nn.Linear(2048, configs.d_model),
        )

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        if self.use_glaff:
            self.plugin = Plugin(configs)

        
    def forecast(self, seq_endogenous_x, x_time, y_time):
        x_enc_copy = seq_endogenous_x.clone()
        x_mark_enc_copy = x_time.clone()
        x_mark_dec_copy = y_time.clone()

        if self.use_norm:
            endogenous_means = seq_endogenous_x.mean(1, keepdim=True).detach()
            seq_endogenous_x = seq_endogenous_x - endogenous_means
            endogenous_st = torch.sqrt(torch.var(seq_endogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seq_endogenous_x /= endogenous_st

        seq_endogenous_x = seq_endogenous_x.permute(0, 2, 1)
        # # 将数值大于 600 的元素设置为 600
        # seq_endogenous_x = torch.where(seq_endogenous_x > 500, torch.tensor(500.0), seq_endogenous_x)

        shape = seq_endogenous_x.shape
        seq_endogenous_x = seq_endogenous_x.reshape(shape[0], 45, 24)
        seq_endogenous_x = seq_endogenous_x.transpose(1, 2)
        seq_endogenous_x = seq_endogenous_x.reshape(shape[0] * 24, 45)


        dec_out = self.net(seq_endogenous_x)

        dec_out = dec_out.reshape(shape[0], 24, 45)
        dec_out = dec_out.transpose(1, 2)
        dec_out = dec_out.reshape(shape[0], 1, 24 * 45)

        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * (endogenous_st[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (endogenous_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = torch.where(dec_out > 500, torch.tensor(500.0), dec_out)

        if self.use_glaff:
            dec_out = self.plugin(x_enc_copy, x_mark_enc_copy, dec_out, x_mark_dec_copy[:, -self.pred_len:, :])

        return dec_out

    def forward(self, seq_endogenous_x, x_time, y_time):
        dec_out = self.forecast(seq_endogenous_x, x_time, y_time)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class CNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.hw_rate = configs.hw_rate
        # seq_len=96:channels=7     seq_len=96*7=672:channels=43
        self.channels = 43

        padding = configs.stride
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # net
        self.width = int(math.sqrt(self.seq_len/self.hw_rate)) + 1
        self.height = int(self.width * self.hw_rate) + 1
        self.pool = nn.AdaptiveAvgPool2d((self.channels, self.height * self.width))
        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = nn.Conv2d(self.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # remove layer3 & layer4
        self.resnet.layer3 = nn.Sequential()
        self.resnet.layer4 = nn.Sequential()
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 96),
            nn.Sigmoid())

    def forward(self, seq_exogenous_x, seq_endogenous_x, seq_x_mark) -> torch.Tensor:
        if self.use_norm:
            endogenous_means = seq_endogenous_x.mean(1, keepdim=True).detach()
            seq_endogenous_x = seq_endogenous_x - endogenous_means
            endogenous_st = torch.sqrt(torch.var(seq_endogenous_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seq_endogenous_x /= endogenous_st

        _, _, N = seq_endogenous_x.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        seq_endogenous_x_for_patch = seq_endogenous_x.permute(0, 2, 1)
        endogenous_patch_embedding, n_vars = self.patch_embedding(seq_endogenous_x_for_patch)

        endogenous_patch_embedding = self.pool(endogenous_patch_embedding)
        endogenous_patch_embedding = endogenous_patch_embedding.reshape(-1, self.channels, self.height, self.width)
        return self.resnet(endogenous_patch_embedding).reshape((-1, 96, 1))


# class CNNForecastModel(nn.Module):
#     def __init__(self, dim=1, hidden_channels=32, e_layer=5):
#         super(CNNForecastModel, self).__init__()

#         self.e_layer = e_layer

#         # 输入 shape: (B, 5, 96, dim)
#         # reshape 成 (B, dim, 96, 5)
        
#         # 构造多个卷积块
#         layers = []
#         for i in range(e_layer):
#             in_ch = dim if i == 0 else hidden_channels
#             layers.append(nn.Conv2d(in_channels=in_ch, out_channels=hidden_channels, kernel_size=(3, 3), padding=(1, 1)))
#             layers.append(nn.BatchNorm2d(hidden_channels))
#             layers.append(nn.ReLU())

#         self.conv_blocks = nn.Sequential(*layers)
        
#         # 输出层
#         self.conv_out = nn.Conv2d(hidden_channels, out_channels=dim, kernel_size=1)

#     def forward(self, x):
#         """
#         输入: x.shape == (batch_size, 5, 96, dim)
#         输出: (batch_size, 96, dim)
#         """
#         # 调整形状为 (B, dim, 96, 5)
#         x = x.permute(0, 3, 2, 1)

#         # 多层卷积块
#         x = self.conv_blocks(x)  # (B, hidden_channels, 96, 5)

#         # 输出卷积
#         x = self.conv_out(x)  # (B, dim, 96, 5)

#         # 取最后一天（第5天）
#         x = x[:, :, :, -1]  # (B, dim, 96)

#         # 转为 (B, 96, dim)
#         x = x.permute(0, 2, 1)
#         return x

class CNNForecastModel(nn.Module):
    def __init__(self, configs, dim=1, seq_len=96, emb_dim=512, hidden_channels=32, e_layer=1, dropout=0.2):
        super(CNNForecastModel, self).__init__()

        self.configs = configs
        self.e_layer = e_layer
        self.emb_dim = emb_dim
        self.seq_len = seq_len

        # 位置嵌入：每个时间步一个512维向量
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, seq_len, emb_dim))

        # 特征线性映射：dim -> emb_dim
        self.input_proj = nn.Linear(dim, emb_dim)

        # 卷积块 + Dropout
        layers = []
        for i in range(e_layer):
            in_ch = emb_dim if i == 0 else hidden_channels
            layers.append(nn.Conv2d(in_channels=in_ch, out_channels=hidden_channels, kernel_size=(3, 3), padding=(1, 1)))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # <-- 加上 Dropout

        self.conv_blocks = nn.Sequential(*layers)

        # 输出通道：hidden_channels -> dim
        self.conv_out = nn.Conv2d(hidden_channels, out_channels=dim, kernel_size=1)

    def forward(self, x, seq_x_mark):
        """
        输入: x.shape == (B, 5, 96, dim)
        输出: (B, 96, dim)
        """
        B, D, T, F = x.shape  # (B, 5, 96, dim)

        # 特征嵌入
        x = self.input_proj(x)  # (B, 5, 96, emb_dim)

        # 加上位置编码
        x = x + self.pos_embedding  # (B, 5, 96, emb_dim)

        # 调整形状为卷积格式
        x = x.permute(0, 3, 2, 1)  # (B, emb_dim, 96, 5)

        # 卷积 + Dropout
        x = self.conv_blocks(x)  # (B, hidden_channels, 96, 5)

        # 输出卷积
        x = self.conv_out(x)  # (B, dim, 96, 5)

        # 取最后一天
        x = x[:, :, :, -1]  # (B, dim, 96)

        # 转换为 (B, 96, dim)
        x = x.permute(0, 2, 1)
        return x


    

if __name__ == '__main__':
    config_file = 'training.ini'
    configs = Config(config_file)
    x = torch.rand(128, 7, 512)
    net = CNN(configs)
    # net = MLP(configs)
    # net = TimeXer(configs)
    # print(net)
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Model Param Num: {param_num:,}')
    # print(net(x).shape)