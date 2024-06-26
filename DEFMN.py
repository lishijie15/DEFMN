import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.fft
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed_test import DataEmbedding_SE
from layers.expert_gate import Expert_seg, Expert_dynamic



def FFT_for_Period(x, k):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    z, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def spilt(x, f):
    if f == 0:
        cs = torch.split(x, 1, dim=1)
        x = torch.concat(cs, dim=0).squeeze(1)
    else:
        cs = torch.split(x, 1, dim=2)
        x = torch.concat(cs, dim=0).squeeze(2)
    return x


def concat(b, x, f):
    if f == 0:
        x = torch.split(x, b, dim=0)
        x = torch.stack(x, dim=1)
    else:
        x = torch.split(x, b, dim=0)
        x = torch.stack(x, dim=2)
    return x


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv


class Times_se(nn.Module):
    def __init__(self, seq_len, horizon, d_model, d_ff, n_heads=8):
        super(Times_se, self).__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.output_attention = 'store_true'
        self.dropout = 0.1
        self.b_layers = 1
        self.activation = 'gelu'
        self.factor = 7

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.b_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x):
        enc_out, attns = self.encoder(x, attn_mask=None)
        return enc_out


class GCTblock_enc(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_out, cheb_k, d_model, seq_len, out_len, b_layers, top_k, d_ff,
                 num_kernels):
        super(GCTblock_enc, self).__init__()
        self.seq_len = seq_len
        self.out_len = out_len
        self.k = top_k
        self.num_nodes = num_nodes
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.cheb_k = cheb_k
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.b_layers = b_layers
        self.d_model = d_model
        self.AGCNs = nn.ModuleList([AGCN(self.d_model, self.d_model, self.cheb_k)
                                    for _ in range(self.b_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.expert_segment = Expert_seg(self.d_model, self.input_dim, self.output_dim, self.cheb_k, self.num_nodes, self.b_layers)
        self.Times_se = nn.ModuleList([Times_se(self.seq_len, self.out_len, self.d_model, self.d_ff)
                                       for _ in range(self.b_layers)])
        self.data_embedding = DataEmbedding_SE(self.input_dim, self.d_model, self.seq_len)

    def forward(self, x, y_cov, supports):
        # shape of x: (B, T, N, D)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.input_dim
        # embedding
        current_inputs = self.data_embedding(x)  # [B,C,N,T]
        o_expert, h_expert = self.expert_segment(current_inputs, supports)

        return o_expert, h_expert, current_inputs


class GCTblock_dec(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_out, cheb_k, seq_len, out_len, d_model, b_layers, top_k, d_ff,
                 num_kernels):
        super(GCTblock_dec, self).__init__()
        self.seq_len = seq_len
        self.out_len = out_len
        self.k = top_k
        self.num_nodes = num_nodes
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.d_ff = d_ff
        self.cheb_k = cheb_k
        self.num_kernels = num_kernels
        self.b_layers = b_layers
        self.d_model = 2 * d_model
        self.AGCNs = nn.ModuleList([AGCN(self.input_dim, self.output_dim, self.cheb_k)
                                    for _ in range(self.b_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.Times_de = nn.ModuleList([Times_se(self.seq_len, self.out_len, self.d_model, self.d_ff)
                                       for _ in range(self.b_layers)])

    def forward(self, current_inputs, supports):
        bat = current_inputs.shape[0]
        for i in range(self.b_layers):
            current_inputs = spilt(current_inputs, 0)
            state = self.AGCNs[i](current_inputs, supports)
            state = concat(bat, state, 0)
            t_dec_out = spilt(state, 1)
            # TimesNet
            t_dec_out = self.layer_norm(self.Times_de[i](t_dec_out))
            current_inputs = concat(bat, t_dec_out, 1)
        return current_inputs

class DEFMN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, seq_len, out_len, b_layers, d_model, top_k, d_ff, num_kernels,
                 cheb_k=3, ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=2000):
        super(DEFMN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.b_layers = b_layers
        self.d_model = d_model
        self.top_k = top_k
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.output_dim = output_dim
        self.out_len = out_len
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps

        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # encoder
        self.GCT_enc = GCTblock_enc(self.num_nodes, self.input_dim, self.output_dim, self.cheb_k,
                                    self.d_model, self.seq_len, self.out_len, self.b_layers, self.top_k, self.d_ff,
                                    self.num_kernels)

        self.dyna = Expert_dynamic(self.d_model, self.input_dim, self.b_layers)
        # decoder
        self.decoder_dim = self.d_model + self.mem_dim
        self.GCT_dec = GCTblock_dec(self.num_nodes, self.d_model + self.mem_dim, self.decoder_dim, self.cheb_k,
                                    self.seq_len, self.out_len, self.d_model, self.b_layers, self.top_k, self.d_ff,
                                    self.num_kernels)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.out_len, bias=True))

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.d_model, self.mem_dim),
                                         requires_grad=True)  # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]]  # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]]  # B, N, d
        return value, query, pos, neg

    def forward(self, x, y_cov, labels=None, batches_seen=None):
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)  # E,ET
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        bat = x.shape[0]

        out_enc, hid_enc, x_emb = self.GCT_enc(x, y_cov, supports)

        o_attention, h_attention = self.dyna(x_emb, hid_enc)
        query_in = out_enc + o_attention.permute(0, 3, 1, 2)

        # MSM
        m_t = spilt(query_in, 0)
        h_att, query, pos, neg = self.query_memory(m_t)
        h_t = torch.cat([m_t, h_att], dim=-1)
        h_b = concat(bat, h_t, 0)

        # decoder only
        h_de = self.GCT_dec(h_b, supports)
        h_de = h_de + h_b
        output = self.proj(h_de).permute(0, 3, 2, 1)

        return output, h_att, query, pos, neg
