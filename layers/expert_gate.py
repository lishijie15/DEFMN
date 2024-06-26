import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as cp

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, dropout=0.1):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.dropout = dropout
        self.MLP = nn.Linear((2 * cheb_k + 1) * dim_in, dim_out)

    def forward(self, x, supports): #B, N, T, C
        x_g = [x]
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("bntc,nm->bmtc", x, support))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = self.MLP(x_g)  # b, N, dim_out
        return x_gconv


class self_Attention(nn.Module):
    """
    Assume input has shape B, N, T, C or B, T, N, C
    Note: Attention map will be B, N, T, T or B, T, N, N
        - Could be utilized for both spatial and temporal modeling
        - Able to get additional kv-input (for Time-Enhanced Attention)
    """
    def __init__(self, in_dim, hidden_size, dropout, num_heads=4):
        super(self_Attention, self).__init__()
        self.query = nn.Linear(in_dim, hidden_size, bias=False)
        self.key = nn.Linear(in_dim, hidden_size, bias=False)
        self.value = nn.Linear(in_dim, hidden_size, bias=False)
        self.num_heads = num_heads
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        assert hidden_size % num_heads == 0

    def forward(self, x, kv=None):
        if kv is None:
            kv = x
        query = self.query(x)
        key = self.key(kv)
        value = self.value(kv)
        num_heads = self.num_heads
        if num_heads > 1:
            query = torch.cat(torch.chunk(query, num_heads, dim=-1), dim=0)
            key = torch.cat(torch.chunk(key, num_heads, dim=-1), dim=0)
            value = torch.cat(torch.chunk(value, num_heads, dim=-1), dim=0)
        d = value.size(-1)
        energy = torch.matmul(query, key.transpose(-1, -2))
        energy = energy / (d ** 0.5)
        score = torch.softmax(energy, dim=-1)
        head_out = torch.matmul(score, value)
        out = torch.cat(torch.chunk(head_out, num_heads, dim=0), dim=-1)
        return self.dropout(self.proj(out))


class LayerNorm(nn.Module):
    # Assume input has shape B, N, T, C
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(*normalized_shape))
        self.beta = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, x):
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        # mean --> shape :(B, C, H, W) --> (B)
        # mean with keepdims --> shape: (B, C, H, W) --> (B, 1, 1, 1)
        mean = x.mean(dim=dims, keepdims=True)
        std = x.std(dim=dims, keepdims=True, unbiased=False)
        # x_norm = (B, C, H, W)
        x_norm = (x - mean) / (std + self.eps)
        out = x_norm * self.gamma + self.beta
        return out


class SkipConnection(nn.Module):
    def __init__(self, module, norm):
        super(SkipConnection, self).__init__()
        self.module = module
        self.norm = norm

    def forward(self, x, aux=None):
        out = self.norm(x + self.module(x, aux))
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_size, dropout, activation=nn.GELU()):
        super(PositionwiseFeedForward, self).__init__()
        self.act = activation
        self.l1 = nn.Linear(in_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, in_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, kv=None):
        return self.dropout(self.l2(self.act(self.l1(x))))



class TCN(nn.Module):
    def __init__(self, d_model, dropout=0.1, activation="relu"):
        super(TCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, res=None):

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return y

class EBFE(nn.Module):
    def __init__(self, d_model):
        super(EBFE, self).__init__()
        self.d_model = d_model
        self.EBFE_weight = nn.Parameter(torch.randn(self.d_model, self.d_model, self.d_model, dtype=torch.float32) * 0.02)
        self.EBFE_bias = nn.Parameter(torch.randn(self.d_model, self.d_model, dtype=torch.float32) * 0.02)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, aux=None):
        out = F.relu(self.multiply(x, self.EBFE_weight) + self.EBFE_bias)
        out = F.softshrink(out, lambd=0.0)
        return out

class HBFE(nn.Module):
    def __init__(self, d_model, num_nodes):
        super(HBFE, self).__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.complex_weight_ll_de = nn.Parameter(torch.randn(self.num_nodes, self.d_model, self.d_model, dtype=torch.float32) * 0.02)

    def forward(self, x, aux=None):
        out = x * self.complex_weight_ll_de
        return out

class Expert_seg(nn.Module):
    def __init__(self, hidden_size, input_dim, output_dim, cheb_k, num_nodes, layers, spatial=False, activation=nn.ReLU()):
        super(Expert_seg, self).__init__()
        self.spatial = spatial
        self.act = activation
        self.cheb_k = cheb_k
        self.out_dim = output_dim
        self.input_dim = input_dim
        dropout = 0.1
        GCN = AGCN(hidden_size, hidden_size, self.cheb_k)
        t_attn = self_Attention(in_dim=hidden_size, hidden_size=hidden_size, dropout=dropout)
        tcn = TCN(hidden_size)
        E_Blend = EBFE(hidden_size)
        H_Blend = HBFE(hidden_size, num_nodes)
        ff = PositionwiseFeedForward(in_dim=hidden_size, hidden_size=4 * hidden_size, dropout=dropout)
        norm = LayerNorm(normalized_shape=(hidden_size,))

        self.total_linear = nn.Linear(input_dim, hidden_size)
        self.load_linear = nn.Linear(1, hidden_size)
        self.pv_linear = nn.Linear(1, hidden_size)
        self.wind_linear = nn.Linear(1, hidden_size)

        if output_dim == 1:
            self.proj = nn.Linear(hidden_size, hidden_size + output_dim)
        else:
            self.proj = nn.Linear(hidden_size * 4, output_dim)

        self.temporal_layers_load = nn.ModuleList()
        self.temporal_layers_load1 = nn.ModuleList()
        self.spatial_layers_load = nn.ModuleList()
        self.ed_layers_load = nn.ModuleList()
        self.ff_load = nn.ModuleList()
        self.temporal_layers_pv = nn.ModuleList()
        self.spatial_layers_pv = nn.ModuleList()
        self.ed_layers_pv = nn.ModuleList()
        self.ff_pv = nn.ModuleList()
        self.temporal_layers_wind = nn.ModuleList()
        self.spatial_layers_wind = nn.ModuleList()
        self.ed_layers_wind = nn.ModuleList()
        self.ff_wind = nn.ModuleList()

        for _ in range(layers):
            self.temporal_layers_load.append(SkipConnection(cp(t_attn), cp(norm)))
            self.temporal_layers_load1.append(SkipConnection(cp(E_Blend), cp(norm)))
            self.temporal_layers_pv.append(SkipConnection(cp(t_attn), cp(norm)))
            self.temporal_layers_wind.append(SkipConnection(cp(t_attn), cp(norm)))
            self.spatial_layers_load.append(SkipConnection(cp(GCN), cp(norm)))
            self.spatial_layers_pv.append(SkipConnection(cp(GCN), cp(norm)))
            self.spatial_layers_wind.append(SkipConnection(cp(GCN), cp(norm)))
            self.ed_layers_load.append(SkipConnection(cp(t_attn), cp(norm)))
            self.ed_layers_pv.append(SkipConnection(cp(t_attn), cp(norm)))
            self.ed_layers_wind.append(SkipConnection(cp(t_attn), cp(norm)))
            self.ff_load.append(SkipConnection(cp(ff), cp(norm)))
            self.ff_pv.append(SkipConnection(cp(ff), cp(norm)))
            self.ff_wind.append(SkipConnection(cp(ff), cp(norm)))

    def forward(self, input, supports):
        load, pv, wind = torch.chunk(input, input.size()[1], dim=1)  #B C N T
        total = self.total_linear(input.permute(0, 2, 3, 1))  #B N T C
        load_ = self.load_linear(load.permute(0, 2, 3, 1))  #B N T C
        pv_ = self.pv_linear(pv.permute(0, 2, 3, 1))
        wind_ = self.wind_linear(wind.permute(0, 2, 3, 1))

        hiddens = []
        for i, (temporal_layer_load, temporal_layer_load1, spatial_layer_load, ed_layer_load, ff_load, temporal_layer_pv, spatial_layer_pv, ed_layer_pv, ff_pv, temporal_layer_wind, spatial_layer_wind, ed_layer_wind, ff_wind) \
                in enumerate(zip(self.temporal_layers_load, self.temporal_layers_load1, self.spatial_layers_load, self.ed_layers_load, self.ff_load,
                                                                                   self.temporal_layers_pv, self.spatial_layers_pv, self.ed_layers_pv, self.ff_pv,
                                                                                   self.temporal_layers_wind, self.spatial_layers_wind, self.ed_layers_wind, self.ff_wind)):
            load_ = temporal_layer_load1(load_)
            pv_ = temporal_layer_pv(pv_)
            wind_ = temporal_layer_wind(wind_)

            load_attn = spatial_layer_load(load_, supports)  # B, N, T, C
            pv_attn = spatial_layer_pv(pv_, supports)
            wind_attn = spatial_layer_wind(wind_, supports)

            load_attn = ed_layer_load(load_attn, total)
            pv_attn = ed_layer_pv(pv_attn, total)
            wind_attn = ed_layer_wind(wind_attn, total)

            load_ = ff_load(load_attn)
            pv_ = ff_pv(pv_attn)
            wind_ = ff_wind(wind_attn)

            hiddens.append(load_)
            hiddens.append(pv_)
            hiddens.append(wind_)
            hiddens.append(total)

        total_ = torch.concat((load_, pv_, wind_, total), dim=-1)
        out = self.proj(self.act(total_))

        return out.permute(0, 3, 1, 2), hiddens


class Expert_dynamic(nn.Module):
    def __init__(self, hidden_size, in_dim, layers, dropout=0.1, edproj=False, out_dim=3, spatial=False,
                 activation=nn.ReLU()):
        super(Expert_dynamic, self).__init__()
        self.spatial = spatial
        self.act = activation

        base_model = SkipConnection(self_Attention(hidden_size, hidden_size, dropout=dropout),
                                    LayerNorm(normalized_shape=(hidden_size,)))
        ff = SkipConnection(PositionwiseFeedForward(hidden_size, 4 * hidden_size, dropout=dropout),
                            LayerNorm(normalized_shape=(hidden_size,)))

        self.total_linear = nn.Linear(in_dim, hidden_size)
        self.load_linear = nn.Linear(1, hidden_size)
        self.pv_linear = nn.Linear(1, hidden_size)
        self.wind_linear = nn.Linear(1, hidden_size)

        self.temporal_layers_load = nn.ModuleList()
        self.temporal_layers_load1 = nn.ModuleList()
        self.spatial_layers_load = nn.ModuleList()
        self.ed_layers_load = nn.ModuleList()
        self.ff_load = nn.ModuleList()
        self.temporal_layers_pv = nn.ModuleList()
        self.spatial_layers_pv = nn.ModuleList()
        self.ed_layers_pv = nn.ModuleList()
        self.ff_pv = nn.ModuleList()
        self.temporal_layers_wind = nn.ModuleList()
        self.spatial_layers_wind = nn.ModuleList()
        self.ed_layers_wind = nn.ModuleList()
        self.ff_wind = nn.ModuleList()

        for i in range(layers):
            self.spatial_layers_load.append(cp(base_model))
            self.spatial_layers_pv.append(cp(base_model))
            self.spatial_layers_wind.append(cp(base_model))
            self.temporal_layers_load.append(cp(base_model))
            self.temporal_layers_load1.append(cp(base_model))
            self.temporal_layers_pv.append(cp(base_model))
            self.temporal_layers_wind.append(cp(base_model))
            self.ed_layers_load.append(cp(base_model))
            self.ed_layers_pv.append(cp(base_model))
            self.ed_layers_wind.append(cp(base_model))
            self.ff_load.append(cp(ff))
            self.ff_pv.append(cp(ff))
            self.ff_wind.append(cp(ff))

        self.proj = nn.Linear(hidden_size * 4, out_dim)

    def forward(self, x, prev_hidden=None):
        load, pv, wind = torch.chunk(x, x.size()[1], dim=1)  # B C N T
        total = self.total_linear(x.permute(0, 2, 3, 1))  # B N T C
        load_ = self.load_linear(load.permute(0, 2, 3, 1))  # B N T C
        pv_ = self.pv_linear(pv.permute(0, 2, 3, 1))
        wind_ = self.wind_linear(wind.permute(0, 2, 3, 1))
        hiddens = []
        for i, (s_layer_load, t_layer_load, ff_load, s_layer_pv, t_layer_pv, ff_pv, s_layer_wind, t_layer_wind, ff_wind) \
                in enumerate(zip(self.spatial_layers_load, self.temporal_layers_load, self.ff_load,
                                 self.spatial_layers_pv, self.temporal_layers_pv, self.ff_pv,
                                 self.spatial_layers_wind, self.temporal_layers_wind, self.ff_wind)):
            if not self.spatial:
                x_load = t_layer_load(load_)
                x_pv = t_layer_pv(pv_)
                x_wind = t_layer_wind(wind_)
                x_load = s_layer_load(x_load.transpose(1, 2))
                x_wind = s_layer_wind(x_wind.transpose(1, 2))
            else:
                x_load = s_layer_load(load_.transpose(1, 2))
                x_pv = s_layer_pv(pv_.transpose(1, 2))
                x_wind = s_layer_wind(wind_.transpose(1, 2))
                x_load = t_layer_load(x_load.transpose(1, 2)).transpose(1, 2)
                x_pv = t_layer_pv(x_pv.transpose(1, 2)).transpose(1, 2)
                x_wind = t_layer_wind(x_wind.transpose(1, 2)).transpose(1, 2)

            if prev_hidden is not None:
                x_load = self.ed_layers_load[i](x_load.transpose(1, 2), prev_hidden[0])
                x_pv = self.ed_layers_pv[i](x_pv, prev_hidden[1])
                x_load = x_load.transpose(1, 2)
                x_pv = x_pv.transpose(1, 2)

            x_load = ff_load(x_load.transpose(1, 2))
            x_pv = ff_pv(x_pv.transpose(1, 2))
            x_wind = ff_wind(x_wind.transpose(1, 2))

            hiddens.append(x_load)
            hiddens.append(x_pv)
            hiddens.append(x_wind)

        total_ = torch.concat((x_load, x_pv, x_wind, total), dim=-1)
        out = self.proj(self.act(total_))

        return out, hiddens

