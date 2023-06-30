import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


def index(feat, uv):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, N, 2] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''

    (B, N, _) = uv.shape
    C = feat.shape[1]

    if uv.shape[-1] == 3:
        # uv = uv[:,:,[2,1,0]]
        # uv = uv * torch.tensor([1.0,-1.0,1.0]).type_as(uv)[None,None,...]
        uv = uv.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1, 3]
    else:
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]

    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples.view(B, C, N)  # [B, C, N]


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


def graph_upsample(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
        x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
        return x
    else:
        return x


def graph_conv_cheby(x, cl, L, K=3):
    # parameters
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = Chebyshev order & support size
    B, V, Fin = x.size()
    B, V, Fin = int(B), int(V), int(Fin)

    # transform to Chebyshev basis
    x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin * B])  # V x Fin*B
    x = x0.unsqueeze(0)  # 1 x V x Fin*B

    def concat(x, x_):
        x_ = x_.unsqueeze(0)  # 1 x V x Fin*B
        return torch.cat((x, x_), 0)  # K x V x Fin*B

    if K > 1:
        x1 = torch.mm(L, x0)  # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for k in range(2, K):
        x2 = 2 * torch.mm(L, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
        x0, x1 = x1, x2

    x = x.view([K, V, Fin, B])  # K x V x Fin x B
    x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    x = x.view([B * V, Fin * K])  # B*V x Fin*K

    # Compose linearly Fin features to get Fout features
    x = cl(x)  # B*V x Fout
    x = x.view([B, V, -1])  # B x V x Fout

    return x


def graph_avg_pool(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.AvgPool1d(p)(x)  # B x F x V/p
        x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
        return x
    else:
        return x


def sparse_python_to_torch(sp_python):
    L = sp_python.tocoo()
    indices = np.column_stack((L.row, L.col)).T
    indices = indices.astype(np.int64)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    L_data = L.data.astype(np.float32)
    L_data = torch.from_numpy(L_data)
    L_data = L_data.type(torch.FloatTensor)
    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
#     if torch.cuda.is_available():
#         L = L.cuda()

    return L


class GCN_ResBlock(nn.Module):
    # x______________conv + norm (optianal)_____________ x ____activate
    #  \____conv____activate____norm____conv____norm____/
    def __init__(self, in_dim, out_dim, mid_dim,
                 graph_L, graph_k,
                 drop_out=0.01):
        super(GCN_ResBlock, self).__init__()
        if isinstance(graph_L, np.ndarray):
            self.register_buffer('graph_L',
                                 torch.from_numpy(graph_L).float(),
                                 persistent=False)
        else:
            self.register_buffer('graph_L',
                                 sparse_python_to_torch(graph_L).to_dense(),
                                 persistent=False)

        self.graph_k = graph_k
        self.in_dim = in_dim

        self.norm1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim * graph_k, mid_dim)
        self.norm2 = nn.LayerNorm(out_dim, eps=1e-6)
        self.fc2 = nn.Linear(mid_dim * graph_k, out_dim)
        self.dropout = nn.Dropout(drop_out)
        self.shortcut = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        # x : B x V x f
        assert x.shape[-1] == self.in_dim

        x1 = F.relu(self.norm1(x))
        x1 = graph_conv_cheby(x1, self.fc1, self.graph_L, K=self.graph_k)  # [bs, V, mid_dim]
        x1 = F.relu(self.norm2(x1))
        x1 = graph_conv_cheby(x1, self.fc2, self.graph_L, K=self.graph_k)  # [bs, V, out_dim]
        x1 = self.dropout(x1)
        x2 = self.shortcut(x)    # [bs, V, out_dim]

        return self.norm3(x1 + x2)


class GraphLayer(nn.Module):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 graph_L=None,
                 graph_k=2,
                 graph_layer_num=3,
                 drop_out=0.01):
        super().__init__()
        assert graph_k > 1

        self.GCN_blocks = nn.ModuleList()
        self.GCN_blocks.append(GCN_ResBlock(in_dim, out_dim, out_dim, graph_L, graph_k, drop_out))
        for i in range(graph_layer_num - 1):
            self.GCN_blocks.append(GCN_ResBlock(out_dim, out_dim, out_dim, graph_L, graph_k, drop_out))

        for m in self.modules():
            weights_init(m)

    def forward(self, verts_f):
        # verts_f:[bs, V, in_dim]
        for i in range(len(self.GCN_blocks)):
            verts_f = self.GCN_blocks[i](verts_f)
            if i != (len(self.GCN_blocks) - 1):
                verts_f = F.relu(verts_f)

        return verts_f  # [bs, V, out_dim]


class GCN_vert_convert():
    def __init__(self, vertex_num=1, graph_perm_reverse=[0], graph_perm=[0]):
        self.graph_perm_reverse = graph_perm_reverse[:vertex_num]
        self.graph_perm = graph_perm

    def vert_to_GCN(self, x):
        # x: B x v x f
        return x[:, self.graph_perm]

    def GCN_to_vert(self, x):
        # x: B x v x f
        return x[:, self.graph_perm_reverse]


class MLP_res_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        # [bs, V, f_dim]
        x = x + self._ff_block(self.layer_norm(x))
        return x  # [bs, V, f_dim]


class SelfAttn(nn.Module):
    def __init__(self, f_dim, hid_dim=None, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads
        if hid_dim is None:
            hid_dim = f_dim

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)

        self.layer_norm = nn.LayerNorm(f_dim, eps=1e-6)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.ff = MLP_res_block(f_dim, hid_dim, dropout)

    def self_attn(self, x):
        # x:[bs, V, f_dim]
        BS, V, f = x.shape

        q = self.w_qs(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        k = self.w_ks(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        v = self.w_vs(x).view(BS, -1, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.norm  # bs, h, V, V
        attn = F.softmax(attn, dim=-1)  # bs, h, V, V
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(BS, V, -1)   # [bs, V, f_dim]
        out = self.dropout2(self.fc(out))
        return out   # [bs, V, f_dim]

    def forward(self, x):
        BS, V, f = x.shape
        assert f == self.f_dim

        x = x + self.self_attn(self.layer_norm(x))    # [bs, V, f_dim]
        x = self.ff(x)    # [bs, V, f_dim]

        return x


class img_feat_to_grid(nn.Module):
    def __init__(self, img_size, img_f_dim, grid_size, grid_f_dim, n_heads=4, dropout=0.01):
        super().__init__()
        self.img_f_dim = img_f_dim
        self.img_size = img_size
        self.grid_f_dim = grid_f_dim
        self.grid_size = grid_size
        self.position_embeddings = nn.Embedding(grid_size * grid_size, grid_f_dim)

        patch_size = img_size // grid_size
        self.proj = nn.Conv2d(img_f_dim, grid_f_dim, kernel_size=patch_size, stride=patch_size)
        self.self_attn = SelfAttn(grid_f_dim, n_heads=n_heads, hid_dim=grid_f_dim, dropout=dropout)

    def forward(self, img):
        # img:[bs, img_f_dim, H, W]
        bs = img.shape[0]
        assert img.shape[1] == self.img_f_dim
        assert img.shape[2] == self.img_size
        assert img.shape[3] == self.img_size

        position_ids = torch.arange(self.grid_size * self.grid_size, dtype=torch.long, device=img.device)  # [gird_size*gird_size,]
        position_ids = position_ids.unsqueeze(0).repeat(bs, 1)              # [bs, gird_size*gird_size]
        position_embeddings = self.position_embeddings(position_ids)        # [bs, gird_size*gird_size, grid_f_dim]

        grid_feat = F.relu(self.proj(img))        # [bs, grid_f_dim, gird_size, gird_size]
        grid_feat = grid_feat.view(bs, self.grid_f_dim, -1).transpose(-1, -2)   # [bs, gird_size*gird_size, grid_f_dim]
        grid_feat = grid_feat + position_embeddings    # [bs, gird_size*gird_size, grid_f_dim]

        grid_feat = self.self_attn(grid_feat)          # [bs, gird_size*gird_size, grid_f_dim]

        return grid_feat


class img_attn(nn.Module):
    def __init__(self, verts_f_dim, img_f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        self.img_f_dim = img_f_dim
        self.verts_f_dim = verts_f_dim

        self.fc = nn.Linear(img_f_dim, verts_f_dim)
        self.Attn = SelfAttn(verts_f_dim, n_heads=n_heads, hid_dim=verts_f_dim, dropout=dropout)

    def forward(self, verts_f, img_f):
        # verts_f:[bs, V, dim]
        # img_f:[bs, H*W, img_f_dim]
        assert verts_f.shape[2] == self.verts_f_dim
        assert img_f.shape[2] == self.img_f_dim
        assert verts_f.shape[0] == img_f.shape[0]
        V = verts_f.shape[1]

        img_f = self.fc(img_f)   # [bs, H*W, dim]

        x = torch.cat([verts_f, img_f], dim=1)   # [bs, V+H*W, dim]
        x = self.Attn(x)                         # [bs, V+H*W, dim]

        verts_f = x[:, :V]

        return verts_f    # [bs, V, dim]


class img_ex(nn.Module):
    def __init__(self, img_size, img_f_dim,
                 grid_size, grid_f_dim,
                 verts_f_dim,
                 n_heads=4,
                 dropout=0.01):
        super().__init__()
        self.verts_f_dim = verts_f_dim
        self.encoder = img_feat_to_grid(img_size, img_f_dim, grid_size, grid_f_dim, n_heads, dropout)
        self.attn = img_attn(verts_f_dim, grid_f_dim, n_heads=n_heads, dropout=dropout)

        for m in self.modules():
            weights_init(m)

    def forward(self, img, verts_f):
        # img:[bs, img_f_dim, H, W]
        # verts_f:[bs, V, dim]
        assert verts_f.shape[2] == self.verts_f_dim
        grid_feat = self.encoder(img)   # [bs, gird_size*gird_size, grid_f_dim]
        verts_f = self.attn(verts_f, grid_feat)   # [bs, V, dim]
        return verts_f


class inter_attn(nn.Module):
    def __init__(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()

        self.L_self_attn_layer = SelfAttn(f_dim, n_heads=n_heads, hid_dim=f_dim, dropout=dropout)
        self.R_self_attn_layer = SelfAttn(f_dim, n_heads=n_heads, hid_dim=f_dim, dropout=dropout)
        self.build_inter_attn(f_dim, n_heads, d_q, d_v, dropout)

        for m in self.modules():
            weights_init(m)

    def build_inter_attn(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.layer_norm1 = nn.LayerNorm(f_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(f_dim, eps=1e-6)
        self.ffL = MLP_res_block(f_dim, f_dim, dropout)
        self.ffR = MLP_res_block(f_dim, f_dim, dropout)

    def inter_attn(self, Lf, Rf, mask_L2R=None, mask_R2L=None):
        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lf2 = self.layer_norm1(Lf)   # [bs, V, f_dim]
        Rf2 = self.layer_norm2(Rf)   # [bs, V, f_dim]

        Lq = self.w_qs(Lf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Lk = self.w_ks(Lf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Lv = self.w_vs(Lf2).view(BS, V, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        Rq = self.w_qs(Rf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Rk = self.w_ks(Rf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Rv = self.w_vs(Rf2).view(BS, V, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn_R2L = torch.matmul(Lq, Rk.transpose(-1, -2)) / self.norm  # bs, h, V, V
        attn_L2R = torch.matmul(Rq, Lk.transpose(-1, -2)) / self.norm  # bs, h, V, V

        if mask_L2R is not None:
            attn_L2R = attn_L2R.masked_fill(mask_L2R == 0, -1e9)
        if mask_R2L is not None:
            attn_R2L = attn_R2L.masked_fill(mask_R2L == 0, -1e9)

        attn_R2L = F.softmax(attn_R2L, dim=-1)  # bs, h, V, V
        attn_L2R = F.softmax(attn_L2R, dim=-1)  # bs, h, V, V

        attn_R2L = self.dropout1(attn_R2L)
        attn_L2R = self.dropout1(attn_L2R)

        feat_L2R = torch.matmul(attn_L2R, Lv).transpose(1, 2).contiguous().view(BS, V, -1)
        feat_R2L = torch.matmul(attn_R2L, Rv).transpose(1, 2).contiguous().view(BS, V, -1)

        feat_L2R = self.dropout2(self.fc(feat_L2R))
        feat_R2L = self.dropout2(self.fc(feat_R2L))

        Lf = self.ffL(Lf + feat_R2L)
        Rf = self.ffR(Rf + feat_L2R)

        return Lf, Rf   # [bs, V, f_dim]

    def forward(self, Lf, Rf, mask_L2R=None, mask_R2L=None):
        # Lf/Rf:[bs, V, f_dim]
        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lf = self.L_self_attn_layer(Lf)   # [bs, V, f_dim]
        Rf = self.R_self_attn_layer(Rf)   # [bs, V, f_dim]
        Lf, Rf = self.inter_attn(Lf, Rf, mask_L2R, mask_R2L)

        return Lf, Rf    # [bs, V, f_dim]


class DualGraphLayer(nn.Module):
    def __init__(self,
                 verts_in_dim=256,
                 verts_out_dim=256,
                 graph_L_Left=None,
                 graph_L_Right=None,
                 graph_k=2,
                 graph_layer_num=4,
                 img_size=64,
                 img_f_dim=256,
                 grid_size=8,
                 grid_f_dim=128,
                 n_heads=4,
                 dropout=0.01):
        super().__init__()
        self.verts_num = graph_L_Left.shape[0]
        self.verts_in_dim = verts_in_dim
        self.img_size = img_size
        self.img_f_dim = img_f_dim

        self.position_embeddings = nn.Embedding(self.verts_num, self.verts_in_dim)

        self.graph_left = GraphLayer(verts_in_dim, verts_out_dim,
                                     graph_L_Left, graph_k, graph_layer_num,
                                     dropout)
        self.graph_right = GraphLayer(verts_in_dim, verts_out_dim,
                                      graph_L_Right, graph_k, graph_layer_num,
                                      dropout)

        self.img_ex_left = img_ex(img_size, img_f_dim,
                                  grid_size, grid_f_dim,
                                  verts_out_dim,
                                  n_heads=n_heads,
                                  dropout=dropout)
        self.img_ex_right = img_ex(img_size, img_f_dim,
                                   grid_size, grid_f_dim,
                                   verts_out_dim,
                                   n_heads=n_heads,
                                   dropout=dropout)
        self.attn = inter_attn(verts_out_dim, n_heads=n_heads, dropout=dropout)

    def forward(self, Lf, Rf, img_f):
        # Lf/Rf:[bs, V, in_dim]
        # img_f:[bs, img_f_dim, H, W]
        BS1, V, f = Lf.shape
        assert V == self.verts_num
        assert f == self.verts_in_dim
        BS2, V, f = Rf.shape
        assert V == self.verts_num
        assert f == self.verts_in_dim
        BS3, C, H, W = img_f.shape
        assert C == self.img_f_dim
        assert H == self.img_size
        assert W == self.img_size
        assert BS1 == BS2
        assert BS2 == BS3
        BS = BS1

        position_ids = torch.arange(self.verts_num, dtype=torch.long, device=Lf.device)
        position_ids = position_ids.unsqueeze(0).repeat(BS, 1)    # [bs, V]
        position_embeddings = self.position_embeddings(position_ids)  # [bs, V, in_dim]
        Lf = Lf + position_embeddings    # [bs, V, in_dim]
        Rf = Rf + position_embeddings    # [bs, V, in_dim]

        Lf = self.graph_left(Lf)         # [bs, V, out_dim]
        Rf = self.graph_right(Rf)        # [bs, V, out_dim]

        Lf = self.img_ex_left(img_f, Lf)   # [bs, V, out_dim]
        Rf = self.img_ex_right(img_f, Rf)  # [bs, V, out_dim]

        Lf, Rf = self.attn(Lf, Rf)   # [bs, V, out_dim]

        return Lf, Rf


class DualGraph(nn.Module):
    def __init__(self,
                 verts_in_dim=[512, 256, 128],
                 verts_out_dim=[256, 128, 64],
                 graph_L_Left=None,
                 graph_L_Right=None,
                 graph_k=[2, 2, 2],
                 graph_layer_num=[4, 4, 4],
                 img_size=[16, 32, 64],
                 img_f_dim=[256, 256, 256],
                 grid_size=[8, 8, 16],
                 grid_f_dim=[256, 128, 64],
                 n_heads=4,
                 dropout=0.01):
        super().__init__()
        for i in range(len(verts_in_dim) - 1):
            assert verts_out_dim[i] == verts_in_dim[i + 1]
        for i in range(len(verts_in_dim) - 1):
            assert graph_L_Left[i + 1].shape[0] == 2 * graph_L_Left[i].shape[0]
            assert graph_L_Right[i + 1].shape[0] == 2 * graph_L_Right[i].shape[0]

        self.layers = nn.ModuleList()
        for i in range(len(verts_in_dim)):
            self.layers.append(DualGraphLayer(verts_in_dim=verts_in_dim[i],
                                              verts_out_dim=verts_out_dim[i],
                                              graph_L_Left=graph_L_Left[i],
                                              graph_L_Right=graph_L_Right[i],
                                              graph_k=graph_k[i],
                                              graph_layer_num=graph_layer_num[i],
                                              img_size=img_size[i],
                                              img_f_dim=img_f_dim[i],
                                              grid_size=grid_size[i],
                                              grid_f_dim=grid_f_dim[i],
                                              n_heads=n_heads,
                                              dropout=dropout))

    def forward(self, Lf, Rf, img_f_list):
        # Lf/Rf:[bs, 63, 512]
        # img_f_list:torch.Size([bs, 256, 8, 8])
        #            torch.Size([bs, 256, 16, 16])
        #            torch.Size([bs, 256, 32, 32])
        assert len(img_f_list) == len(self.layers)
        for i in range(len(self.layers)):
            Lf, Rf = self.layers[i](Lf, Rf, img_f_list[i])

            if i != len(self.layers) - 1:
                Lf = graph_upsample(Lf, 2)
                Rf = graph_upsample(Rf, 2)

        return Lf, Rf


class Decoder(nn.Module):
    def __init__(self,
                 global_feature_dim=2048,
                 f_in_Dim=[256, 256, 256, 256],
                 f_out_Dim=[128, 64, 32],
                 gcn_in_dim=[256, 128, 128],
                 gcn_out_dim=[128, 128, 64],
                 graph_k=2,
                 graph_layer_num=4,
                 left_graph_dict={},
                 right_graph_dict={},
                 vertex_num=778,
                 dense_coor=None,
                 num_attn_heads=4,
                 upsample_weight=None,
                 dropout=0.05):
        super(Decoder, self).__init__()
        assert len(f_in_Dim) == 4
        f_in_Dim = f_in_Dim[:-1]
        # assert len(gcn_in_dim) == 3
        for i in range(len(gcn_out_dim) - 1):
            assert gcn_out_dim[i] == gcn_in_dim[i + 1]

        graph_dict = {'left': left_graph_dict, 'right': right_graph_dict}
        graph_dict['left']['coarsen_graphs_L'].reverse()
        graph_dict['right']['coarsen_graphs_L'].reverse()
        graph_L = {}
        for hand_type in ['left', 'right']:
            graph_L[hand_type] = graph_dict[hand_type]['coarsen_graphs_L']

        self.vNum_in = graph_L['left'][0].shape[0]      # 63
        self.vNum_out = graph_L['left'][2].shape[0]     # 252
        self.vNum_all = graph_L['left'][-1].shape[0]    # 1008
        self.vNum_mano = vertex_num
        self.gf_dim = global_feature_dim
        self.gcn_in_dim = gcn_in_dim
        self.gcn_out_dim = gcn_out_dim

        if dense_coor is not None:
            dense_coor = torch.from_numpy(dense_coor).float()    # [778, 3]
            self.register_buffer('dense_coor', dense_coor)

        self.converter = {}
        for hand_type in ['left', 'right']:
            self.converter[hand_type] = GCN_vert_convert(vertex_num=self.vNum_mano,
                                                         graph_perm_reverse=graph_dict[hand_type]['graph_perm_reverse'],
                                                         graph_perm=graph_dict[hand_type]['graph_perm'])

        self.dual_gcn = DualGraph(verts_in_dim=self.gcn_in_dim,
                                  verts_out_dim=self.gcn_out_dim,
                                  graph_L_Left=graph_L['left'][:3],
                                  graph_L_Right=graph_L['right'][:3],
                                  graph_k=[graph_k, graph_k, graph_k],
                                  graph_layer_num=[graph_layer_num, graph_layer_num, graph_layer_num],
                                  img_size=[8, 16, 32],
                                  img_f_dim=f_in_Dim,
                                  grid_size=[8, 8, 8],
                                  grid_f_dim=f_out_Dim,
                                  n_heads=num_attn_heads,
                                  dropout=dropout)

        self.unsample_layer = nn.Linear(self.vNum_out, self.vNum_mano, bias=False)
        if upsample_weight is not None:
            state = {'weight': upsample_weight.to(self.unsample_layer.weight.data.device)}
            self.unsample_layer.load_state_dict(state)
        else:
            weights_init(self.unsample_layer)

        self.gf_layer_left = nn.Sequential(*(nn.Linear(self.gf_dim, self.gcn_in_dim[0] - 3),
                                             nn.LayerNorm(self.gcn_in_dim[0] - 3, eps=1e-6)))
        self.gf_layer_right = nn.Sequential(*(nn.Linear(self.gf_dim, self.gcn_in_dim[0] - 3),
                                              nn.LayerNorm(self.gcn_in_dim[0] - 3, eps=1e-6)))
        weights_init(self.gf_layer_left)
        weights_init(self.gf_layer_right)
        # weights_init(self.coord_head)
        # weights_init(self.avg_head)
        # weights_init(self.params_head)

    def get_upsample_weight(self):
        return self.unsample_layer.weight.data

    def get_converter(self):
        return self.converter

    def get_hand_pe(self, bs, num=None):
        if num is None:
            num = self.vNum_in
        dense_coor = self.dense_coor.repeat(bs, 1, 1) * 2 - 1  # [bs, 778, 3], -1~1
        pel = self.converter['left'].vert_to_GCN(dense_coor)   # [bs, 1008, 3]
        pel = graph_avg_pool(pel, p=pel.shape[1] // num)       # [bs, 63, 3]
        per = self.converter['right'].vert_to_GCN(dense_coor)  # [bs, 1008, 3]
        per = graph_avg_pool(per, p=per.shape[1] // num)       # [bs, 63, 3]
        return pel, per


    def forward(self, x, fmaps):
        assert x.shape[1] == self.gf_dim
        fmaps = fmaps[:-1]
        # x:[bs, 2048]
        # fmaps: torch.Size([bs, 256, 8, 8])
        #        torch.Size([bs, 256, 16, 16])
        #        torch.Size([bs, 256, 32, 32])
        bs = x.shape[0]

        pel, per = self.get_hand_pe(bs, num=self.vNum_in)     # [bs, 63, 3]
        Lf = torch.cat([self.gf_layer_left(x).unsqueeze(1).repeat(1, self.vNum_in, 1), pel], dim=-1)    # [bs, 63, 512]
        Rf = torch.cat([self.gf_layer_right(x).unsqueeze(1).repeat(1, self.vNum_in, 1), per], dim=-1)   # [bs, 63, 512]

        Lf, Rf = self.dual_gcn(Lf, Rf, fmaps)   # B x N x C
        return Lf, Rf   # [bs, 252, 64]


def get_decoder(encoder_info):
    current_path = os.path.abspath(__file__)
    dir_path = os.path.abspath(current_path + '/../../../../..')
    with open(os.path.join(dir_path, 'pretrained', 'graph_left.pkl'), 'rb') as file:
        left_graph_dict = pickle.load(file)
    with open(os.path.join(dir_path, 'pretrained', 'graph_left.pkl'), 'rb') as file:
        right_graph_dict = pickle.load(file)
    with open(os.path.join(dir_path, 'pretrained', 'v_color.pkl'), 'rb') as file:
        dense_coor = pickle.load(file)
    with open(os.path.join(dir_path, 'pretrained', 'upsample.pkl'), 'rb') as file:
        upsample_weight = pickle.load(file)
    upsample_weight = torch.from_numpy(upsample_weight).float()
    model = Decoder(
        global_feature_dim=encoder_info['global_feature_dim'],
        f_in_Dim=encoder_info['fmaps_dim'],
        f_out_Dim=[256, 128, 64],
        gcn_in_dim=[512, 256, 128],
        gcn_out_dim=[256, 128, 64],
        graph_k=2,
        graph_layer_num=4,
        vertex_num=778,
        dense_coor=dense_coor,
        left_graph_dict=left_graph_dict,
        right_graph_dict=right_graph_dict,
        num_attn_heads=4,
        upsample_weight=upsample_weight,
        dropout=0.05
    )
    return model


if __name__ == "__main__":
    from common.nets.encoder import get_encoder
    encoder, mid_model = get_encoder()
    hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = encoder(torch.randn(1, 3, 256, 256))
    global_feature, fmaps = mid_model(img_fmaps, hms_fmaps, dp_fmaps)
    decoder = get_decoder(mid_model.get_info())
    lf, rf = decoder(global_feature, fmaps)
    print(lf.shape)
    print(rf.shape)