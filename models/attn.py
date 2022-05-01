import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from utils.tools import default

def to(t):
    return {'dtype': t.dtype, 'device': t.device}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim+1] = split_dims
    return t.reshape(shape)

def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class MemAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.attn_dropout = nn.Dropout(0.1)
        # self.dropout = nn.Dropout(0.1)
        # self.memory_attn_dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, attn_mask, pos_emb=None ,kv_len=None, mem_len=None, lmem_len=None):
        B, H, L, E = q.shape
        scale = self.scale or 1. / sqrt(E)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * scale
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * scale
            pos_dots = shift(pos_dots)
            pos_dots = F.pad(pos_dots, (dots.shape[-1] - pos_dots.shape[-1], 0), value = 0.)
            dots = dots + pos_dots

        if attn_mask is not None:
            mask = attn_mask[:, None, :, None] * attn_mask[:, None, None, :]
            mask = F.pad(mask, (mem_len + lmem_len, 0), value = True)
            dots.masked_fill_(~mask, mask_value)

        total_mem_len = mem_len + lmem_len
        mask = torch.ones(L, L + total_mem_len, **to(q)).triu_(diagonal = 1 + total_mem_len).bool()

        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return out

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).double().to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.view(B, H, L_Q, -1)
        keys = keys.view(B, H, L_K, -1)
        values = values.view(B, H, L_K, -1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dim_heads = d_model//n_heads

    def forward(self, queries, keys, values, attn_mask, memories=None, pos_emb=None):
        if isinstance(self.inner_attention, MemAttention) is False:
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
                attn_mask
            )
            out = out.view(B, L, -1)
            return self.out_projection(out), attn
        else:
            b, t, e, h, dim_h = *queries.shape, self.n_heads, self.dim_heads

            memories = default(memories, (None, None))
            mem, lmem = memories

            init_mem = lambda: torch.empty(b, 0, e, **to(queries))
            mem = default(mem, init_mem)
            lmem = default(lmem, init_mem)
            mem_len, lmem_len = map(lambda t: t.shape[1], (mem, lmem))

            kv_input = torch.cat((lmem, mem, queries), dim=1)

            queries = self.query_projection(queries)

            kv_len = kv_input.shape[1]
            keys = self.key_projection(kv_input)
            values = self.value_projection(kv_input)

            merge_heads = lambda x: reshape_dim(x, -1, (-1, dim_h)).transpose(1, 2)
            q, k, v = map(merge_heads, (queries, keys, values))
            k, v = map(lambda x: x.expand(-1, h, -1, -1), (k, v))

            out = self.inner_attention(
                q,
                k,
                v,
                attn_mask,
                pos_emb,
                kv_len,
                mem_len,
                lmem_len
            )
            return self.out_projection(out), None
