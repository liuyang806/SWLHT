import torch
from torch import nn
import torch.nn.functional as F
import math


def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim+1] = split_dims
    return t.reshape(shape)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def queue_fifo(*args, length, dim=-2):
    queue = torch.cat(args, dim=dim)
    if length > 0:
        return split_at_index(dim, -length, queue)

    device = queue.device
    shape = list(queue.shape)
    shape[dim] = 0
    return queue, torch.empty(shape, device = device)

def init_parameter(shape, dim):
    t = torch.zeros(shape)
    std = 1 / math.sqrt(dim)
    t.uniform_(-std, std)
    return nn.Parameter(t)

class nBRC(nn.Module):
    def __init__(self, dims, hidden_dims):
        super().__init__()
        self.Ua = nn.Linear(dims, hidden_dims)
        self.Wa = nn.Linear(dims, hidden_dims)
        self.Uc = nn.Linear(dims, hidden_dims)
        self.Wc = nn.Linear(dims, hidden_dims)
        self.U  = nn.Linear(dims, hidden_dims)

    def forward(self, x, h):
        l = lambda linear, tensor: F.linear(tensor, linear.weight.clone(), linear.bias.clone())

        a = 1 + torch.tanh(l(self.Ua, x) + l(self.Wa, h))
        c = torch.sigmoid(l(self.Uc, x) + l(self.Wc, h))
        return c * h + (1 - c) * torch.tanh(l(self.U, x) + a * h)

def linear_attn(q, k, v):
    q, k = q.softmax(dim=-1), k.softmax(dim=-2)
    context = torch.einsum('bhnd,bhne->bhde', k, v)
    out = torch.einsum('bhnd,bhde->bhne', q, context)
    return out

class LinearSelfAttention(nn.Module):
    def __init__(self, dim, depth, heads=8):
        super().__init__()
        self.dim_head = dim // heads
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        self.to_q = init_parameter((dim, dim), dim)
        self.to_kv = init_parameter((dim, 2 * dim), dim)
        self.to_out = init_parameter((dim, dim), dim)

    def forward(self, x, hiddens=None):
        dim_head = self.dim_head
        w_q, w_kv, w_out = map(torch.clone, (self.to_q, self.to_kv, self.to_out))

        normed_lmem = self.norm(x)
        q = torch.einsum('bnd,de->bne', normed_lmem, w_q)

        kv_input = torch.cat((normed_lmem, hiddens), dim=1)
        k, v = torch.einsum('bnd,de->bne', kv_input, w_kv).chunk(2, dim=-1)

        q, k, v = map(lambda t: reshape_dim(t, -1, (-1, dim_head)).transpose(-2, -3), (q, k, v))

        out = linear_attn(q, k, v)

        out = out.transpose(2, 3).reshape_as(x)
        out = torch.einsum('bnd,de->bne', out, w_out)
        return out


class MemoryAttentionNetwork(nn.Module):
    def __init__(self, dim, num_memory_depth, mem_len, lmem_len, heads = 8, num_mem_kv = 0, mem_write_iters = 2):
        super().__init__()
        self.num_memory_depth = num_memory_depth
        self.mem_len = mem_len
        self.lmem_len = lmem_len

        self.dim = dim
        dim_head = dim // heads
        self.dim_head = dim_head

        self.depth_emb = init_parameter((num_memory_depth, 1, 1, 1), dim)
        self.init_lmem = init_parameter((1, 1, dim), dim)
        self.lmem_pos_emb = init_parameter((1, lmem_len, dim), dim)

        self.mem_kv = init_parameter((1, num_mem_kv, dim), dim)

        self.attn = LinearSelfAttention(dim, num_memory_depth, heads = heads)
        self.gate = nBRC(dim, dim)
        self.mem_write_iters = mem_write_iters
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.85)

    def forward(self, lmem, smem, hiddens, detach_lmem = False):

        batch, dim, dim_head, mem_depth, lmem_len = lmem.shape[0], self.dim, self.dim_head, self.num_memory_depth, self.lmem_len

        # properly detach hidden state, and detach long term memory if truncate signal is given

        hiddens = hiddens.detach()
        # print('hiddens.shape: ', hiddens.shape)
        # hiddens.shape: torch.Size([3, 4, 512, 512])

        if detach_lmem:
            lmem = lmem.detach()

        # initialize long term memory state if none provided

        if lmem is None or lmem.shape[1] == 0:
            lmem = self.init_lmem.clone().expand(batch, lmem_len, -1)
        # print('lmem.shape: ', lmem.shape)
        # lmem.shape: torch.Size([4, 128, 512])

        # use efficient linear attention for updating long term memory

        next_lmem = lmem + self.lmem_pos_emb
        # print('next_lmem.shape: ', next_lmem.shape)
        # next_lmem.shape: torch.Size([4, 128, 512])

        hiddens_and_smem = torch.cat((smem, hiddens), dim=-2)
        # print('hiddens_and_smem.shape: ', hiddens_and_smem.shape)
        # hiddens_and_smem.shape: torch.Size([3, 4, 512, 512])
        # ==>>
        # hiddens_and_smem.shape: torch.Size([3, 4, 1024, 512])
        all_hiddens = (hiddens_and_smem + self.depth_emb).transpose(0, 1).reshape(batch, -1, dim)
        # print('all_hiddens.shape: ', all_hiddens.shape)
        # all_hiddens.shape: torch.Size([4, 1536, 512])
        # ==>>
        # all_hiddens.shape: torch.Size([4, 3072, 512])
        all_hiddens = torch.cat((all_hiddens, self.mem_kv.expand(batch, -1, -1)), dim=1)
        # print('all_hiddens.shape: ', all_hiddens.shape)
        # all_hiddens.shape: torch.Size([4, 1536, 512])
        # ==>>
        # all_hiddens.shape: torch.Size([4, 3072, 512])

        for _ in range(self.mem_write_iters):
            attn_out = self.attn(next_lmem, hiddens = all_hiddens)
            # print('attn_out.shape: ', attn_out.shape)
            # attn_out.shape: torch.Size([4, 128, 512])
            next_lmem = self.gate(attn_out, next_lmem)

        # print('next_lmem.shape: ', next_lmem.shape)
        # next_lmem.shape:  torch.Size([32, 12, 512])

        # fifo queue the short term memory
        _, next_mem = queue_fifo(smem, hiddens, length = self.mem_len, dim = 2)

        # print('next_mem.shape: ', next_mem.shape)
        # next_mem.shape:  torch.Size([2, 32, 48, 512])

        return self.dropout(self.norm1(next_mem.detach())), \
               self.dropout(self.norm2(next_lmem))
