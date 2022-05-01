import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from utils.tools import default
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, MemAttention, AttentionLayer
from models.embed import DataEmbedding
from models.memnet import MemoryAttentionNetwork

from collections import namedtuple
# structs
Memory = namedtuple('Memory', ['short', 'long'])

def to(t):
    return {'dtype': t.dtype, 'device': t.device}

class SWLHT(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, out_segment, smem_len,
                lmem_len, memory_layers = None, factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(SWLHT, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pred_len = int(out_len / out_segment)
        self.attn = attn
        self.output_attention = output_attention
        self.memory_layers = list(memory_layers)

        # input and memory emb
        seq_and_mem_len = seq_len + smem_len + lmem_len
        self.pos_emb = nn.Parameter(torch.zeros(n_heads, seq_and_mem_len, d_model // n_heads))

        self.memory_network = MemoryAttentionNetwork(d_model, len(self.memory_layers), smem_len, lmem_len, heads=n_heads)

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(MemAttention(False, factor),
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, memories=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        b, t, d = enc_out.shape

        memories = default(memories, (None, None))
        mem, lmem = memories

        num_memory_layers = len(self.memory_layers)

        mem = default(mem, lambda: torch.empty(num_memory_layers, b, 0, d, **to(enc_out)))
        lmem = default(lmem, lambda: torch.empty(b, 0, d, **to(enc_out)))

        mem_len = mem.shape[2]
        lmem_len = lmem.shape[1]

        total_len = mem_len + lmem_len + self.seq_len

        pos_emb = self.pos_emb[:, (self.seq_len - t):total_len]

        enc_out, hiddens, attns = self.encoder(enc_out, attn_mask=enc_self_mask,
                                      smemories=mem, lmemory=lmem, pos_emb=pos_emb, mem_layers=self.memory_layers)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        dec_out = dec_out # + self.dropout(highway_dec)

        hiddens = torch.stack(hiddens)
        next_memory = self.memory_network(lmem, mem, hiddens, detach_lmem=False)

        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], next_memory, attns
        else:
            return dec_out[:,-self.pred_len:,:], next_memory  # [B, L, D]
