import torch
import torch.nn as nn
import torch.nn.functional as F

def iterate_tensor(t):
    length = t.shape[0]
    for ind in range(length):
        yield t[ind]

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        # self.gru = nn.GRUCell(d_model, d_model)
        # self.elu = nn.ELU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, memories=None, pos_emb=None):
        # x [B, L, D]

        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask,
            memories = memories,
            pos_emb = pos_emb
        )
        x = x + self.dropout(new_x)

        # batch, dim = new_x.shape[0], x.shape[2]
        # gated_output = self.gru(
        #     new_x.reshape(-1, dim),
        #     x.reshape(-1, dim)
        # )
        # gated_output = gated_output.reshape(batch, -1, dim)
        # gated_output = x + self.dropout(self.elu(gated_output))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        # self.norm_hidden = nn.LayerNorm(512)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, smemories=None, lmemory=None, pos_emb=None, mem_layers=None):
        # x [B, L, D]

        mem_iter = iterate_tensor(smemories)
        hiddens = []

        attns = []
        if self.conv_layers is not None:
            for ind, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                layer_num = ind + 1
                use_memory = layer_num in mem_layers
                memories = (next(mem_iter), lmemory) if use_memory else None

                if use_memory:
                    hiddens.append(self.norm(x))

                x, attn = attn_layer(x, attn_mask=attn_mask, memories=memories, pos_emb=pos_emb)
                # print(x.shape) torch.Size([32, 96, 512]) ==> torch.Size([32, 49, 512])

                x = conv_layer(x)
                # print(x.shape) torch.Size([32, 49, 512]) ==> torch.Size([32, 26, 512])

                attns.append(attn)

            x, attn = self.attn_layers[-1](x)
            # print(x.shape) torch.Size([32, 26, 512])
            attns.append(attn)
        else:
            for ind, attn_layer in enumerate(self.attn_layers):
                layer_num = ind + 1
                use_memory = layer_num in mem_layers
                memories = (next(mem_iter), lmemory) if use_memory else None

                if use_memory:
                    hiddens.append(self.norm(x))

                x, attn = attn_layer(x, attn_mask=attn_mask, memories=memories, pos_emb=pos_emb)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, hiddens, attns
