from Modules import PositionalEncoding
from Model import EncoderLayer,DecoderLayer
import torch
import torch.nn as nn
import copy

def clone(module,N):
    """
    Produce identical module list
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self,head,d_model,d_k,d_v,d_ff,n_layers):
        super(Encoder, self).__init__()
        self.LayerStacks=clone(EncoderLayer(head,d_model,d_k,d_v,d_ff),n_layers)
        self.layernorm=nn.LayerNorm(d_model)

    def forward(self, enc_input,attn_mask):
        for layer in self.LayerStacks:
            enc_input=layer(enc_input,attn_mask)
        return self.layernorm(enc_input)

class Decoder(nn.Module):
    def __init__(self,head,d_model,d_k,d_v,d_ff,n_layers):
        super(Decoder, self).__init__()
        self.LayerStacks=clone(DecoderLayer(head,d_model,d_k,d_v,d_ff),n_layers)
        self.layernorm=nn.LayerNorm(d_model)

    def forward(self, enc_output,dec_input,slf_attn_mask,enc_dec_attn_mask):
        for layer in self.LayerStacks:
            dec_input=layer(enc_output,dec_input,slf_attn_mask,enc_dec_attn_mask)
            # print("dec input size:{}".format(dec_input.size()))
        return self.layernorm(dec_input)

# def get_subsequent_mask(seq_len):
#     '''
#     :param seq_len: length of src sentence or trg sentence
#     :return: seq_len*seq_len matrix:
#     {
#         [1,0,...,0],
#         [1,1,...,0],
#         .
#         .
#         [1,1,...,1]
#     } every line is the mask in different steps.(1-no mask; 0-mask)
#     '''
#     subsequent_mask=torch.triu(torch.ones(1,seq_len,seq_len),diagonal=1)
#     subsequent_mask=(1-subsequent_mask).int()
#     return subsequent_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).int()
    return subsequent_mask



def get_pad_mask(seq_idx, pad_idx):
    '''
    :param seq:[batch,seq_len] every elem represents a word idx in vocab
    :param pad_idx: a scalar represents the idx of <pad> word in vocab
    :return: [batch,seq_len] masks to filter the <pad> word in vocab
    '''
    return (seq_idx != pad_idx).unsqueeze(-2).int()

class Fus_Embeddings(nn.Module):
    def __init__(self,n_src_embeddings,d_model):
        super(Fus_Embeddings, self).__init__()
        self.embedding_lookup=nn.Embedding(n_src_embeddings,d_model)
        self.pos_enc=PositionalEncoding(d_model)

    def forward(self,input_idx):
        '''
        :param input_idx: index of word in sentence. shape:[batch,seq_len]
        :return: merge of embeddings and positional encoding. shape:[batch,seq_len,d_model]
        '''
        x=self.embedding_lookup(input_idx)
        return self.pos_enc(x)

# Embed=Fus_Embeddings(5,8)
# input_idx=torch.LongTensor([[1,2],[3,4]])
# res=Embed(input_idx)
# print(res.size())

# trg_seq=torch.Tensor([[1,2,1,4],[3,1,2,4]])
# subseq_mask=get_subsequent_mask(trg_seq.size(1))
# print(subseq_mask.size())
# pad_mask=get_pad_mask(trg_seq,1)
# print(pad_mask)
# merge_mask=subseq_mask & pad_mask
# print(merge_mask)