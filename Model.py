import torch
import torch.nn as nn
from Modules import MultiHeadAttention,PositionWiseFFN

class EncoderLayer(nn.Module):
    def __init__(self,head,d_model,d_k,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.self_atten=MultiHeadAttention(head,d_model,d_k,d_v)
        self.ffn=PositionWiseFFN(d_model,d_ff)

    def forward(self, x,attn_mask):
        x=self.self_atten(x,x,x,attn_mask)
        res=self.ffn(x)
        return res

class DecoderLayer(nn.Module):
    def __init__(self,head,d_model,d_k,d_v,d_ff):
        super(DecoderLayer, self).__init__()
        self.self_atten=MultiHeadAttention(head,d_model,d_k,d_v)
        self.enc_atten=MultiHeadAttention(head,d_model,d_k,d_v)
        self.ffn=PositionWiseFFN(d_model,d_ff)

    def forward(self,enc_output,dec_input,slf_attn_mask,enc_dec_attn_mask):
        dec_output=self.self_atten(dec_input,dec_input,dec_input,slf_attn_mask)
        dec_output=self.enc_atten(dec_output,enc_output,enc_output,enc_dec_attn_mask)
        res=self.ffn(dec_output)
        return res





# x=torch.rand(1,5,7)
# y=torch.rand(1,5,7)
# Encoder=DecoderLayer(3,7,4,4,8)
# res=Encoder(x,y)
# print(res.size())