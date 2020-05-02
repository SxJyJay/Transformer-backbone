import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy
import math
# import matplotlib.pyplot as plt

def attention(Query,Key,Value,mask=None,dropout=None):
    """
    :param Query:(batch,seq_len,embed_size)
    :param Key:(batch,seq_len,embed_size)
    :param Value:(batch,seq_len,embed_size)
    :param mask:(batch,1,seq_len) for encoder, (batch,seq_len,seq_len) for decoder
    :return:batches of sentences embeddings after attention operation
    """
    dk=Query.size(-1)
    scores=torch.matmul(Query,Key.transpose(-2,-1))/math.sqrt(dk+1e-7)
    if mask is not None:
        mask=mask.unsqueeze(1)
        scores=scores.masked_fill(mask==0,-1e9)
    value_weight=F.softmax(scores,dim=-1)
    if dropout is not None:
        value_weight=dropout(value_weight)

    return torch.matmul(value_weight,Value)


class MultiHeadAttention(nn.Module):
    def __init__(self,head,d_model,d_k,d_v,prob=0.1):
        super(MultiHeadAttention,self).__init__()
        self.d_k=d_k
        self.d_v=d_v
        self.d_model=d_model
        self.head=head
        self.w_qs=nn.Linear(d_model,d_k*head)
        self.w_ks=nn.Linear(d_model,d_k*head)
        self.w_vs=nn.Linear(d_model,d_v*head)
        self.fc=nn.Linear(d_v*head,d_model)
        self.dropout=nn.Dropout(prob)
        self.layernorm=nn.LayerNorm(d_model)


    def forward(self,Q,K,V,mask=None):
        """
        :param Q:(batch,seq_len,d_model)
        :param K: (batch,seq_len,d_model)
        :param V: (batch,seq_len,d_model)
        :param mask: Tensor of shape (seq_len)
        :return: res:(batch,seq_len,d_model)
        """
        batch,Qseq_len=Q.size(0),Q.size(1)
        Kseq_len=K.size(1)
        # print("Q size:{}".format(Q.size()))

        #convert shape (batch,seq_len,head*d_k/d_v) to shape (batch,seq_len,head,d_k/d_v)

        query=self.w_qs(Q)
        query=query.view(batch,Qseq_len,self.head,self.d_k)

        key=self.w_ks(K)
        # print("key size:{}".format(key.size()))
        key=key.view(batch,Kseq_len,self.head,self.d_k)

        value=self.w_vs(V)
        value=value.view(batch,Kseq_len,self.head,self.d_v)

        #convert shape (batch,seq_len,head,d_k/d_v) to shape (batch,head,seq_len,d_k/d_v) so as to pass into attention func
        q,k,v=query.transpose(1,2),key.transpose(1,2),value.transpose(1,2)

        res=attention(q,k,v,mask=mask)

        #conver shape (batch,head,seq_len,d_k/d_v) to shape (batch,seq_len,head,d_k/d_v) so as to merge the last two dims
        res=res.transpose(1,2).contiguous().view(batch,Qseq_len,-1)
        res=self.dropout(self.fc(res))

        res+=Q
        return self.layernorm(res)

class PositionWiseFFN(nn.Module):
    def __init__(self,d_model,d_ff,prob=0.1):
        super(PositionWiseFFN, self).__init__()
        self.ffn=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model),
            nn.Dropout(prob)
        )
        self.layernorm=nn.LayerNorm(d_model)

    def forward(self, x):
        res=self.ffn(x)
        res+=x

        return self.layernorm(res)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,seq_len=5000,prob=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(prob)

        pos=torch.arange(0.,seq_len,1).unsqueeze(1)
        div_term=torch.exp(torch.arange(0.,d_model,2)/d_model*(-math.log(10000.0)))
        # div_term=torch.FloatTensor(np.power(1/10000,np.arange(0.,d_model,2)/d_model))

        PE=torch.zeros(seq_len,d_model)

        # d_model should be an even number, otherwise will raise error in PE[:,1::2]=torch.cos(pos*div_term)
        # thanks to the d_model in the paper is 512
        PE[:,0::2]=torch.sin(pos*div_term)
        PE[:,1::2]=torch.cos(pos*div_term)
        PE=PE.unsqueeze(0)
        self.register_buffer('PE',PE)

    def forward(self, embed):
        # truncate the length of seq_len PE as text_PE (embed shape:[batch_size,seq_len,d_model])
        text_PE=self.PE[:,:embed.size(1)]
        # print("embed size and PE size:")
        # print(embed.size())
        # print(text_PE.size())
        embed=embed+text_PE.detach()
        return self.dropout(embed)


# plt.figure(figsize=(15, 5))
# pe = PositionalEncoding(20,prob=0)
# y = pe.forward(Variable(torch.zeros(1, 100, 20)))
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d"%p for p in [4,5,6,7]])
# plt.show()

# q=torch.rand(1,5,8)
# PE_model=PositionalEncoding(8,5)
# fused_embed=PE_model(q)
# print(fused_embed.size())

#
# q=torch.rand(1,5,7)
# k=torch.rand(1,5,7)
# v=torch.rand(1,5,7)
# mask=torch.Tensor([0,0,1,1,0])
#
# model=MultiHeadAttention(head=2,d_model=7,d_k=3,d_v=3,mask=mask)
# model=PositionWiseFFN(7,14,)
# res=model(q)
# print(res.size())
# print(value_weight)
