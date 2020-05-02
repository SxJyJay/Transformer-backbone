import torch
import torch.nn as nn
from AggregationModel import *

class Transformer(nn.Module):
    def __init__(
            self,n_src_vocab,n_trg_vocab,d_model,d_k,d_v,d_ff,n_head,n_layers,src_pad_idx,trg_pad_idx):
        super(Transformer, self).__init__()

        self.src_fusembed=Fus_Embeddings(n_src_embeddings=n_src_vocab,d_model=d_model)
        self.trg_fusembed=Fus_Embeddings(n_src_embeddings=n_trg_vocab,d_model=d_model)
        self.encoder=Encoder(head=n_head,d_model=d_model,d_k=d_k,d_v=d_v,d_ff=d_ff,
                             n_layers=n_layers)
        self.decoder=Decoder(head=n_head,d_model=d_model,d_k=d_k,d_v=d_v,d_ff=d_ff,n_layers=n_layers)
        self.feat_trgword_proj=nn.Linear(in_features=d_model, out_features=n_trg_vocab)
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx

    def forward(self, src_seq_idx,trg_seq_idx):
        '''
        :param src_seq_idx:[batch,seq_len] index of every line must be torch.LongTensor
        :param trg_seq_idx:[batch,seq_len_trg]
        :param enc_attn_mask:[seq_len]
        :param dec_attn_mask:[seq_len_trg]
        :param enc_dec_mask:[]
        :return:
        '''
        enc_attn_mask=get_pad_mask(src_seq_idx,self.src_pad_idx)
       # print(get_pad_mask(trg_seq_idx,self.trg_pad_idx))
       # print(get_subsequent_mask(trg_seq_idx))
        dec_attn_mask=get_pad_mask(trg_seq_idx,self.trg_pad_idx) & get_subsequent_mask(trg_seq_idx)    
        

        enc_input=self.src_fusembed(src_seq_idx)
        # print("enc_input size:{}".format(enc_input.size()))
        enc_output=self.encoder(enc_input,enc_attn_mask)
        # print("enc_output size:{}".format(enc_output.size()))
        dec_input=self.trg_fusembed(trg_seq_idx)
        # print("dec_input size:{}".format(dec_input.size()))
        dec_output=self.decoder(enc_output,dec_input,dec_attn_mask,enc_attn_mask)

        scores=self.feat_trgword_proj(dec_output)
        return scores.view(-1,scores.size(2))

# transformer=Transformer(
#     n_src_vocab=10,n_trg_vocab=12,d_model=6,d_k=4,d_v=4,d_ff=12,n_head=2,n_layers=2,src_pad_idx=0,trg_pad_idx=0
# )
#
# src_seq_idx=torch.LongTensor([[0,1,2],[2,2,0]])
# trg_seq_idx=torch.LongTensor([[0,1,2,3],[2,3,2,0]])
# # print(trg_seq_idx.size())
#
# scores=transformer(src_seq_idx,trg_seq_idx)
# print(scores.size())





