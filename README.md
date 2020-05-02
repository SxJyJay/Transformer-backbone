# Transformer-backbone

The aim of this repository is to help those who want an insight to the details of Transformer realization, without being bothered with data preprocessing.    
The structure of Transformer is illustrated as bellow  
![Transformer](https://camo.githubusercontent.com/88e8f36ce61dedfd2491885b8df2f68c4d1f92f5/687474703a2f2f696d6775722e636f6d2f316b72463252362e706e67)

Thus, we build the network hierarchically. From the top to bottom level is  

Transformer--Fused_Embedding Encoder Decoder--Encoder_layer Decoder_layer--Multiheaded Attention PositionWise_FeedForwardNetwork  

the tree structure is shown as bellow:  

-_Transformer.py_  
>--_Fus_Embeddings(AggregationModel.py)_  
  >>-- _word Embedding Vectors_    
  >>-- _Positional Encoding(Modules.py)_  
  
>--_Encoder(AggregationModel.py)_  
  >>-- _Encoder Layer(Model.py)_  
    &nbsp;&nbsp;&nbsp;-- _MultiHeadedAttention(Modules.py)_  
    &nbsp;&nbsp;&nbsp;-- _PostionWiseFFN(Modules.py)_
  
>--_Decoder(AggregationModel.py)_  
  >>-- _Decoder Layer(Model.py)_  
    &nbsp;&nbsp;&nbsp;-- _MultiHeadedAttention(Modules.py)_  
    &nbsp;&nbsp;&nbsp;-- _PostionWiseFFN(Modules.py)_

# Environment Configuration  
* pytorch 1.1.0  
* python 3.6.8
* torchtext 0.5.0
* tqdm
* dill

#Usage  
##WMT'17 Multimodal Translation: de-en BPE  
1. The byte-pair-encoding has already been processed so that you can focus on the specific structure of Transformer
2.  
