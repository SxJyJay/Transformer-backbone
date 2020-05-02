class Config():
    def __init__(self):
        self.d_k=64
        self.d_v=64
        self.n_head=8
        self.d_model=512
        self.d_ff=2048
        self.n_layers=6
        self.src_pad_idx=None
        self.trg_pad_idx=None
