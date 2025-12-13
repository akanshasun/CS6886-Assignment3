import torch 
class Config:
    data_root = "./data"
    batch_size = 128
    num_workers = 2

    width_mult = 1.0
    dropout = 0.2
    num_classes = 10

    epochs = 150
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    label_smoothing = 0.1

    #Compression
    quant_bits = 8
    prune_sparsity = 0.5

    T_max = epochs
    device = "cuda"
    save_path = "./cifar_mobilenetv2.pth"
