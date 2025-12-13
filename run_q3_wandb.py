# run_q3_wandb.py
import torch
import wandb

from config import Config
from dataset import get_cifar10_loaders
from model import CIFARMobileNetV2
from train import evaluate
from pruning import global_magnitude_prune
from apply_compression import apply_quantization


def compute_weight_compression_ratio(num_bits, sparsity):
    return 32 / (num_bits * (1 - sparsity))


def run_experiment(quant_bits, prune_sparsity):
    cfg = Config()
    cfg.quant_bits = quant_bits
    cfg.prune_sparsity = prune_sparsity

    # ---------- WandB init (ONE RUN) ----------
    wandb.init(
        project="CS6886-Q3-Compression",
        config={
            "quant_bits": quant_bits,
            "prune_sparsity": prune_sparsity
        }
    )

    # ---------- Data ----------
    _, test_loader = get_cifar10_loaders(cfg)

    # ---------- Model ----------
    model = CIFARMobileNetV2(
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout
    )

    model.load_state_dict(
        torch.load("cifar_mobilenetv2.pth", map_location=cfg.device)
    )
    model.to(cfg.device)

    # ---------- Pruning ----------
    if prune_sparsity > 0:
        global_magnitude_prune(model, prune_sparsity)

    # ---------- Quantization ----------
    if quant_bits is not None:
        apply_quantization(model, quant_bits)

    # ---------- Evaluation ----------
    criterion = torch.nn.CrossEntropyLoss()
    _, test_acc = evaluate(model, test_loader, criterion, cfg.device)

    # ---------- Compression metric ----------
    weight_cr = compute_weight_compression_ratio(
        quant_bits, prune_sparsity
    )

    # ---------- Log to WandB ----------
    wandb.log({
        "test_accuracy": test_acc * 100,
        "weight_compression_ratio": weight_cr
    })

    wandb.finish()


if __name__ == "__main__":
    # ===== MANUAL SWEEP =====
    quant_bits_list = [8, 4, 2]
    prune_sparsity_list = [0.0, 0.3, 0.5]

    for b in quant_bits_list:
        for s in prune_sparsity_list:
            print(f"Running: quant_bits={b}, prune_sparsity={s}")
            run_experiment(b, s)
