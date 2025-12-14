import torch
import os
import wandb

from config import Config,set_seed
from dataset import get_cifar10_loaders
from model import CIFARMobileNetV2
from train import evaluate

from pruning import global_magnitude_prune
from apply_compression import apply_quantization


def main():
    # ---------------- CONFIG ----------------
    set_seed(42)
    cfg = Config()

    # Override from environment (for sweeps)
    cfg.quant_bits = int(os.getenv("QUANT_BITS", cfg.quant_bits))
    cfg.prune_sparsity = float(os.getenv("PRUNE_SPARSITY", cfg.prune_sparsity))

    # ---------------- WANDB INIT ----------------
    wandb.init(
        project="CS6886-Q3-Compression",
        config={
            "quant_bits": cfg.quant_bits,
            "prune_sparsity": cfg.prune_sparsity
        }
    )

    # ---------------- DATA ----------------
    _, test_loader = get_cifar10_loaders(cfg)

    # ---------------- MODEL ----------------
    model = CIFARMobileNetV2(
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout
    )

    model.load_state_dict(
        torch.load("cifar_mobilenetv2.pth", map_location=cfg.device)
    )
    model.to(cfg.device)

    print("Loaded Q1 baseline model")

    # ---------------- PRUNING ----------------
    if cfg.prune_sparsity > 0:
        threshold = global_magnitude_prune(model, cfg.prune_sparsity)
        print(f"Applied pruning | sparsity={cfg.prune_sparsity}, threshold={threshold:.6f}")

    # ---------------- QUANTIZATION ----------------
    if cfg.quant_bits is not None:
        apply_quantization(model, cfg.quant_bits)
        print(f"Applied quantization | {cfg.quant_bits}-bit")

    # ---------------- EVALUATION ----------------
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, cfg.device
    )

    print("\n===== Q2 RESULTS =====")
    print(f"Quant bits     : {cfg.quant_bits}")
    print(f"Prune sparsity : {cfg.prune_sparsity}")
    print(f"Test Accuracy  : {test_acc*100:.2f}%")

    # ---------------- WANDB LOG ----------------
    wandb.log({
        "test_accuracy": test_acc,
        "quant_bits": cfg.quant_bits,
        "prune_sparsity": cfg.prune_sparsity
    })

    wandb.finish()


if __name__ == "__main__":
    main()
