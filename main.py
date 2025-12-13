import torch
import os
from config import Config
from dataset import get_cifar10_loaders
from model import CIFARMobileNetV2
from train import train_one_epoch, evaluate
from utils import save_checkpoint
from utils import plot_curves

def main():
    cfg = Config()
    os.makedirs("plots",exist_ok=True)
    
    train_loader, test_loader = get_cifar10_loaders(cfg)

    model = CIFARMobileNetV2(num_classes=cfg.num_classes,
                             width_mult=cfg.width_mult,
                             dropout=cfg.dropout).to(cfg.device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay,
                                nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.T_max
    )
    # Lists for plots
    train_acc_list = [] 
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, cfg.device)
        # Log values
        train_acc_list.append(train_acc)
        val_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(test_loss)
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


    # ---------------- SAVE PLOTS ----------------
    plot_curves(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    save_checkpoint(model, cfg.save_path)
    print("Model saved to:", cfg.save_path)


if __name__ == "__main__":
    main()
