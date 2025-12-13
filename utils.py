import torch
import matplotlib.pyplot as plt
def accuracy(logits, targets):
    _, preds = logits.max(1)
    return (preds == targets).float().mean().item()

def plot_curves(train_acc_list, val_acc_list, train_loss_list, val_loss_list):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.savefig("plots/q1_curves.png")
    print("Saved curves → plots/q1_curves.png")

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
