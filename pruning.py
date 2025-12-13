import torch
import torch.nn as nn

def global_magnitude_prune(model, sparsity):
    """
    Global magnitude-based pruning.
    Args:
        model: PyTorch model
        sparsity: fraction of weights to prune (e.g., 0.3 = 30%)
    Returns:
        threshold used for pruning
    """

    assert 0.0 <= sparsity < 1.0, "Sparsity must be in [0,1)"

    # 1. Collect all weights (Conv + Linear only)
    all_weights = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            all_weights.append(m.weight.data.abs().view(-1))

    if len(all_weights) == 0:
        return 0.0

    all_weights = torch.cat(all_weights)

    # 2. Compute global threshold
    num_prune = int(sparsity * all_weights.numel())
    if num_prune == 0:
        return 0.0

    threshold = torch.topk(
        all_weights, num_prune, largest=False
    ).values.max()

    # 3. Apply pruning mask
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            mask = m.weight.data.abs() >= threshold
            m.weight.data.mul_(mask)

    return threshold
