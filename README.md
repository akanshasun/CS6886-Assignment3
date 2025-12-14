
# CS6886-Assignment3
Training MobileNet-v2 on CIFAR-10 and applying model compression techniques to reduce model size while retaining accuracy

Repository structure
project/
├── main.py # Baseline training (Q1)
├── main_q2_eval.py # Compression evaluation (Q2–Q4)
├── run_q3_wandb.py # WandB sweep (Python only)
├── compression.py # Quantization
├── pruning.py # Pruning
├── apply_compression.py # Apply compression
├── model.py # MobileNet-V2
├── dataset.py # CIFAR-10 data
├── train.py # Train/eval loops
├── config.py # Configurations
└── utils.py

## Environment & Dependencies

**Platform:** Google Colab  
**Python:** 3.10+

Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install wandb numpy matplotlib

Seed used was random but now fixed to 42. Expect accuracy delta of 0.5% 

How to Run
Q1 – Baseline Training
python main.py
Produces trained model and accuracy/loss plots.

Q2 – Compression Evaluation
python main_q2_eval.py
Compression controlled via config.py
quant_bits = 8
prune_sparsity = 0.3

Q3 – WandB Sweep
python run_q3_wandb.py

Login to WandB once
