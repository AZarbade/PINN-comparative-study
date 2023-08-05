import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch device: {device}")
print("Everything is setup properly!\n")
print("Have fun...")
