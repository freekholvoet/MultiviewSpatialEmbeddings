import torch

def get_default_dtype():
    if torch.backends.mps.is_available():  # Check if MPS is being used
        print("Using MPS backend, setting default dtype to float32.")
        return torch.float32
    else:
        print("Using standard backend, setting default dtype to float64.")
        return torch.float64