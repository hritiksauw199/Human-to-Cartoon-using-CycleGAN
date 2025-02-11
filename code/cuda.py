import torch

print(torch.__version__)
# Check if CUDA (NVIDIA GPU) is available
print(torch.cuda.is_available())

# Check the number of available GPUs
print(torch.cuda.device_count())

# Get the name of the GPU (if available)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Check the current device
print(torch.cuda.current_device())
