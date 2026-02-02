import torch

def check_gpu_access():
    """Check if GPU is accessible and print device information."""
    print("PyTorch GPU Access Check")
    print("-" * 40)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Print info for each GPU
        for i in range(gpu_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Test GPU allocation
        try:
            x = torch.randn(1000, 1000).cuda()
            print("\n✓ GPU memory allocation test: PASSED")
        except Exception as e:
            print(f"\n✗ GPU memory allocation test: FAILED - {e}")
    else:
        print("No GPU detected. Running on CPU only.")

if __name__ == "__main__":
    check_gpu_access()