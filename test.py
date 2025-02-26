import os
import sys
import subprocess

# Define required packages with correct versions
REQUIRED_PACKAGES = [
    "torch --index-url https://download.pytorch.org/whl/cu121",
    "torchvision --index-url https://download.pytorch.org/whl/cu121",
    "torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "bitsandbytes",
    "transformers",
    "unsloth",
    "torchtriton",
    "scipy",
    "numpy<2.0",
    "scikit-learn",
    "fsspec==2024.9.0"
]


def install_package(package):
    """ Installs a package using pip """
    try:
        print(f"\n🔹 Installing: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-cache-dir"])
        print(f"✅ Successfully installed: {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install: {package}")

def check_python_version():
    """ Ensures the user is running Python 3.10 or 3.11 """
    if sys.version_info.major == 3 and sys.version_info.minor >= 12:
        print("\n🚨 WARNING: You are using Python 3.12, which is NOT fully compatible with Triton and bitsandbytes.")
        print("⚡ Please install Python 3.10 or 3.11 before proceeding.")
        sys.exit(1)

def check_cuda():
    """ Checks if CUDA is available in PyTorch """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ CUDA is available! Using: {torch.cuda.get_device_name(0)}")
        else:
            print("\n🚨 CUDA is NOT available! Check your PyTorch installation and GPU drivers.")
    except ImportError:
        print("\n❌ PyTorch is not installed correctly.")

def test_imports():
    """ Tests if required packages are installed properly """
    try:
        import torch
        import bitsandbytes as bnb
        from unsloth import FastLanguageModel
        from torchtriton import triton
        
        print("\n✅ All necessary packages are installed and working correctly!")
    except ImportError as e:
        print(f"\n❌ Missing package: {e}")

def main():
    """ Main script to install packages and verify setup """
    check_python_version()

    print("\n🚀 Setting up LLM environment...")
    
    for package in REQUIRED_PACKAGES:
        install_package(package)

    check_cuda()
    test_imports()

    print("\n🎉 Setup Complete! You are ready to run your Llama model.")

if __name__ == "__main__":
    main()
