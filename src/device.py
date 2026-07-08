"""
Device detection established as a project-wide convention starting
Week 3. CUDA on Colab and Linux and Windows, MPS on the M3 Mac locally,
CPU as last-resort fallback. Every training and inference script imports
get_device from here rather than re-detecting inline.
"""
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
