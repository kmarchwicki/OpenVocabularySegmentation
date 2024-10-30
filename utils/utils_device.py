import torch
import os


# Detect device: CUDA, MPS or CPU
def get_device_string():
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        print(
            "Warning: MPS device does not support 'aten::upsample_bicubic2d.out'. Falling back to CPU for this operation. This will be slower than running natively on MPS."
        )
    else:
        device_str = "cpu"
    return device_str


def get_device(device_str):
    return torch.device(device_str)


def set_device(device_str):
    torch.autocast(device_type=device_str, dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
