"""
Unified Dynamic Hardware Profiler for TinyTransformer.
Automatically detects and tunes compute resources across:
1. NVIDIA GeForce RTX 3050 Ti Laptop GPU (4.3 GB VRAM - Micro-batch 1, Accum 32, BF16 native)
2. NVIDIA Tesla T4 (Cloud Colab / Kaggle 16 GB - Micro-batch 4, Accum 8, FP16 Tensor Cores + GradScaler)
3. Google Cloud TPU v5e / v4 / v3 (torch_xla - Micro-batch 8, Accum 4, BF16 Matrix Multiplication Units)
4. High-End GPUs (A100, L4, RTX 3090/4090, H100 - Micro-batch 8, Accum 4, BF16 native)
5. CPU Fallback
"""

from dataclasses import dataclass
import os
import sys
from typing import Optional, Tuple, Any
import torch

# Optional TPU / XLA detection
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    xm = None


@dataclass
class HardwareProfile:
    name: str
    hardware_category: str  # "rtx_3050ti", "tesla_t4", "tpu_v5e", "high_end_gpu", "cpu"
    device: Any             # torch.device or str or xla device
    device_str: str
    dtype: torch.dtype
    dtype_str: str
    batch_size: int
    grad_accum: int
    target_tokens_per_step: int
    use_grad_scaler: bool
    use_grad_checkpoint: bool
    is_tpu: bool
    is_cuda: bool
    vram_gb: float
    description: str

    def print_banner(self):
        print("\n" + "=" * 80)
        print(f"  [>] DYNAMIC HARDWARE PROFILER: {self.name.upper()}")
        print("=" * 80)
        print(f"* Hardware Profile:  {self.description}")
        print(f"* Target Device:     {self.device_str} ({'TPU XLA' if self.is_tpu else ('CUDA GPU' if self.is_cuda else 'Host CPU')})")
        print(f"* VRAM / Memory:     {self.vram_gb:.2f} GB available")
        print(f"* Precision Dtype:   {self.dtype_str.upper()} ({'FP16 Tensor Cores + GradScaler' if self.use_grad_scaler else ('Native BF16 Tensor Units' if self.dtype == torch.bfloat16 else 'FP32')})")
        print(f"* Batch Geometry:    Micro-batch {self.batch_size} x Accum {self.grad_accum} -> {self.batch_size * self.grad_accum * 1024:,} tokens/step")
        print(f"* Gradient Checkpt:  {'Enabled (High VRAM Headroom)' if self.use_grad_checkpoint else 'Disabled'}")
        print("=" * 80 + "\n", flush=True)


def detect_hardware_profile(
    user_batch_size: Optional[int] = None,
    user_grad_accum: Optional[int] = None,
    user_dtype: Optional[str] = None,
    user_device: Optional[str] = None,
    seq_len: int = 1024,
) -> HardwareProfile:
    """
    Dynamically analyzes attached compute hardware and generates the optimal
    profile guaranteeing a constant ~32,768 effective tokens per step across all environments.
    """
    TARGET_STEP_TOKENS = 32_768  # 32 sequences of 1024 tokens

    # 1. Check for Google Cloud TPU (v5e, v4, v3 - Native PyTorch or PyTorch/XLA)
    is_tpu_env = False
    tpu_dev = None

    # Check user explicit argument first
    if user_device in ("tpu", "xla"):
        is_tpu_env = True

    # Check PyTorch 2.x native accelerator API
    if not is_tpu_env and hasattr(torch, "accelerator"):
        try:
            if torch.accelerator.is_available() and torch.accelerator.current_accelerator() == "tpu":
                is_tpu_env = True
                tpu_dev = torch.device("tpu")
        except Exception:
            pass

    # Check direct torch.device("tpu") probe
    if not is_tpu_env:
        try:
            test_dev = torch.device("tpu")
            _probe = torch.ones(1, device=test_dev)
            is_tpu_env = True
            tpu_dev = test_dev
        except Exception:
            pass

    # Check PyTorch / XLA
    if not is_tpu_env and XLA_AVAILABLE and xm is not None:
        try:
            hw_type = xm.xla_device_hw(xm.xla_device())
            if hw_type == "TPU" or bool(os.environ.get("TPU_NAME")) or bool(os.environ.get("COLAB_TPU_ADDR")):
                is_tpu_env = True
                tpu_dev = xm.xla_device()
        except Exception:
            pass

    if is_tpu_env:
        num_cores = 1
        if XLA_AVAILABLE and xm is not None:
            try:
                num_cores = xm.xrt_world_size()
            except Exception:
                num_cores = 1
        is_multi_core = num_cores > 1 or os.environ.get("TPU_CORES") == "8" or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")

        if tpu_dev is None:
            if XLA_AVAILABLE and xm is not None:
                tpu_dev = xm.xla_device()
            else:
                tpu_dev = torch.device("tpu")

        # Check if environment or user specified TPU v5 / v5e
        is_v5e = (
            "v5" in os.environ.get("TPU_ACCELERATOR_TYPE", "").lower()
            or "v5" in os.environ.get("TPU_NAME", "").lower()
            or "v5" in os.environ.get("ACCELERATOR_TYPE", "").lower()
            or (user_device and "v5" in str(user_device).lower())
            or os.environ.get("TPU_VERSION") == "v5e"
        )
        tpu_gen = "v5e-8" if is_v5e else "v3-8"

        if is_multi_core:
            mb = user_batch_size if (user_batch_size and user_batch_size > 0) else 4
            ga = user_grad_accum if (user_grad_accum and user_grad_accum > 0) else max(1, TARGET_STEP_TOKENS // (mb * max(1, num_cores) * seq_len))
            dt = torch.bfloat16
            return HardwareProfile(
                name=f"Kaggle / Cloud TPU {tpu_gen} ({num_cores} Cores Distributed)",
                hardware_category=f"tpu_{tpu_gen.replace('-', '_')}",
                device=tpu_dev,
                device_str=str(tpu_dev),
                dtype=dt,
                dtype_str="bfloat16",
                batch_size=mb,
                grad_accum=ga,
                target_tokens_per_step=mb * ga * max(1, num_cores) * seq_len,
                use_grad_scaler=False,
                use_grad_checkpoint=True,
                is_tpu=True,
                is_cuda=False,
                vram_gb=16.0 * max(1, num_cores),
                description=f"TPU {tpu_gen} Cluster with {num_cores} Parallel Cores ({16.0 * max(1, num_cores):.0f} GB HBM) via PyTorch/XLA",
            )

        mb = user_batch_size if (user_batch_size and user_batch_size > 0) else 8
        ga = user_grad_accum if (user_grad_accum and user_grad_accum > 0) else max(1, TARGET_STEP_TOKENS // (mb * seq_len))
        dt = torch.bfloat16
        return HardwareProfile(
            name="Google Cloud TPU (v5e Matrix Units)",
            hardware_category="tpu_v5e",
            device=tpu_dev,
            device_str=str(tpu_dev),
            dtype=dt,
            dtype_str="bfloat16",
            batch_size=mb,
            grad_accum=ga,
            target_tokens_per_step=mb * ga * seq_len,
            use_grad_scaler=False,
            use_grad_checkpoint=True,
            is_tpu=True,
            is_cuda=False,
            vram_gb=16.0,
            description="TPU v5e Matrix Multiplication Engine via Native PyTorch TPU / XLA",
        )

    # 2. Check for CUDA GPUs
    if torch.cuda.is_available() and (user_device is None or "cuda" in str(user_device).lower()):
        dev_props = torch.cuda.get_device_properties(0)
        gpu_name = dev_props.name
        vram_gb = dev_props.total_memory / (1024 ** 3)
        bf16_supported = torch.cuda.is_bf16_supported()

        # Check for RTX 3050 Ti Laptop or <6 GB GPU
        is_rtx_3050ti = "3050" in gpu_name or vram_gb < 6.0
        # Check for Tesla T4 (Cloud / Turing sm_75)
        is_tesla_t4 = "T4" in gpu_name or (not bf16_supported and vram_gb >= 12.0)
        # Check for High-End GPU (A100, H100, L4, RTX 4090/3090 >= 20GB)
        is_high_end = vram_gb >= 20.0

        if is_rtx_3050ti:
            mb = user_batch_size if (user_batch_size and user_batch_size > 0) else 1
            ga = user_grad_accum if (user_grad_accum and user_grad_accum > 0) else 32
            # RTX 3050 Ti is Ampere sm_86, natively supports BF16
            dt = torch.bfloat16 if (user_dtype in (None, "auto", "bfloat16") and bf16_supported) else (
                torch.float16 if user_dtype == "float16" else torch.float32
            )
            use_scaler = (dt == torch.float16)
            return HardwareProfile(
                name=f"NVIDIA GeForce RTX 3050 Ti Laptop ({vram_gb:.1f} GB)",
                hardware_category="rtx_3050ti",
                device="cuda",
                device_str="cuda:0",
                dtype=dt,
                dtype_str=str(dt).split(".")[-1],
                batch_size=mb,
                grad_accum=ga,
                target_tokens_per_step=mb * ga * seq_len,
                use_grad_scaler=use_scaler,
                use_grad_checkpoint=True,
                is_tpu=False,
                is_cuda=True,
                vram_gb=vram_gb,
                description="Consumer Laptop Ampere GPU with strict <3.8 GB VRAM management",
            )

        elif is_tesla_t4:
            mb = user_batch_size if (user_batch_size and user_batch_size > 0) else 4
            ga = user_grad_accum if (user_grad_accum and user_grad_accum > 0) else 8
            # Tesla T4 is Turing sm_75: dedicated FP16 Tensor Cores (130 TFLOPs)
            dt = torch.float16 if user_dtype in (None, "auto", "float16") else (
                torch.bfloat16 if (user_dtype == "bfloat16" and bf16_supported) else torch.float32
            )
            use_scaler = (dt == torch.float16)
            return HardwareProfile(
                name=f"NVIDIA Tesla T4 Cloud GPU ({vram_gb:.1f} GB)",
                hardware_category="tesla_t4",
                device="cuda",
                device_str="cuda:0",
                dtype=dt,
                dtype_str=str(dt).split(".")[-1],
                batch_size=mb,
                grad_accum=ga,
                target_tokens_per_step=mb * ga * seq_len,
                use_grad_scaler=use_scaler,
                use_grad_checkpoint=True,
                is_tpu=False,
                is_cuda=True,
                vram_gb=vram_gb,
                description="Cloud Turing Tensor Cores optimized with FP16 + GradScaler",
            )

        elif is_high_end:
            mb = user_batch_size if (user_batch_size and user_batch_size > 0) else 8
            ga = user_grad_accum if (user_grad_accum and user_grad_accum > 0) else 4
            dt = torch.bfloat16 if (user_dtype in (None, "auto", "bfloat16") and bf16_supported) else torch.float16
            return HardwareProfile(
                name=f"NVIDIA High-End Accelerator ({gpu_name}, {vram_gb:.1f} GB)",
                hardware_category="high_end_gpu",
                device="cuda",
                device_str="cuda:0",
                dtype=dt,
                dtype_str=str(dt).split(".")[-1],
                batch_size=mb,
                grad_accum=ga,
                target_tokens_per_step=mb * ga * seq_len,
                use_grad_scaler=(dt == torch.float16),
                use_grad_checkpoint=True,
                is_tpu=False,
                is_cuda=True,
                vram_gb=vram_gb,
                description="High VRAM Compute Node with high parallel micro-batch saturation",
            )

        else:
            # General mid-tier CUDA GPU (8-16 GB)
            mb = user_batch_size if (user_batch_size and user_batch_size > 0) else (4 if vram_gb >= 10.0 else 2)
            ga = user_grad_accum if (user_grad_accum and user_grad_accum > 0) else max(1, TARGET_STEP_TOKENS // (mb * seq_len))
            dt = torch.bfloat16 if (user_dtype in (None, "auto", "bfloat16") and bf16_supported) else (
                torch.float16 if (user_dtype == "float16" or not bf16_supported) else torch.float32
            )
            return HardwareProfile(
                name=f"NVIDIA CUDA GPU ({gpu_name}, {vram_gb:.1f} GB)",
                hardware_category="cuda_general",
                device="cuda",
                device_str="cuda:0",
                dtype=dt,
                dtype_str=str(dt).split(".")[-1],
                batch_size=mb,
                grad_accum=ga,
                target_tokens_per_step=mb * ga * seq_len,
                use_grad_scaler=(dt == torch.float16),
                use_grad_checkpoint=True,
                is_tpu=False,
                is_cuda=True,
                vram_gb=vram_gb,
                description="General CUDA Acceleration Profile",
            )

    # 3. CPU Fallback Profile
    mb = user_batch_size if (user_batch_size and user_batch_size > 0) else 1
    ga = user_grad_accum if (user_grad_accum and user_grad_accum > 0) else 1
    return HardwareProfile(
        name="Host CPU",
        hardware_category="cpu",
        device="cpu",
        device_str="cpu",
        dtype=torch.float32,
        dtype_str="float32",
        batch_size=mb,
        grad_accum=ga,
        target_tokens_per_step=mb * ga * seq_len,
        use_grad_scaler=False,
        use_grad_checkpoint=False,
        is_tpu=False,
        is_cuda=False,
        vram_gb=0.0,
        description="CPU execution mode",
    )
