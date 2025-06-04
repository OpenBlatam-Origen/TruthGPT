"""
Optimization Core Module for TruthGPT
Advanced performance optimizations and CUDA/Triton kernels
"""

from .cuda_kernels import OptimizedLayerNorm, CUDAOptimizations
from .triton_optimizations import TritonLayerNorm, TritonOptimizations  
from .enhanced_grpo import EnhancedGRPOTrainer, EnhancedGRPOArgs, KalmanFilter
from .optimization_registry import OptimizationRegistry, apply_optimizations, get_optimization_config

__all__ = [
    'OptimizedLayerNorm',
    'CUDAOptimizations', 
    'TritonLayerNorm',
    'TritonOptimizations',
    'EnhancedGRPOTrainer',
    'EnhancedGRPOArgs',
    'KalmanFilter',
    'OptimizationRegistry',
    'apply_optimizations',
    'get_optimization_config'
]

__version__ = "1.0.0"
