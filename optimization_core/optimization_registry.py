"""
Optimization registry for managing and applying optimizations across TruthGPT variants.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, List
import warnings

from .cuda_kernels import CUDAOptimizations, OptimizedLayerNorm, OptimizedRMSNorm
from .triton_optimizations import TritonOptimizations, TritonLayerNormModule
from .enhanced_grpo import EnhancedGRPOTrainer, EnhancedGRPOArgs

class OptimizationRegistry:
    """Registry for managing optimization strategies."""
    
    _optimizations: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register optimization functions."""
        def decorator(func: Callable):
            cls._optimizations[name] = func
            return func
        return decorator
    
    @classmethod
    def get_optimization(cls, name: str) -> Optional[Callable]:
        """Get optimization function by name."""
        return cls._optimizations.get(name)
    
    @classmethod
    def list_optimizations(cls) -> List[str]:
        """List all available optimizations."""
        return list(cls._optimizations.keys())

@OptimizationRegistry.register('cuda_layer_norm')
def apply_cuda_layer_norm(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Apply CUDA-optimized LayerNorm to model."""
    return CUDAOptimizations.replace_layer_norm(model, config.get('eps', 1e-5))

@OptimizationRegistry.register('cuda_rms_norm')
def apply_cuda_rms_norm(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Apply CUDA-optimized RMSNorm to model."""
    return CUDAOptimizations.replace_rms_norm(model)

@OptimizationRegistry.register('triton_layer_norm')
def apply_triton_layer_norm(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Apply Triton-optimized LayerNorm to model."""
    if not TritonOptimizations.is_triton_available():
        warnings.warn("Triton not available, skipping Triton optimization.")
        return model
    return TritonOptimizations.replace_layer_norm_with_triton(model)

@OptimizationRegistry.register('mixed_precision')
def apply_mixed_precision(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Apply mixed precision optimization."""
    if config.get('enable_fp16', False):
        model = model.half()
    elif config.get('enable_bf16', False) and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
    return model

@OptimizationRegistry.register('gradient_checkpointing')
def apply_gradient_checkpointing(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Apply gradient checkpointing optimization."""
    if config.get('enable_gradient_checkpointing', False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        else:
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
    return model

@OptimizationRegistry.register('torch_compile')
def apply_torch_compile(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Apply torch.compile optimization."""
    if config.get('enable_compilation', False) and hasattr(torch, 'compile'):
        compile_mode = config.get('compile_mode', 'default')
        model = torch.compile(model, mode=compile_mode)
    return model

def apply_optimizations(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Apply multiple optimizations to a model based on configuration."""
    optimized_model = model
    
    optimization_order = config.get('optimization_order', [
        'cuda_layer_norm',
        'cuda_rms_norm', 
        'mixed_precision',
        'gradient_checkpointing',
        'torch_compile'
    ])
    
    for opt_name in optimization_order:
        if config.get(f'enable_{opt_name}', True):
            optimization_func = OptimizationRegistry.get_optimization(opt_name)
            if optimization_func:
                try:
                    optimized_model = optimization_func(optimized_model, config)
                    print(f"✅ Applied {opt_name} optimization")
                except Exception as e:
                    warnings.warn(f"Failed to apply {opt_name} optimization: {e}")
            else:
                warnings.warn(f"Unknown optimization: {opt_name}")
    
    return optimized_model

def get_optimization_config(variant_name: str) -> Dict[str, Any]:
    """Get default optimization configuration for a variant."""
    base_config = {
        'enable_cuda_layer_norm': True,
        'enable_cuda_rms_norm': True,
        'enable_mixed_precision': True,
        'enable_gradient_checkpointing': True,
        'enable_compilation': True,
        'eps': 1e-5,
        'enable_fp16': False,
        'enable_bf16': True,
        'compile_mode': 'default'
    }
    
    variant_configs = {
        'deepseek_v3': {
            **base_config,
            'enable_cuda_rms_norm': True,
            'enable_fp8_quantization': True
        },
        'qwen': {
            **base_config,
            'enable_cuda_rms_norm': True,
            'enable_moe_optimization': True
        },
        'qwen_qwq': {
            **base_config,
            'enable_cuda_rms_norm': True,
            'enable_reasoning_optimization': True
        },
        'viral_clipper': {
            **base_config,
            'enable_multimodal_optimization': True
        },
        'brandkit': {
            **base_config,
            'enable_brand_optimization': True
        },
        'ia_generative': {
            **base_config,
            'enable_generative_optimization': True
        }
    }
    
    return variant_configs.get(variant_name, base_config)

def create_enhanced_grpo_trainer(model: nn.Module, config: Dict[str, Any]) -> EnhancedGRPOTrainer:
    """Create enhanced GRPO trainer with optimizations."""
    args = EnhancedGRPOArgs(
        process_noise=config.get('process_noise', 0.01),
        measurement_noise=config.get('measurement_noise', 0.1),
        kalman_memory_size=config.get('kalman_memory_size', 1000),
        pruning_threshold=config.get('pruning_threshold', 0.1),
        use_amp=config.get('use_amp', True),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        max_grad_norm=config.get('max_grad_norm', 1.0)
    )
    
    return EnhancedGRPOTrainer(model, args)
