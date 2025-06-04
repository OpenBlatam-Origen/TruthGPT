"""
Optimization Core Module for TruthGPT
Advanced performance optimizations and CUDA/Triton kernels
Enhanced with MCTS, parallel training, and advanced optimization techniques
"""

from .cuda_kernels import OptimizedLayerNorm, OptimizedRMSNorm, CUDAOptimizations
from .triton_optimizations import TritonLayerNorm, TritonOptimizations  
from .enhanced_grpo import EnhancedGRPOTrainer, EnhancedGRPOArgs, KalmanFilter
from .mcts_optimization import MCTSOptimizer, MCTSOptimizationArgs, create_mcts_optimizer
from .parallel_training import EnhancedPPOActor, ParallelTrainingConfig, create_parallel_actor
from .experience_buffer import ReplayBuffer, Experience, PrioritizedExperienceReplay, create_experience_buffer
from .advanced_losses import GRPOLoss, EnhancedGRPOLoss, AdversarialLoss, CurriculumLoss, create_loss_function
from .reward_functions import GRPORewardFunction, AdaptiveRewardFunction, MultiObjectiveRewardFunction, create_reward_function
from .advanced_normalization import AdvancedRMSNorm, LlamaRMSNorm, CRMSNorm, AdvancedNormalizationOptimizations, create_advanced_rms_norm, create_llama_rms_norm, create_crms_norm
from .positional_encodings import RotaryEmbedding, LlamaRotaryEmbedding, FixedLlamaRotaryEmbedding, AliBi, SinusoidalPositionalEmbedding, PositionalEncodingOptimizations, create_rotary_embedding, create_llama_rotary_embedding, create_alibi, create_sinusoidal_embedding
from .enhanced_mlp import SwiGLU, GatedMLP, MixtureOfExperts, AdaptiveMLP, EnhancedMLPOptimizations, create_swiglu, create_gated_mlp, create_mixture_of_experts, create_adaptive_mlp
from .rl_pruning import RLPruning, RLPruningAgent, RLPruningOptimizations, create_rl_pruning, create_rl_pruning_agent
from .optimization_registry import OptimizationRegistry, apply_optimizations, get_optimization_config, register_optimization, get_optimization_report
from .advanced_optimization_registry import AdvancedOptimizationConfig, get_advanced_optimization_config, apply_advanced_optimizations, get_advanced_optimization_report

__all__ = [
    'OptimizedLayerNorm',
    'OptimizedRMSNorm',
    'CUDAOptimizations',

    'TritonLayerNorm',
    'TritonOptimizations',
    'EnhancedGRPOTrainer',
    'EnhancedGRPOArgs',
    'KalmanFilter',
    'MCTSOptimizer',
    'MCTSOptimizationArgs',
    'create_mcts_optimizer',
    'EnhancedPPOActor',
    'ParallelTrainingConfig',
    'create_parallel_actor',
    'ReplayBuffer',
    'Experience',
    'PrioritizedExperienceReplay',
    'create_experience_buffer',
    'GRPOLoss',
    'EnhancedGRPOLoss',
    'AdversarialLoss',
    'CurriculumLoss',
    'create_loss_function',
    'GRPORewardFunction',
    'AdaptiveRewardFunction',
    'MultiObjectiveRewardFunction',
    'create_reward_function',
    'AdvancedRMSNorm',
    'LlamaRMSNorm',
    'CRMSNorm',
    'AdvancedNormalizationOptimizations',
    'create_advanced_rms_norm',
    'create_llama_rms_norm',
    'create_crms_norm',
    'RotaryEmbedding',
    'LlamaRotaryEmbedding',
    'FixedLlamaRotaryEmbedding',
    'AliBi',
    'SinusoidalPositionalEmbedding',
    'PositionalEncodingOptimizations',
    'create_rotary_embedding',
    'create_llama_rotary_embedding',
    'create_alibi',
    'create_sinusoidal_embedding',
    'SwiGLU',
    'GatedMLP',
    'MixtureOfExperts',
    'AdaptiveMLP',
    'EnhancedMLPOptimizations',
    'create_swiglu',
    'create_gated_mlp',
    'create_mixture_of_experts',
    'create_adaptive_mlp',
    'RLPruning',
    'RLPruningAgent',
    'RLPruningOptimizations',
    'create_rl_pruning',
    'create_rl_pruning_agent',
    'OptimizationRegistry',
    'apply_optimizations',
    'get_optimization_config',
    'register_optimization',
    'get_optimization_report',
    'AdvancedOptimizationConfig',
    'get_advanced_optimization_config',
    'apply_advanced_optimizations',
    'get_advanced_optimization_report'
]

__version__ = "3.0.0"
