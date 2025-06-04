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
from .optimization_registry import OptimizationRegistry, apply_optimizations, get_optimization_config, register_optimization, get_optimization_report

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
    'OptimizationRegistry',
    'apply_optimizations',
    'get_optimization_config',
    'register_optimization',
    'get_optimization_report'
]

__version__ = "2.0.0"
