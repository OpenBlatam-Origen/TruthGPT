"""
Enhanced GRPO training with Kalman filtering and advanced optimizations.
Integrated from kf-grpo-train.py and GRPO.py optimization files.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Union, Tuple

def compute_reward_function(outputs, targets, reward_config=None):
    """Advanced reward computation with multiple reward signals."""
    if reward_config is None:
        reward_config = {
            'accuracy_weight': 0.4,
            'fluency_weight': 0.3,
            'relevance_weight': 0.2,
            'safety_weight': 0.1
        }
    
    batch_size, seq_len = outputs.shape[:2]
    
    accuracy_reward = torch.cosine_similarity(
        outputs.view(batch_size, -1), 
        targets.view(batch_size, -1), 
        dim=1
    ).unsqueeze(1).expand(-1, seq_len)
    
    fluency_reward = torch.exp(-torch.var(outputs, dim=-1))
    
    relevance_reward = torch.sigmoid(torch.mean(outputs * targets, dim=-1))
    
    safety_reward = torch.clamp(1.0 - torch.abs(outputs).mean(dim=-1), 0.0, 1.0)
    
    total_reward = (
        reward_config['accuracy_weight'] * accuracy_reward +
        reward_config['fluency_weight'] * fluency_reward +
        reward_config['relevance_weight'] * relevance_reward +
        reward_config['safety_weight'] * safety_reward
    )
    
    return total_reward
import time
import warnings

@dataclass
class EnhancedGRPOArgs:
    """Enhanced GRPO training arguments with advanced optimizations."""
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate for optimizer"})
    beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    eps: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for optimizer"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm for clipping"})
    warmup_steps: int = field(default=100, metadata={"help": "Number of warmup steps"})
    total_steps: int = field(default=1000, metadata={"help": "Total training steps"})
    reward_scaling: float = field(default=1.0, metadata={"help": "Reward scaling factor"})
    kl_penalty: float = field(default=0.1, metadata={"help": "KL penalty coefficient"})
    entropy_bonus: float = field(default=0.01, metadata={"help": "Entropy bonus coefficient"})
    advantage_normalization: bool = field(default=True, metadata={"help": "Whether to normalize advantages"})
    use_kalman_filter: bool = field(default=True, metadata={"help": "Whether to use Kalman filtering"})
    kalman_process_noise: float = field(default=0.01, metadata={"help": "Kalman filter process noise"})
    kalman_measurement_noise: float = field(default=0.1, metadata={"help": "Kalman filter measurement noise"})
    
    use_enhanced_loss: bool = field(default=True, metadata={"help": "Use enhanced loss computation"})
    use_dynamic_clipping: bool = field(default=True, metadata={"help": "Use dynamic clipping"})
    
    process_noise: float = field(default=0.01, metadata={"help": "Process noise covariance (Q)"})
    measurement_noise: float = field(default=0.1, metadata={"help": "Measurement noise covariance (R)"})
    kalman_memory_size: int = field(default=1000, metadata={"help": "Size of Kalman filter memory buffer"})
    
    pruning_threshold: float = field(default=0.1, metadata={"help": "Threshold for sample pruning"})
    pruning_alpha: float = field(default=0.5, metadata={"help": "Alpha for dynamic K adjustment"})
    k_min: int = field(default=1, metadata={"help": "Minimum K value"})
    k_max: int = field(default=10, metadata={"help": "Maximum K value"})
    
    policy_clip_delta: float = field(default=0.2, metadata={"help": "Policy clipping delta"})
    length_penalty_lambda: float = field(default=0.1, metadata={"help": "Length penalty coefficient"})
    max_length: int = field(default=1000, metadata={"help": "Maximum sequence length for normalization"})
    
    use_amp: bool = field(default=True, metadata={"help": "Use automatic mixed precision"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Ratio of warmup steps"})

class KalmanFilter:
    def __init__(self, process_noise: float, measurement_noise: float, memory_size: int = 1000):
        self.Q = process_noise
        self.R = measurement_noise
        self.mu = 0.0
        self.P = 1.0
        self.memory = []
        self.memory_size = memory_size
        self.momentum = 0.9
        self.velocity = 0.0
        
    def update(self, measurement: float) -> float:
        mu_pred = self.mu + self.momentum * self.velocity
        P_pred = self.P + self.Q
        
        K = P_pred / (P_pred + self.R)
        innovation = measurement - mu_pred
        self.mu = mu_pred + K * innovation
        self.P = (1 - K) * P_pred + self.Q
        
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * innovation
        
        self.memory.append(measurement)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
            
        return self.mu
    
    def get_statistics(self) -> Tuple[float, float]:
        """Get mean and standard deviation of recent measurements with exponential weighting."""
        if not self.memory:
            return 0.0, 1.0
        
        weights = np.exp(np.linspace(-1, 0, len(self.memory)))
        weights /= weights.sum()
        
        weighted_mean = np.average(self.memory, weights=weights)
        weighted_std = np.sqrt(np.average((np.array(self.memory) - weighted_mean) ** 2, weights=weights))
        
        return weighted_mean, weighted_std

def get_token_log_probs(model, input_ids, attention_mask):
    """Compute log-probabilities of tokens under the given model."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    token_logp = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return token_logp

def compute_probability_ratio(curr_logp, old_logp):
    """Compute the probability ratio between current and old policies."""
    return torch.exp(curr_logp - old_logp)

def compute_clipped_ratio(ratio, epsilon):
    """Clip the ratio to [1-epsilon, 1+epsilon]"""
    return torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

def compute_kl_penalty(curr_logp, ref_logp):
    """Compute per-token KL divergence penalty term."""
    diff = ref_logp - curr_logp
    return torch.exp(diff) - diff - 1

def compute_surrogate_advantage(ratio, clipped_ratio, advantages):
    """Compute the surrogate advantage loss per token using PPO-style clipping."""
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    return torch.min(loss1, loss2)

def compute_per_token_loss(ratio, clipped_ratio, advantages, kl_penalty, beta):
    """Combine surrogate advantage and KL penalty per token."""
    adv_loss = compute_surrogate_advantage(ratio, clipped_ratio, advantages)
    return - (adv_loss - beta * kl_penalty)

def compute_enhanced_grpo_loss(
    current_model,
    old_model,
    ref_model,
    input_ids,
    attention_mask,
    advantages,
    beta=1.0,
    epsilon=0.2,
):
    """Compute the enhanced GRPO loss for a batch of sequences."""
    curr_logp = get_token_log_probs(current_model, input_ids, attention_mask)
    with torch.no_grad():
        old_logp = get_token_log_probs(old_model, input_ids, attention_mask)
        ref_logp = get_token_log_probs(ref_model, input_ids, attention_mask)

    ratio = compute_probability_ratio(curr_logp, old_logp)
    clipped = compute_clipped_ratio(ratio, epsilon)
    kl_penalty = compute_kl_penalty(curr_logp, ref_logp)

    per_token_loss = compute_per_token_loss(ratio, clipped, advantages, kl_penalty, beta)

    mask = attention_mask.float()
    lengths = mask.sum(dim=1).clamp(min=1)
    loss_per_seq = (per_token_loss * mask).sum(dim=1) / lengths
    return loss_per_seq.mean()

class EnhancedGRPOTrainer:
    """Enhanced GRPO trainer with Kalman filtering and advanced optimizations."""
    
    def __init__(self, model, args: EnhancedGRPOArgs):
        self.model = model
        self.args = args
        
        self.kf = KalmanFilter(
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            memory_size=args.kalman_memory_size
        )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=1e-4 * 0.1
        )
        
        self._metrics = {
            "kalman_reward": [],
            "pruned_samples": [],
            "length_penalty": [],
            "learning_rate": [],
            "gradient_norm": [],
            "memory_usage": [],
            "throughput": [],
            "gpu_utilization": []
        }
        
        if args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def compute_enhanced_loss(self, model, inputs, return_outputs=False):
        """Enhanced loss computation with Kalman filtering and optimizations."""
        if hasattr(inputs, 'input_ids'):
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
        else:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
        
        rewards = self._get_rewards(inputs)
        filtered_rewards = torch.tensor([
            self.kf.update(r.item()) for r in rewards
        ], device=rewards.device)
        
        mean_reward, std_reward = self.kf.get_statistics()
        adaptive_threshold = self.args.pruning_threshold * (1 + std_reward)
        
        advantages = self._compute_advantages(filtered_rewards)
        pruned_mask = torch.abs(advantages) > adaptive_threshold
        pruned_advantages = advantages[pruned_mask]
        
        pruning_ratio = pruned_mask.float().mean()
        k_next = torch.clamp(
            self.args.pruning_alpha * pruning_ratio * (1 + self.kf.velocity),
            self.args.k_min,
            self.args.k_max
        )
        
        sequence_lengths = self._get_sequence_lengths(inputs)
        length_penalties = self.args.length_penalty_lambda * (
            sequence_lengths / self.args.max_length
        ) * (1 + std_reward)
        penalized_rewards = filtered_rewards - length_penalties
        
        self._update_metrics(
            filtered_rewards,
            pruning_ratio,
            length_penalties,
            self._calculate_dynamic_learning_rate(filtered_rewards, pruning_ratio)
        )
        
        base_loss = F.cross_entropy(
            model(input_ids, attention_mask=attention_mask).logits.view(-1, model.config.vocab_size),
            input_ids.view(-1),
            ignore_index=-100
        )
        
        final_loss = base_loss + self._compute_additional_losses(
            penalized_rewards,
            pruned_advantages,
            k_next
        )
        
        return final_loss
    
    def _get_rewards(self, inputs):
        """Advanced reward computation with multi-objective optimization."""
        if hasattr(inputs, 'input_ids'):
            input_ids = inputs.input_ids
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
        else:
            input_ids = inputs['input_ids']
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                try:
                    model_outputs = self.model(input_ids)
                    if hasattr(model_outputs, 'logits'):
                        outputs = model_outputs.logits
                    else:
                        outputs = model_outputs
                except:
                    outputs = torch.randn(batch_size, seq_len, 512, device=input_ids.device)
            else:
                outputs = torch.randn(batch_size, seq_len, 512, device=input_ids.device)
        
        targets = torch.nn.functional.one_hot(input_ids, num_classes=outputs.size(-1)).float()
        
        reward_config = {
            'accuracy_weight': 0.4,
            'fluency_weight': 0.25,
            'relevance_weight': 0.2,
            'safety_weight': 0.1,
            'diversity_weight': 0.05
        }
        
        return compute_reward_function(outputs, targets, reward_config)
    
    def _compute_advantages(self, rewards):
        """Compute advantages from rewards."""
        return rewards - rewards.mean()
    
    def _get_sequence_lengths(self, inputs):
        """Get sequence lengths from inputs."""
        if hasattr(inputs, 'attention_mask'):
            return inputs.attention_mask.sum(dim=1).float()
        else:
            return inputs['attention_mask'].sum(dim=1).float()
    
    def _compute_additional_losses(self, rewards, advantages, k_next):
        """Compute advanced additional loss terms with regularization."""
        reward_loss = 0.01 * rewards.mean()
        
        advantage_var_loss = 0.005 * torch.var(advantages) if len(advantages) > 1 else torch.tensor(0.0)
        
        k_regularization = 0.001 * torch.abs(k_next - self.args.k_min - (self.args.k_max - self.args.k_min) / 2)
        
        entropy_loss = -self.args.entropy_bonus * torch.mean(torch.log(torch.abs(rewards) + 1e-8))
        
        total_additional_loss = reward_loss + advantage_var_loss + k_regularization + entropy_loss
        
        return total_additional_loss
    
    def _calculate_dynamic_learning_rate(self, rewards, pruning_ratio):
        """Calculate advanced dynamic learning rate with momentum and adaptive scaling."""
        base_lr = self.args.learning_rate
        
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item() if len(rewards) > 1 else 0.1
        reward_factor = torch.tanh(torch.tensor(reward_mean / (reward_std + 1e-8))).item()
        
        pruning_factor = pruning_ratio.item() if hasattr(pruning_ratio, 'item') else pruning_ratio
        pruning_adjustment = torch.exp(torch.tensor(-pruning_factor * 2.0)).item()
        
        velocity_factor = 1.0 + 0.1 * torch.tanh(torch.tensor(self.kf.velocity, dtype=torch.float32)).item()
        
        dynamic_lr = base_lr * (1.0 + 0.5 * reward_factor) * pruning_adjustment * velocity_factor
        
        min_lr = base_lr * 0.01
        max_lr = base_lr * 10.0
        
        return max(min_lr, min(dynamic_lr, max_lr))
    
    def _update_metrics(self, rewards, pruning_ratio, length_penalties, lr):
        """Update training metrics."""
        self._metrics["kalman_reward"].append(rewards.mean().item())
        self._metrics["pruned_samples"].append(pruning_ratio.item())
        self._metrics["length_penalty"].append(length_penalties.mean().item())
        self._metrics["learning_rate"].append(lr)
        
        if torch.cuda.is_available():
            self._metrics["memory_usage"].append(torch.cuda.memory_allocated() / 1024**2)
            self._metrics["gpu_utilization"].append(torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0)
    
    def get_metrics(self):
        """Get current training metrics."""
        return {k: np.mean(v) if v else 0 for k, v in self._metrics.items()}
    
    def clear_metrics(self):
        """Clear accumulated metrics."""
        for v in self._metrics.values():
            v.clear()
